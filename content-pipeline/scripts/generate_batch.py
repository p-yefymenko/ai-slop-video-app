#!/usr/bin/env python3
"""content:frames and content:generate, scene by scene.

frames: previs (the 3D scene and its checks), then what an image model needs to
        draw the scene's start still in one pass: the shot's 3D layout with each
        person, prop and landmark in it in their mesh's own colors, and the words
        (``describe.layout_prompt``).
        The still is drawn by hand in Gemini for now and saved where the stage
        says (``start_still_path``).
video:  the scene's voice track (``voice``), then the take:
        - an on-screen speaker whose face the shot shows: InfiniteTalk from the
          start still and the voice (``describe.talking_speaker``), the voice tied
          to the speaker's place in the still and silence to everyone else's;
        - anything else: Wan VACE over the whole take, the voice laid under it;
        then RIFE and 24 fps, cut to the scene's exact length. When every scene
        of an episode has its clip, they are joined and captioned.

Beside each clip, ``inputs/<name>/`` holds only what a review needs: its log and,
for a talking take with anyone else in it, a collage of whose the voice is. What
a stage keeps for itself is in ``work/`` under it.

Everything a model is given comes from the script, the 3D scene, and the
character references; the words from ``describe``, the settings from ``wan``.
Each output is made once, at a seed fixed by its scene, and logs what it was
made from; an output whose inputs changed (the words, the size, the still under
a clip) is made again. Nothing checks the outputs; they are reviewed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

import wan
from captions import caption_episode
from collage import over_nobody, write_collage
from comfy import (
    format_duration,
    free_comfy_models,
    inputs_dir,
    load_dotenv,
    present,
    require_comfy,
    run_graph,
    stable_seed,
    stage_input,
    write_log,
)
from describe import layout_prompt, plain_still_prompt, reference_people, talking_speaker, video_prompt
from ffmpeg_tools import concat_videos
from pipeline_paths import (
    clip_path,
    discover_show_scripts,
    episode_video_path,
    manifest_path,
    start_still_path,
    voice_track_path,
)
from previs import current_previs, load_observation, previs_episode
from shot_inputs import colored_layout, reference_sheet, start_frame, view_degrees, write_depth_video
from voice import track_seconds, write_voice_track
from world import camera_at, load_show, reference_path, settings

# A 720p talking take can take well over an hour. Past these a run has hung.
TAKE_TIMEOUT_SECONDS = 4 * 3600
RIFE_TIMEOUT_SECONDS = 1800


def _ids(show: dict, episode: dict, scene: dict) -> tuple[str, int, int]:
    return show["id"], episode["episodeNumber"], scene["sceneNumber"]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def made_from(dest: Path, expected: dict) -> bool:
    """Whether ``dest`` exists and its log says it was made from exactly ``expected``."""
    log = inputs_dir(dest) / "log.json"
    if not present(dest) or not log.is_file():
        return False
    recorded = json.loads(log.read_text(encoding="utf-8"))
    return all(recorded.get(key) == value for key, value in expected.items())


def require_previs(show: dict, episode: dict, scene: dict) -> dict:
    """The scene's start observation, after previs has checked the script against its 3D scene."""
    errors = current_previs(show, episode, scene)
    if errors is None:
        errors, _written = previs_episode(show, episode, scene["sceneNumber"])
    if errors:
        raise SystemExit("The script does not match its 3D scene:\n- " + "\n- ".join(errors))
    return load_observation(*_ids(show, episode, scene), "start")


def shot_inputs(show: dict, episode: dict, scene: dict, people: list[str], size: str, length: int, folder: Path,
                pace: float = 1.0) -> tuple[str, str | None]:
    """The depth video for a take of ``length`` frames (the timeline at ``pace``, see
    ``write_depth_video``) and the reference sheet of ``people``, staged for ComfyUI."""
    width, height = wan.SIZES[size]
    depth = write_depth_video(show, episode, scene, width, height, length, wan.FPS, folder / "depth.mp4", pace=pace)
    photos = [reference_path(show, cid) for cid in people]
    missing = [str(path) for path in photos if not present(path)]
    if missing:
        raise SystemExit("Missing reference photos (run `pnpm run content:plates`):\n- " + "\n- ".join(missing))
    sheet = reference_sheet(photos, width, height, folder / "reference.png") if photos else None
    return stage_input(depth, "wan_depth"), stage_input(sheet, "wan_reference") if sheet else None


def print_prompts(show: dict, episode: dict, scene: dict, size: str) -> None:
    """Every prompt a frames and a video run of the scene would send, from the same code, and
    nothing generated: the still's (to the image model) and the take's. Read before any run."""
    start = require_previs(show, episode, scene)
    _entities, frame, words = still_request(show, episode, scene, size)
    print(f"  Still ({'with the layout' if frame.entity.any() else 'words only, no layout'}):\n    {words}", flush=True)
    speaker = talking_speaker(scene)
    print(f"  Take ({'InfiniteTalk, ' + speaker if speaker else 'VACE'}):\n    {video_prompt(show, episode, scene, start, speaker)}", flush=True)


# The layout is rendered this many times the take's size: the image model draws from it, and detail helps.
STILL_LAYOUT_SCALE = 2


def still_request_dir(still: Path) -> Path:
    """Where a start still's request to the image model is: the layout to upload and its words."""
    return still.parent / "to_draw" / still.stem


def still_request(show: dict, episode: dict, scene: dict, size: str):
    """The start frame at ``STILL_LAYOUT_SCALE`` times the take's size and the words for the image
    model: (the entities, the render, the prompt). When none of the 3D scene is in the frame (only
    the backdrop, which the layout has no shape for) the layout would be a blank, so the words go
    alone (``plain_still_prompt``)."""
    start = require_previs(show, episode, scene)
    width, height = (side * STILL_LAYOUT_SCALE for side in wan.SIZES[size])
    people, entities, frame = start_frame(show, episode, scene, width, height)
    if not frame.entity.any():
        return entities, frame, plain_still_prompt(show, scene, start, (width, height))
    camera = camera_at(scene, float(scene["timeRangeSeconds"][0]))
    views = {cid: view_degrees(next(entity for entity in entities if entity.id == cid), camera) for cid in people}
    return entities, frame, layout_prompt(show, scene, start, views, (width, height))


def write_still_request(show: dict, episode: dict, scene: dict, size: str) -> Path:
    """What an image model needs to draw the scene's start still in one pass, for a person to give
    it (the still is drawn by hand in Gemini for now; nothing here is generated): ``layout.png``
    (the shot's 3D layout, everyone and everything in it in their mesh's own colors,
    ``colored_layout``; none when it would be a blank) and ``prompt.txt``
    (``describe.layout_prompt``). Beside the full-length reference photos the model drew everyone
    full length, facing the camera, and lost the layout's framing; from the layout alone it kept
    it. The picture it draws is saved as the still (``start_still_path``)."""
    still = start_still_path(*_ids(show, episode, scene))
    folder = still_request_dir(still)
    entities, frame, prompt = still_request(show, episode, scene, size)
    if folder.exists():
        for old in folder.iterdir():
            old.unlink()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    if not frame.entity.any():
        print(f"  Give the words in {folder / 'prompt.txt'} alone (none of the 3D scene is in the frame); save the picture as {still}", flush=True)
        return folder
    colored = [number + 1 for number, entity in enumerate(entities) if entity.mesh.colors is not None]
    Image.fromarray(colored_layout(frame.clay, frame.albedo, frame.entity, colored)).save(folder / "layout.png")
    print(f"  Upload {folder / 'layout.png'} with the words in prompt.txt; save the picture as {still}", flush=True)
    return folder


# What an image model's download may be saved as, beside the still's own name.
STILL_FORMATS = (".png", ".jpg", ".jpeg", ".webp")


def saved_still(still: Path) -> Path:
    """The still as saved: under its own name in any of ``STILL_FORMATS`` (a download comes in the
    image model's format); ``still`` itself when there is none yet. Two of them is an error."""
    found = [still.with_suffix(suffix) for suffix in STILL_FORMATS if still.with_suffix(suffix).is_file()]
    if len(found) > 1:
        raise SystemExit(f"Keep one start still, not {', '.join(path.name for path in found)}")
    return found[0] if found else still


def fitted_still(still: Path, size: str, folder: Path) -> Path:
    """The still at the take's size: scaled to cover it and cut to it about the middle (a picture
    drawn by hand comes back at the image model's own size). Saved in ``folder``."""
    width, height = wan.SIZES[size]
    picture = Image.open(still).convert("RGB")
    scale = max(width / picture.width, height / picture.height)
    picture = picture.resize((max(width, round(picture.width * scale)), max(height, round(picture.height * scale))), Image.Resampling.LANCZOS)
    left, top = (picture.width - width) // 2, (picture.height - height) // 2
    dest = folder / "start.png"
    picture.crop((left, top, left + width, top + height)).save(dest)
    return dest


def generate_voice(show: dict, episode: dict, scene: dict) -> Path:
    """The scene's voice track, remade only when its lines change. Its length is the scene's (``voice.lay_out``)."""
    dest = voice_track_path(*_ids(show, episode, scene))
    expected = {"lines": scene.get("lines") or [], "timeRangeSeconds": scene["timeRangeSeconds"], "length": "the scene's, or as long as its speech"}
    if made_from(dest, expected):
        return dest
    if scene.get("lines"):
        print("  Voice track...", flush=True)
        free_comfy_models()
    problems = write_voice_track(show, episode, scene, dest)
    for problem in problems:
        print(f"  Voice: {problem}", flush=True)
    write_log(dest, {**expected, "problems": problems})
    return dest


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *args], check=True)


def talk_masks(show: dict, episode: dict, scene: dict, speaker: str, size: str, folder: Path) -> tuple[Path, Path] | None:
    """The speaker's place in the still and everyone else's together, when the speaker and anyone
    else are in it; else None (one voice, no mask). The still is drawn from the start frame's
    layout, so each person is where their mesh is in that frame (the nearest mesh at each pixel)."""
    people, entities, frame = start_frame(show, episode, scene, *wan.SIZES[size])
    others = [cid for cid in people if cid != speaker]
    if speaker not in people or not others:
        return None
    numbers = {entity.id: number + 1 for number, entity in enumerate(entities)}
    share = {cid: np.where(frame.entity == numbers[cid], 255, 0).astype(np.uint8) for cid in people}
    speaker_mask, others_mask = folder / "speaker_mask.png", folder / "others_mask.png"
    Image.fromarray(share[speaker]).save(speaker_mask)
    Image.fromarray(np.max(np.stack([share[cid] for cid in others]), axis=0)).save(others_mask)
    return speaker_mask, others_mask


def generate_clip(show: dict, episode: dict, scene: dict, size: str, seed: int, force: bool) -> Path:
    """The scene's take (InfiniteTalk for an on-screen speaker, else VACE), its voice, then 24 fps at the scene's length."""
    ids = _ids(show, episode, scene)
    dest = clip_path(*ids)
    start = require_previs(show, episode, scene)
    voice = generate_voice(show, episode, scene)
    # The scene's real length: as scripted, or as long as its speech runs.
    scripted = float(scene["timeRangeSeconds"][1]) - float(scene["timeRangeSeconds"][0])
    seconds = track_seconds(voice)
    speaker = talking_speaker(scene)
    still = saved_still(start_still_path(*ids))
    if speaker is not None and not present(still):
        raise SystemExit(f"Missing start still {still}. Run `pnpm run content:frames`, draw it from {still_request_dir(still)}, and save it there.")
    prompt = video_prompt(show, episode, scene, start, speaker)
    expected = {
        "prompt": prompt, "size": size, "model": "infinitetalk" if speaker else "vace",
        "voice": _digest(voice), "still": _digest(still) if speaker else None,
        "guidance": {"text": wan.TALK_TEXT_CFG, "audio": wan.TALK_AUDIO_CFG} if speaker else None,
    }
    if made_from(dest, expected) and not force:
        print("  Clip is current.", flush=True)
        return dest

    # The take's own files, under its log.
    folder = inputs_dir(dest) / "work"
    folder.mkdir(parents=True, exist_ok=True)
    fps = wan.TALK_FPS if speaker is not None else wan.FPS
    length = wan.frames_for(seconds, fps)
    take = folder / f"take_{fps}fps.mp4"
    print(f"  Prompt: {prompt}", flush=True)
    if speaker is not None:
        start_picture = fitted_still(still, size, folder)
        masks = talk_masks(show, episode, scene, speaker, size, folder)
        others = None
        if masks is not None:
            silence = folder / "silence.wav"
            _ffmpeg("-f", "lavfi", "-i", f"anullsrc=r={wan.TALK_SILENCE_RATE}:cl=mono", "-t", f"{wan.TALK_SILENCE_SECONDS}", str(silence))
            others = (stage_input(masks[0], "talk_speaker"), stage_input(masks[1], "talk_others"), stage_input(silence, "talk_silence"))
            pictures = np.asarray(Image.open(start_picture))
            write_collage([[("still", Image.fromarray(pictures)),
                            ("the voice is theirs", over_nobody(pictures, np.asarray(Image.open(masks[0]), dtype=np.float32) / 255.0)),
                            ("silent", over_nobody(pictures, np.asarray(Image.open(masks[1]), dtype=np.float32) / 255.0))]],
                          inputs_dir(dest) / "collage.png", wan.SIZES[size][0])
        graph = wan.talk_graph(prompt, settings()["talkNegative"], size, length, seed, stage_input(start_picture, "wan_start"), stage_input(voice, "wan_voice"), others)
        run_graph(graph, {wan.TALK_PREFIX: take}, f"InfiniteTalk take {size}p, {length} frames, seed {seed}", TAKE_TIMEOUT_SECONDS)
    else:
        control, reference = shot_inputs(show, episode, scene, reference_people(show, scene), size, length, folder, pace=scripted / seconds)
        graph = wan.vace_graph(prompt, settings()["vaceNegative"], size, length, seed, control, reference, still=False)
        silent = folder / "take_silent.mp4"
        run_graph(graph, {wan.VACE_PREFIX: silent}, f"VACE take {size}p, {length} frames, seed {seed}", TAKE_TIMEOUT_SECONDS)
        _ffmpeg("-i", str(silent), "-i", str(voice), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-shortest", str(take))
    filled_fps = fps * wan.rife_multiplier(fps)
    filled = folder / f"take_{filled_fps}fps.mp4"
    free_comfy_models()
    run_graph(wan.rife_graph(stage_input(take, "wan_take"), fps), {wan.RIFE_PREFIX: filled}, f"RIFE to {filled_fps} fps", RIFE_TIMEOUT_SECONDS)
    # 24 fps taken from the filled take; the take ran a few frames past the scene, so it is cut to the scene.
    _ffmpeg("-i", str(filled), "-vf", f"fps={wan.OUTPUT_FPS}", "-t", f"{seconds:.3f}", "-c:v", "libx264", "-crf", "16",
            "-pix_fmt", "yuv420p", "-c:a", "aac", str(dest))
    write_log(dest, {**expected, "seed": seed, "speaker": speaker, "seconds": round(seconds, 3)}, {})
    return dest


def write_manifest(show: dict, episode: dict) -> None:
    manifest = manifest_path(show["id"], episode["episodeNumber"])
    manifest.parent.mkdir(parents=True, exist_ok=True)
    fields = ("title", "isFree", "coinCost")
    manifest.write_text(
        json.dumps({"series": show["id"], "episodeNumber": episode["episodeNumber"], **{key: episode[key] for key in fields}}, indent=2),
        encoding="utf-8",
    )


def generate_show(show: dict, size: str, partial: bool, force: bool, seed_override: int | None) -> None:
    """Every scene's clip, then, unless only some scenes were asked for, the episode cut."""
    started = time.time()
    for episode in show["episodes"]:
        write_manifest(show, episode)

        def seed(scene: dict, kind: str) -> int:
            return seed_override if seed_override is not None else stable_seed(show["id"], episode["episodeNumber"], scene["sceneNumber"], kind)

        for scene in episode["scenes"]:
            print(f"Scene {scene['sceneNumber']:02d} clip...", flush=True)
            generate_clip(show, episode, scene, size, seed(scene, "clip"), force)
        if not partial:
            ordered = [clip_path(show["id"], episode["episodeNumber"], scene["sceneNumber"]) for scene in episode["scenes"]]
            episode_mp4 = episode_video_path(show["id"], episode["episodeNumber"])
            concat_videos(ordered, episode_mp4, loudness_target_lufs=settings()["episodeLoudnessTargetLufs"])
            free_comfy_models()
            caption_episode(show, episode, ordered, episode_mp4)
            print(f"Wrote {episode_mp4}", flush=True)
    print(f"Show {show['id']} clips finished in {format_duration(time.time() - started)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write what an image model needs to draw each start still (frames), render clips through ComfyUI (video), or print every prompt a run would send (prompts).")
    parser.add_argument("--stage", choices=("prompts", "frames", "video"), default="video")
    parser.add_argument("--show", help="Render only this show id")
    parser.add_argument("--episode", type=int, help="Render only this episode number")
    parser.add_argument("--scene", type=int, help="Render only this scene number (requires --episode)")
    parser.add_argument("--size", choices=sorted(wan.SIZES), default=wan.DEFAULT_SIZE, help="Frame height class: 480 for drafts, 720 for the final look")
    parser.add_argument("--force", action="store_true", help="Remake the selected scene even when it is current (requires --scene)")
    parser.add_argument("--seed", type=int, help="Override the deterministic seed for one selected scene (requires --scene)")
    args = parser.parse_args()
    if args.scene is not None and args.episode is None:
        parser.error("--scene requires --episode")
    if (args.force or args.seed is not None) and args.scene is None:
        parser.error("--force and --seed require --scene")
    if args.seed is not None and not 0 <= args.seed <= 2**32 - 1:
        parser.error("--seed must be between 0 and 4294967295")
    load_dotenv()
    scripts = discover_show_scripts(args.show)
    if not scripts:
        raise SystemExit(f"No show JSON files found{f' for show {args.show!r}' if args.show else ''}")
    shows = [load_show(path) for path in scripts]
    for show in shows:
        if args.episode is not None:
            show["episodes"] = [episode for episode in show["episodes"] if episode["episodeNumber"] == args.episode]
            if not show["episodes"]:
                raise SystemExit(f"Show {show['id']!r} has no episode {args.episode}")
        if args.scene is not None:
            show["episodes"][0]["scenes"] = [scene for scene in show["episodes"][0]["scenes"] if scene["sceneNumber"] == args.scene]
            if not show["episodes"][0]["scenes"]:
                raise SystemExit(f"Show {show['id']!r} episode {args.episode} has no scene {args.scene}")
    if args.stage in ("prompts", "frames"):
        for show in shows:
            for episode in show["episodes"]:
                for scene in episode["scenes"]:
                    print(f"Scene {scene['sceneNumber']:02d} {'prompts' if args.stage == 'prompts' else 'still to draw'}:", flush=True)
                    if args.stage == "prompts":
                        print_prompts(show, episode, scene, args.size)
                    else:
                        write_still_request(show, episode, scene, args.size)
        if args.stage == "frames":
            print("Draw each still in Gemini from its to_draw folder, save it where it says, then run `pnpm run content:generate`.", flush=True)
        return
    require_comfy()
    for show in shows:
        generate_show(show, args.size, partial=args.scene is not None, force=args.force, seed_override=args.seed)


if __name__ == "__main__":
    main()
