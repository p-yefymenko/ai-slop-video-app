#!/usr/bin/env python3
"""content:frames and content:generate: the Wan 2.2 chain, scene by scene.

frames: previs (the 3D scene and its checks), then the start still: Wan VACE
        from the shot's depth and the reference sheet (``shot_inputs``).
video:  the scene's voice track (``voice``), then the take:
        - an on-screen speaker whose face the shot shows: Wan S2V from the start
          still and the voice (``describe.talking_speaker``);
        - anything else: Wan VACE over the whole take, the voice laid under it;
        then RIFE to 24 fps, cut to the scene's exact length. When every scene
        of an episode has its clip, they are joined and captioned.

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

import wan
from captions import caption_episode
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
from describe import reference_people, talking_speaker, video_prompt
from ffmpeg_tools import concat_videos
from pipeline_paths import (
    OUTPUT_DIR,
    clip_path,
    discover_show_scripts,
    episode_video_path,
    manifest_path,
    start_still_path,
    voice_track_path,
)
from previs import current_previs, load_observation, previs_episode
from shot_inputs import reference_sheet, write_depth_video
from voice import write_voice_track
from world import load_show, reference_path, settings

# A still takes minutes; a 720p talking take can take well over an hour. Past these a run has hung.
STILL_TIMEOUT_SECONDS = 1800
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


def shot_inputs(show: dict, episode: dict, scene: dict, people: list[str], size: str, length: int, folder: Path) -> tuple[str, str | None]:
    """The depth video for a take of ``length`` frames and the reference sheet of ``people``, staged for ComfyUI."""
    width, height = wan.SIZES[size]
    depth = write_depth_video(show, episode, scene, width, height, length, wan.FPS, folder / "depth.mp4")
    photos = [reference_path(show, cid) for cid in people]
    missing = [str(path) for path in photos if not present(path)]
    if missing:
        raise SystemExit("Missing reference photos (run `pnpm run content:plates`):\n- " + "\n- ".join(missing))
    sheet = reference_sheet(photos, width, height, folder / "reference.png") if photos else None
    return stage_input(depth, "wan_depth"), stage_input(sheet, "wan_reference") if sheet else None


def generate_still(show: dict, episode: dict, scene: dict, size: str, seed: int, force: bool) -> Path:
    """The scene's start still: the first frame of a short VACE take (a one-frame take ignores the depth)."""
    dest = start_still_path(*_ids(show, episode, scene))
    start = require_previs(show, episode, scene)
    prompt = video_prompt(show, episode, scene, start)
    people = reference_people(show, scene)
    expected = {"prompt": prompt, "size": size, "references": people}
    if made_from(dest, expected) and not force:
        print("  Still is current.", flush=True)
        return dest
    control, reference = shot_inputs(show, episode, scene, people, size, wan.STILL_FRAMES, inputs_dir(dest))
    graph = wan.vace_graph(prompt, settings()["vaceNegative"], size, wan.STILL_FRAMES, seed, control, reference, still=True)
    print(f"  Prompt: {prompt}", flush=True)
    run_graph(graph, {wan.STILL_PREFIX: dest}, f"VACE still {size}p, seed {seed}", STILL_TIMEOUT_SECONDS)
    write_log(dest, {**expected, "seed": seed})
    return dest


def generate_voice(show: dict, episode: dict, scene: dict) -> Path:
    """The scene's voice track, remade only when its lines change."""
    dest = voice_track_path(*_ids(show, episode, scene))
    expected = {"lines": scene.get("lines") or [], "timeRangeSeconds": scene["timeRangeSeconds"]}
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


def generate_clip(show: dict, episode: dict, scene: dict, size: str, seed: int, force: bool) -> Path:
    """The scene's take at 16 fps (S2V for an on-screen speaker, else VACE), its voice, then 24 fps at the scene's length."""
    ids = _ids(show, episode, scene)
    dest = clip_path(*ids)
    start = require_previs(show, episode, scene)
    seconds = float(scene["timeRangeSeconds"][1]) - float(scene["timeRangeSeconds"][0])
    length = wan.frames_for(seconds)
    voice = generate_voice(show, episode, scene)
    speaker = talking_speaker(scene)
    still = start_still_path(*ids)
    if speaker is not None and not present(still):
        raise SystemExit(f"Missing start still {still}. Run `pnpm run content:frames` first, and review it.")
    prompt = video_prompt(show, episode, scene, start, speaker)
    expected = {
        "prompt": prompt, "size": size, "model": "s2v" if speaker else "vace",
        "voice": _digest(voice), "still": _digest(still) if speaker else None,
    }
    if made_from(dest, expected) and not force:
        print("  Clip is current.", flush=True)
        return dest

    folder = inputs_dir(dest)
    folder.mkdir(parents=True, exist_ok=True)
    take = folder / "take_16fps.mp4"
    print(f"  Prompt: {prompt}", flush=True)
    if speaker is not None:
        graph = wan.s2v_graph(prompt, settings()["s2vNegative"], size, length, seed, stage_input(still, "wan_start"), stage_input(voice, "wan_voice"))
        run_graph(graph, {wan.S2V_PREFIX: take}, f"S2V take {size}p, {length} frames, seed {seed}", TAKE_TIMEOUT_SECONDS)
    else:
        control, reference = shot_inputs(show, episode, scene, reference_people(show, scene), size, length, folder)
        graph = wan.vace_graph(prompt, settings()["vaceNegative"], size, length, seed, control, reference, still=False)
        silent = folder / "take_silent.mp4"
        run_graph(graph, {wan.VACE_PREFIX: silent}, f"VACE take {size}p, {length} frames, seed {seed}", TAKE_TIMEOUT_SECONDS)
        _ffmpeg("-i", str(silent), "-i", str(voice), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-shortest", str(take))
    tripled = folder / "take_48fps.mp4"
    free_comfy_models()
    run_graph(wan.rife_graph(stage_input(take, "wan_take")), {wan.RIFE_PREFIX: tripled}, "RIFE to 48 fps", RIFE_TIMEOUT_SECONDS)
    # Every other frame of 48 is 24; the take ran a few frames past the scene, so it is cut to the scene.
    _ffmpeg("-i", str(tripled), "-vf", f"fps={wan.OUTPUT_FPS}", "-t", f"{seconds:.3f}", "-c:v", "libx264", "-crf", "16",
            "-pix_fmt", "yuv420p", "-c:a", "aac", str(dest))
    write_log(dest, {**expected, "seed": seed, "speaker": speaker}, {})
    return dest


def write_manifest(show: dict, episode: dict) -> None:
    manifest = manifest_path(show["id"], episode["episodeNumber"])
    manifest.parent.mkdir(parents=True, exist_ok=True)
    fields = ("title", "isFree", "coinCost")
    manifest.write_text(
        json.dumps({"series": show["id"], "episodeNumber": episode["episodeNumber"], **{key: episode[key] for key in fields}}, indent=2),
        encoding="utf-8",
    )


def generate_show(show: dict, stage: str, size: str, partial: bool, force: bool, seed_override: int | None) -> None:
    started = time.time()
    for episode in show["episodes"]:
        write_manifest(show, episode)

        def seed(scene: dict, kind: str) -> int:
            return seed_override if seed_override is not None else stable_seed(show["id"], episode["episodeNumber"], scene["sceneNumber"], kind)

        for scene in episode["scenes"]:
            print(f"Scene {scene['sceneNumber']:02d} {'still' if stage == 'frames' else 'clip'}...", flush=True)
            if stage == "frames":
                generate_still(show, episode, scene, size, seed(scene, "still"), force)
            else:
                generate_clip(show, episode, scene, size, seed(scene, "clip"), force)
        if stage == "video" and not partial:
            ordered = [clip_path(show["id"], episode["episodeNumber"], scene["sceneNumber"]) for scene in episode["scenes"]]
            episode_mp4 = episode_video_path(show["id"], episode["episodeNumber"])
            concat_videos(ordered, episode_mp4, loudness_target_lufs=settings()["episodeLoudnessTargetLufs"])
            free_comfy_models()
            caption_episode(show, episode, ordered, episode_mp4)
            print(f"Wrote {episode_mp4}", flush=True)
    if stage == "frames":
        print(f"Stills are in {OUTPUT_DIR / 'frames' / show['id']}. Review them, then run `pnpm run content:generate`.", flush=True)
    print(f"Show {show['id']} {stage} finished in {format_duration(time.time() - started)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render scene stills (frames) or clips (video) with Wan 2.2 through ComfyUI.")
    parser.add_argument("--stage", choices=("frames", "video"), default="video")
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
    require_comfy()
    for show in shows:
        generate_show(show, args.stage, args.size, partial=args.scene is not None, force=args.force, seed_override=args.seed)


if __name__ == "__main__":
    main()
