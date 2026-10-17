#!/usr/bin/env python3
"""content:frames and content:generate: the Wan 2.2 chain, scene by scene.

frames: previs (the 3D scene and its checks), then the start still: each person
        in the start frame drawn alone by Qwen-Image from their words and depth, the
        shot's piece of them restored by SeedVR2 and cut out, laid on the frame, and
        the shot drawn around them by Wan VACE (``shot_inputs``).

Beside each output, ``inputs/<name>/`` holds only what a review needs: its log
and, for a still, a collage of every step. What a stage keeps for itself is in
``work/`` under it.
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

import numpy as np
from PIL import Image

import wan
from captions import caption_episode
from collage import first_frame, frame_outline, over_nobody, write_collage
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
from describe import alone_prompt, reference_people, talking_speaker, video_prompt
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
from asset_generate import CHARACTER_PICTURE_SIZE, PICTURE_PREFIX, picture_graph
from shot_inputs import (
    drawing_camera,
    drawing_pixels,
    figure_box,
    fit_similarity,
    layer_people,
    own_colors,
    person_depth,
    placed,
    reference_sheet,
    seen_surface,
    shot_piece,
    start_frame,
    window_around,
    write_depth_video,
    write_kept_control,
)
from voice import write_voice_track
from render import render
from world import camera_at, entities_at, load_show, reference_path, settings

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


def shot_inputs(show: dict, episode: dict, scene: dict, people: list[str], size: str, length: int, folder: Path, still: bool) -> tuple[str, str | None]:
    """The depth video for a take of ``length`` frames and the reference sheet of ``people``, staged for ComfyUI.

    A still's depth keeps the people's relief; a clip's has them flat, so VACE does not freeze the faces.
    """
    width, height = wan.SIZES[size]
    depth = write_depth_video(show, episode, scene, width, height, length, wan.FPS, folder / "depth.mp4", flat_people=not still)
    photos = [reference_path(show, cid) for cid in people]
    missing = [str(path) for path in photos if not present(path)]
    if missing:
        raise SystemExit("Missing reference photos (run `pnpm run content:plates`):\n- " + "\n- ".join(missing))
    sheet = reference_sheet(photos, width, height, folder / "reference.png") if photos else None
    return stage_input(depth, "wan_depth"), stage_input(sheet, "wan_reference") if sheet else None


def draw_person(show: dict, episode: dict, scene: dict, start: dict, cid: str, frame, number: int, seed: int, folder: Path, force: bool):
    """A person of a still, alone, upright and full height, by Qwen-Image from their words and their
    own depth (through the ControlNet Union), seen through a long lens from the direction the shot
    (whose render is ``frame``, where they are entity ``number``) sees them from. Returns the
    drawing and the scale, turn and shift that lay it on the shot."""
    width, height = CHARACTER_PICTURE_SIZE
    time_seconds = float(scene["timeRangeSeconds"][0])
    person = next(entity for entity in entities_at(show, episode, scene, time_seconds) if entity.id == cid)
    shot_pixels, seen = seen_surface(frame.entity, frame.depth, camera_at(scene, time_seconds), number)
    camera = drawing_camera(person, seen, camera_at(scene, time_seconds))
    box = figure_box(person, camera, width, height)
    window = window_around(box[:2], box[2:], width, height)
    turn, shift = fit_similarity(drawing_pixels(seen, camera, window, (width, height)), shot_pixels)
    dest = folder / f"alone_{cid}.png"
    guides = inputs_dir(dest)
    guides.mkdir(parents=True, exist_ok=True)
    depth = guides / "depth.png"
    person_depth(person, camera, window, (width, height), (width, height)).save(depth)
    prompt = alone_prompt(show, scene, start, cid)
    negative = settings()["aloneNegative"]
    expected = {"prompt": prompt, "negative": negative, "size": [width, height], "depth": _digest(depth)}
    if not made_from(dest, expected) or force:
        print(f"  Alone: {prompt}", flush=True)
        graph = picture_graph(prompt, negative, (width, height), seed, stage_input(depth, "alone_depth"))
        run_graph(graph, {PICTURE_PREFIX: dest}, f"Qwen-Image {cid} alone, seed {seed}", STILL_TIMEOUT_SECONDS)
        write_log(dest, {**expected, "seed": seed})
    return dest, (turn, shift), depth


def shot_part(cid: str, drawing: Path, placement: tuple[complex, complex], frame: tuple[int, int], seed: int, folder: Path, force: bool):
    """The shot's piece of a person's drawing, laid on the frame by ``placement`` at its size in the
    shot and given back its detail by SeedVR2; its cut-out, made on that piece, so its edge is as
    sharp as the picture (made on the drawing and enlarged with it, the edge blurred two people
    into each other); where it goes in the frame; and the piece as laid, before SeedVR2."""
    part, box = shot_piece(Image.open(drawing).convert("RGB"), *placement, frame)
    enlarged = folder / f"alone_{cid}_shot_enlarged.png"
    part.save(enlarged)
    dest = folder / f"alone_{cid}_shot.png"
    expected = {"part": _digest(enlarged), "model": wan.RESTORE_MODEL, "color": wan.RESTORE_COLOR}
    if not made_from(dest, expected) or force:
        run_graph(wan.restore_graph(stage_input(enlarged, "alone_part"), seed), {wan.RESTORE_PREFIX: dest}, f"SeedVR2 {cid}", STILL_TIMEOUT_SECONDS)
        write_log(dest, {**expected, "seed": seed})
    matte = folder / f"alone_{cid}_shot_matte.png"
    if not made_from(matte, {"piece": _digest(dest), "model": wan.MATTE_MODEL}) or force:
        run_graph(wan.matte_graph(stage_input(dest, "alone_piece")), {wan.MATTE_PREFIX: matte}, f"Cut out {cid}", STILL_TIMEOUT_SECONDS)
        write_log(matte, {"piece": _digest(dest), "model": wan.MATTE_MODEL})
    return dest, matte, box, enlarged


def generate_still(show: dict, episode: dict, scene: dict, size: str, seed: int, force: bool) -> Path:
    """The scene's start still: the first frame of a short VACE take (a one-frame take ignores the depth).

    Everyone with any pixel in the start frame is drawn alone and full height first and cut out of
    their drawing; the shot's piece of them is enlarged and given back its detail, and laid in by
    their cut-out, nearer people over farther, where no thing is in front of them; the still's own
    take keeps them and draws everything around them, then they are laid in again exactly (the
    sampler works in latent space).

    Beside it, in ``inputs/<still>/``: its log and a collage of every step, for review; what the
    steps keep for the next run (the drawings, pieces, cut-outs, the final pass's inputs) is in
    ``work/`` under it.
    """
    dest = start_still_path(*_ids(show, episode, scene))
    start = require_previs(show, episode, scene)
    prompt = video_prompt(show, episode, scene, start)
    work = inputs_dir(dest) / "work"
    width, height = wan.SIZES[size]
    people, entities, frame = start_frame(show, episode, scene, width, height)
    numbers = {entity.id: number + 1 for number, entity in enumerate(entities)}
    drawn = {cid: draw_person(show, episode, scene, start, cid, frame, numbers[cid], stable_seed(seed, "alone", cid), work, force)
             for cid in people}
    parts = {cid: shot_part(cid, path, placement, (width, height), stable_seed(seed, "restore", cid), work, force)
             for cid, (path, placement, _depth) in drawn.items()}
    expected = {"prompt": prompt, "size": size,
                "people": {cid: [_digest(piece), _digest(matte)] for cid, (piece, matte, _box, _laid) in parts.items()},
                "depth": "without people", "strength": wan.STILL_VACE_STRENGTH, "kept": "blended by cut-out, own colors, every frame",
                "layered": "people cover each other by cut-outs only"}
    if made_from(dest, expected) and present(inputs_dir(dest) / "collage.png") and not force:
        print("  Still is current.", flush=True)
        return dest
    camera = camera_at(scene, float(scene["timeRangeSeconds"][0]))
    layers, rows = [], []
    for cid, (piece, matte, box, laid) in parts.items():
        size_in_shot = (box[2] - box[0], box[3] - box[1])
        cut = np.asarray(Image.open(matte).convert("L").resize(size_in_shot, Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
        colors = own_colors(np.asarray(Image.open(piece).convert("RGB").resize(size_in_shot, Image.Resampling.LANCZOS)), cut)
        layers.append((numbers[cid], render([next(e for e in entities if e.id == cid)], camera, width, height).depth,
                       placed(colors, box, (width, height)), placed(cut, box, (width, height))))
        drawing, placement, depth = drawn[cid]
        rows.append([(f"{cid}: 1 depth drawn from", Image.open(depth)),
                     (f"{cid}: 2 drawing, red = the shot", frame_outline(Image.open(drawing), *placement, (width, height))),
                     (f"{cid}: 3 shot piece, laid", Image.open(laid)),
                     (f"{cid}: 4 SeedVR2", Image.open(piece)),
                     (f"{cid}: 5 cut-out", Image.open(matte)),
                     (f"{cid}: 6 own colors", over_nobody(colors, cut))])
    floors = [number + 1 for number, entity in enumerate(entities) if entity.kind == "floor"]
    picture, alpha = layer_people(frame.entity, frame.depth, layers, floors)
    depth = write_depth_video(show, episode, scene, width, height, wan.STILL_FRAMES, wan.FPS, work / "depth.mp4", flat_people=False, people=False)
    write_kept_control(depth, picture, alpha, wan.STILL_FRAMES, wan.FPS, work / "control.mp4", work / "keep.mp4")
    graph = wan.vace_graph(prompt, settings()["vaceNegative"], size, wan.STILL_FRAMES, seed,
                           stage_input(work / "control.mp4", "wan_control"), None, still=True, mask=stage_input(work / "keep.mp4", "wan_keep"))
    print(f"  Prompt: {prompt}", flush=True)
    print(f"  People drawn alone: {', '.join(people) or 'none'}", flush=True)
    # What the final pass drew, then the people laid over it.
    around = work / "around.png"
    run_graph(graph, {wan.STILL_PREFIX: around}, f"VACE still {size}p, seed {seed}", STILL_TIMEOUT_SECONDS)
    drew = Image.open(around).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    still = Image.fromarray(np.rint(picture * alpha[..., None] + np.asarray(drew) * (1.0 - alpha[..., None])).astype(np.uint8))
    still.save(dest)
    steps = [("7 people layered", over_nobody(picture, alpha)),
             ("8 how much is a person", Image.fromarray(np.rint(alpha * 255).astype(np.uint8))),
             ("9 final pass control", first_frame(work / "control.mp4")),
             ("10 keep mask (black)", first_frame(work / "keep.mp4")),
             ("11 final pass drew", drew),
             ("12 still", still)]
    # At the frame's own size, so every pixel of the still can be checked in it.
    write_collage([*rows, steps], inputs_dir(dest) / "collage.png", width)
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

    # The take's own files, under its log.
    folder = inputs_dir(dest) / "work"
    folder.mkdir(parents=True, exist_ok=True)
    take = folder / "take_16fps.mp4"
    print(f"  Prompt: {prompt}", flush=True)
    if speaker is not None:
        graph = wan.s2v_graph(prompt, settings()["s2vNegative"], size, length, seed, stage_input(still, "wan_start"), stage_input(voice, "wan_voice"))
        run_graph(graph, {wan.S2V_PREFIX: take}, f"S2V take {size}p, {length} frames, seed {seed}", TAKE_TIMEOUT_SECONDS)
    else:
        control, reference = shot_inputs(show, episode, scene, reference_people(show, scene), size, length, folder, still=False)
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
