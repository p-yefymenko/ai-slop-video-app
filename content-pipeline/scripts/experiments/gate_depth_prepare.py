#!/usr/bin/env python3

"""Prepare depth control videos A_gamma / B / C for the gate experiment."""

from __future__ import annotations

import _scripts_path  # noqa: F401

import argparse
import shutil
import subprocess
from pathlib import Path

from clip_spec import clip_frame_count
from gate_depth_lib import (
    COMFY_INPUT,
    GATE_ROOT,
    PROMPTS_PATH,
    assert_depth_levels,
    decode_mp4_gray,
    depth_anything_workflow,
    mux_silent_aac,
    scene_gate_dir,
    write_gamma_depth_mp4,
    write_linear_depth_mp4,
)
from pipeline_paths import (
    clay_24fps_path,
    depth_video_mp4_path,
    discover_show_scripts,
)
from spatial_previs import load_show


def _run_depth_anything(clay_src: Path, dest: Path, frame_count: int, scene_number: int) -> None:
    from gate_depth_lib import queue_and_download_video

    COMFY_INPUT.mkdir(parents=True, exist_ok=True)
    clay_name = f"gate_clay_{scene_number:02d}.mp4"
    shutil.copy2(clay_src, COMFY_INPUT / clay_name)
    prefix = f"gate_da_{scene_number:02d}"
    graph = depth_anything_workflow(clay_name, frame_count, prefix)
    tmp_out = GATE_ROOT / f"_da_raw_{scene_number:02d}.mp4"
    stats = queue_and_download_video(graph, tmp_out, prefix)
    mux_silent_aac(tmp_out, dest)
    frames = decode_mp4_gray(dest)
    assert_depth_levels(frames[0], f"DepthAnything scene {scene_number}")
    print(
        f"Wrote {dest} ({len(frames)} frames) peak_vram={stats['peak_vram_mib']:.0f} MiB "
        f"wall={stats['seconds']:.1f}s",
        flush=True,
    )


def prepare_scene(show: dict, episode: dict, scene: dict, variants: list[str]) -> None:
    n = int(scene["sceneNumber"])
    out = scene_gate_dir(n)
    show_id = show["id"]
    episode_number = episode["episodeNumber"]
    native = depth_video_mp4_path(show_id, episode_number, n)
    clay = clay_24fps_path(show_id, episode_number, n)
    if "A" in variants:
        if not native.is_file():
            raise SystemExit(f"Missing native depth {native}; run content:previs first")
        frames = decode_mp4_gray(native)
        assert_depth_levels(frames[0], f"A native scene {n}")
        shutil.copy2(native, out / "depth_inverse.mp4")
        print(f"A: copied {native} -> {out / 'depth_inverse.mp4'}", flush=True)
    if "A_gamma" in variants:
        path = write_gamma_depth_mp4(show_id, episode_number, n)
        print(f"A_gamma: {path}", flush=True)
    if "B" in variants:
        before = PROMPTS_PATH.read_text(encoding="utf-8")
        path = write_linear_depth_mp4(show, episode, scene)
        after = PROMPTS_PATH.read_text(encoding="utf-8")
        if before != after:
            raise SystemExit("prompts.json was not restored after linear depth render")
        # Confirm restore vs pre-B snapshot (file may already differ from git HEAD from step 2).
        print(f"B: {path} (prompts.json restored byte-identical to pre-B)", flush=True)
    if "C" in variants:
        if not clay.is_file():
            raise SystemExit(f"Missing clay_24fps {clay}")
        count = clip_frame_count(scene["timeRangeSeconds"])
        dest = out / "depth_depthanything.mp4"
        _run_depth_anything(clay, dest, count, n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", default="the-iron-bride")
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--scene", type=int, required=True)
    parser.add_argument(
        "--variants",
        default="A,A_gamma,B,C",
        help="Comma list: A,A_gamma,B,C",
    )
    args = parser.parse_args()
    # spatial_previs.load_show keeps full mesh fields needed for linear depth re-render.
    show = load_show(discover_show_scripts(args.show)[0])
    episode = next(ep for ep in show["episodes"] if ep["episodeNumber"] == args.episode)
    scene = next(sc for sc in episode["scenes"] if sc["sceneNumber"] == args.scene)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    prepare_scene(show, episode, scene, variants)


if __name__ == "__main__":
    main()
