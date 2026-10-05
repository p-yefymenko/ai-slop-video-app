#!/usr/bin/env python3

"""Run Depth Anything on an LTX output mp4 for camera-follow Spearman metric."""

from __future__ import annotations

import _scripts_path  # noqa: F401

import argparse
import shutil
from pathlib import Path

from clip_spec import CLIP_HEIGHT, CLIP_WIDTH, FPS
from gate_depth_lib import (
    COMFY_INPUT,
    assert_depth_levels,
    decode_mp4_gray,
    depth_anything_workflow,
    mux_silent_aac,
    scene_gate_dir,
)
from gate_depth_lib import queue_and_download_video


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=int, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tag", required=True, help="variant tag for output filename")
    args = parser.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"missing {args.input}")
    frames = decode_mp4_gray(args.input)
    frame_count = len(frames)
    COMFY_INPUT.mkdir(parents=True, exist_ok=True)
    clay_name = f"gate_ltx_src_s{args.scene:02d}_{args.tag}.mp4"
    shutil.copy2(args.input, COMFY_INPUT / clay_name)
    prefix = f"gate_da_out_s{args.scene:02d}_{args.tag}"
    dest = scene_gate_dir(args.scene) / f"da_on_{args.tag}.mp4"
    tmp = scene_gate_dir(args.scene) / f"_da_on_{args.tag}_raw.mp4"
    graph = depth_anything_workflow(clay_name, frame_count, prefix)
    queue_and_download_video(graph, tmp, prefix)
    mux_silent_aac(tmp, dest)
    out = decode_mp4_gray(dest)
    assert_depth_levels(out[0], f"DA-on-output {args.tag}")
    print(f"Wrote {dest} frames={len(out)} size={CLIP_WIDTH}x{CLIP_HEIGHT}@{FPS}", flush=True)


if __name__ == "__main__":
    main()
