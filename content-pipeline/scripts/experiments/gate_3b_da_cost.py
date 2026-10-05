#!/usr/bin/env python3

"""Measure Depth Anything wall/VRAM at 121 frames and confirm unload before LTX."""

from __future__ import annotations

import _scripts_path  # noqa: F401

import json
import shutil
import subprocess
import time
from pathlib import Path

from clip_spec import CLIP_HEIGHT, CLIP_WIDTH, FPS
from gate_depth_lib import (
    COMFY_INPUT,
    GATE_ROOT,
    decode_mp4_gray,
    depth_anything_workflow,
    mux_silent_aac,
    queue_and_download_video,
)
from generate_batch import free_comfy_models
from pipeline_paths import clay_24fps_path


def smi_used() -> float:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    return float(out.splitlines()[0])


def main() -> None:
    clay = clay_24fps_path("the-iron-bride", 1, 19)
    if not clay.is_file():
        raise SystemExit(f"missing {clay}")
    # Use first 121 frames of scene 19 clay (already 121).
    frames = 121
    COMFY_INPUT.mkdir(parents=True, exist_ok=True)
    clay_name = "gate_da_cost_clay.mp4"
    shutil.copy2(clay, COMFY_INPUT / clay_name)
    prefix = "gate_da_cost"
    dest = GATE_ROOT / "da_cost_121.mp4"
    tmp = GATE_ROOT / "da_cost_121_raw.mp4"
    before = smi_used()
    print(f"nvidia-smi before DA: {before:.0f} MiB", flush=True)
    graph = depth_anything_workflow(clay_name, frames, prefix)
    t0 = time.time()
    stats = queue_and_download_video(graph, tmp, prefix)
    mux_silent_aac(tmp, dest)
    wall = time.time() - t0
    after_free = smi_used()
    print(
        f"DA done wall={wall:.1f}s peak_during={stats['peak_vram_mib']:.0f} "
        f"after_free_comfy={after_free:.0f} MiB",
        flush=True,
    )
    free_comfy_models()
    between = smi_used()
    print(f"nvidia-smi between DA unload and LTX: {between:.0f} MiB", flush=True)
    # Confirm frames/size
    gray = decode_mp4_gray(dest)
    report = {
        "frames": len(gray),
        "size": f"{CLIP_WIDTH}x{CLIP_HEIGHT}",
        "fps": FPS,
        "wall_seconds": wall,
        "peak_vram_mib": stats["peak_vram_mib"],
        "vram_before_mib": before,
        "vram_after_unload_mib": between,
        "unloaded_before_ltx": between < 4000,
        "output": str(dest),
    }
    out = GATE_ROOT / "da_cost_121.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
