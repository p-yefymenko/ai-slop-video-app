#!/usr/bin/env python3

"""Build S2 native re-encodes: A_eq, A_gamma06, B_linear under tmp/depth_gate/scene_19/."""

from __future__ import annotations

import _scripts_path  # noqa: F401

import json

import numpy as np
from clip_spec import FPS, clip_frame_count
from gate_depth_lib import (
    GAMMA,
    PROMPTS_PATH,
    assert_depth_levels,
    encode_norm_frames_mp4,
    read_depth_png16,
    scene_gate_dir,
    write_gamma_depth_mp4,
    write_linear_depth_mp4,
)
from pipeline_paths import depth_video_dir, discover_show_scripts
from spatial_previs import load_show


def write_equalized_depth_mp4(show_id: str, episode: int, scene_number: int):
    depth_dir = depth_video_dir(show_id, episode, scene_number)
    pngs = sorted(depth_dir.glob("frame_*.png"))
    if not pngs:
        raise FileNotFoundError(depth_dir)
    norms = [read_depth_png16(p) for p in pngs]
    assert_depth_levels(norms[0], "A_eq source")
    # CDF over all valid pixels of all frames
    chunks = [n[n > 1e-6].ravel() for n in norms]
    all_valid = np.concatenate(chunks)
    # Histogram CDF on 4096 bins for smooth monotonic map
    hist, edges = np.histogram(all_valid, bins=4096, range=(0.0, 1.0))
    cdf = np.cumsum(hist).astype(np.float64)
    if cdf[-1] <= 0:
        raise RuntimeError("empty CDF")
    cdf /= cdf[-1]
    # Map bin centers; interpolate
    centers = 0.5 * (edges[:-1] + edges[1:])

    def apply_cdf(norm: np.ndarray) -> np.ndarray:
        out = np.zeros_like(norm, dtype=np.float32)
        valid = norm > 1e-6
        out[valid] = np.interp(norm[valid], centers, cdf).astype(np.float32)
        return out

    mapped = [apply_cdf(n) for n in norms]
    dest = scene_gate_dir(scene_number) / "depth_inverse_eq.mp4"
    encode_norm_frames_mp4(mapped, dest, gamma=None)
    meta = {
        "mapping": "inverse_histogram_eq",
        "bins": 4096,
        "frameCount": len(norms),
        "validPixels": int(all_valid.size),
    }
    (scene_gate_dir(scene_number) / "depth_inverse_eq_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"A_eq: {dest}", flush=True)
    return dest


def main() -> None:
    show = load_show(discover_show_scripts("the-iron-bride")[0])
    ep = show["episodes"][0]
    scene = next(sc for sc in ep["scenes"] if sc["sceneNumber"] == 19)
    write_equalized_depth_mp4(show["id"], 1, 19)
    # A_gamma may already exist from earlier prepare; rewrite for clarity
    path = write_gamma_depth_mp4(show["id"], 1, 19)
    print(f"A_gamma06: {path}", flush=True)
    before = PROMPTS_PATH.read_text(encoding="utf-8")
    path_b = write_linear_depth_mp4(show, ep, scene)
    after = PROMPTS_PATH.read_text(encoding="utf-8")
    if before != after:
        raise SystemExit("prompts.json not restored")
    print(f"B_linear: {path_b} (prompts restored)", flush=True)


if __name__ == "__main__":
    main()
