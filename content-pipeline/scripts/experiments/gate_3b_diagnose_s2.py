#!/usr/bin/env python3

"""Diagnose why native inverse depth fails on scene 19 vs Depth Anything control."""

from __future__ import annotations

import _scripts_path  # noqa: F401

import json
from pathlib import Path

import numpy as np
from PIL import Image
from clip_spec import CLIP_HEIGHT, CLIP_WIDTH, FPS, clip_frame_count, fit_to_clip
from gate_depth_lib import (
    assert_depth_levels,
    decode_mp4_gray,
    decode_mp4_rgb,
    read_depth_png16,
    scene_gate_dir,
)
from pipeline_paths import depth_video_dir, discover_show_scripts
from spatial_previs import (
    PROXY_HEIGHT,
    PROXY_WIDTH,
    load_show,
    render_blocked_frame,
)


def _pcts(vals: np.ndarray) -> dict:
    return {
        "min": float(vals.min()),
        "max": float(vals.max()),
        **{f"p{p}": float(np.percentile(vals, p)) for p in (1, 5, 25, 50, 75, 95, 99)},
    }


def _castle_mask(gray: np.ndarray) -> np.ndarray:
    """Nearest 40% of valid (non-empty) pixels вЂ” proxy for castle mass on S2."""
    valid = gray > 8.0
    if not bool(valid.any()):
        return valid
    thresh = float(np.percentile(gray[valid], 60.0))  # top 40% brightest = nearest in inverse
    return valid & (gray >= thresh)


def main() -> None:
    show = load_show(discover_show_scripts("the-iron-bride")[0])
    ep = show["episodes"][0]
    scene = next(sc for sc in ep["scenes"] if sc["sceneNumber"] == 19)
    gate = scene_gate_dir(19)
    a_ctrl = gate / "depth_inverse.mp4"
    c_ctrl = gate / "depth_depthanything.mp4"
    meta = json.loads((depth_video_dir(show["id"], 1, 19) / "meta.json").read_text(encoding="utf-8"))
    print("meta_near_far", meta["near"], meta["far"], "mapping", meta["mapping"])

    # Float z percentiles over the whole scene (re-render all clip times).
    start, finish = (float(v) for v in scene["timeRangeSeconds"])
    count = clip_frame_count([start, finish])
    zs = []
    for i in range(count):
        t = start + i / FPS
        _img, zbuf, _ = render_blocked_frame(show, ep, scene, t)
        z = np.asarray(zbuf, dtype=np.float32)
        zs.append(z[np.isfinite(z)])
    z_all = np.concatenate(zs)
    print("float_z_valid_pcts", _pcts(z_all))

    a_frames = decode_mp4_gray(a_ctrl)
    c_frames = decode_mp4_gray(c_ctrl)
    assert_depth_levels(a_frames[0], "A control")
    assert_depth_levels(c_frames[0], "C control")
    print("A_control_8bit_valid_pcts", _pcts(a_frames[0][a_frames[0] > 8]))
    print("C_control_8bit_valid_pcts", _pcts(c_frames[0][c_frames[0] > 8]))

    report = {
        "meta": meta,
        "float_z_pcts": _pcts(z_all),
        "frames": {},
    }
    for fi in (0, 60, 120):
        a = a_frames[fi]
        c = c_frames[fi]
        mask_a = _castle_mask(a)
        mask_c = _castle_mask(c)
        # Use A mask for both so region matches
        mask = mask_a
        a_levels = np.unique(np.rint(a[mask]).astype(np.uint8))
        c_levels = np.unique(np.rint(c[mask]).astype(np.uint8))
        entry = {
            "A_castle_distinct": int(a_levels.size),
            "C_castle_distinct": int(c_levels.size),
            "A_castle_std": float(a[mask].std()) if mask.any() else 0.0,
            "C_castle_std": float(c[mask].std()) if mask.any() else 0.0,
            "A_castle_mean": float(a[mask].mean()) if mask.any() else 0.0,
            "C_castle_mean": float(c[mask].mean()) if mask.any() else 0.0,
            "castle_pixels": int(mask.sum()),
        }
        report["frames"][str(fi)] = entry
        print(f"frame_{fi}", entry)

    # Contact sheet A over C
    percents = [0, 25, 50, 75, 100]
    idxs = [min(count - 1, int(round(p / 100.0 * (count - 1)))) for p in percents]
    sheet = Image.new("RGB", (CLIP_WIDTH * len(idxs), CLIP_HEIGHT * 2), (0, 0, 0))
    a_rgb = decode_mp4_rgb(a_ctrl)
    c_rgb = decode_mp4_rgb(c_ctrl)
    for col, i in enumerate(idxs):
        sheet.paste(Image.fromarray(a_rgb[i], "RGB"), (col * CLIP_WIDTH, 0))
        sheet.paste(Image.fromarray(c_rgb[i], "RGB"), (col * CLIP_WIDTH, CLIP_HEIGHT))
    dest = gate / "control_compare.png"
    sheet.save(dest)
    print("control_compare", dest)

    # Plain-language summary
    means_a = [report["frames"][str(i)]["A_castle_std"] for i in (0, 60, 120)]
    means_c = [report["frames"][str(i)]["C_castle_std"] for i in (0, 60, 120)]
    print(
        "PLAIN:",
        f"Native A castle std={np.mean(means_a):.1f} vs C std={np.mean(means_c):.1f}; "
        f"A distinct~{report['frames']['0']['A_castle_distinct']} vs "
        f"C~{report['frames']['0']['C_castle_distinct']} at f0. "
        + (
            "Native depth is low-contrast / nearly flat on the castle compared with C."
            if np.mean(means_a) < 0.5 * np.mean(means_c)
            else "Native depth contrast on the castle is comparable to C."
        ),
    )
    (gate / "diagnose_s2.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
