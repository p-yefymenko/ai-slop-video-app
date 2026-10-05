#!/usr/bin/env python3

"""Metrics + contact sheets for the depth gate experiment."""

from __future__ import annotations

import _scripts_path  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from clip_spec import CLIP_HEIGHT, CLIP_WIDTH, fit_to_clip
from gate_depth_lib import (
    assert_depth_levels,
    decode_mp4_gray,
    decode_mp4_rgb,
    rgb_to_lab,
    scene_gate_dir,
    spearman_corr,
)
from pipeline_paths import discover_show_scripts, start_still_path
from spatial_previs import load_show

VARIANT_CONTROL = {
    "A": "depth_inverse.mp4",
    "A_gamma": "depth_inverse_gamma06.mp4",
    "A_eq": "depth_inverse_eq.mp4",
    "B": "depth_linear.mp4",
    "C": "depth_depthanything.mp4",
}
VARIANT_OUTPUT = {
    "A": "A_inverse.mp4",
    "A_gamma": "A_gamma06.mp4",
    "A_eq": "A_eq.mp4",
    "B": "B_linear.mp4",
    "C": "C_depthanything.mp4",
}


def _mad_series(frames: list[np.ndarray]) -> list[float]:
    return [
        float(np.mean(np.abs(frames[i + 1].astype(np.float32) - frames[i].astype(np.float32))))
        for i in range(len(frames) - 1)
    ]


def _camera_follow(control: list[np.ndarray], output_depth: list[np.ndarray]) -> dict:
    cors = []
    for c, o in zip(control, output_depth):
        mask = c > 8.0
        if int(mask.sum()) < 64:
            cors.append(float("nan"))
            continue
        cors.append(spearman_corr(c[mask].ravel(), o[mask].ravel()))
    vals = [v for v in cors if v == v]
    return {
        "mean": float(np.mean(vals)) if vals else float("nan"),
        "min": float(np.min(vals)) if vals else float("nan"),
        "per_frame": cors,
    }


def _appearance_delta(control: list[np.ndarray], rgb_frames: list[np.ndarray]) -> dict:
    idxs = [0]
    for frac in (0.25, 0.5, 0.75, 1.0):
        idxs.append(min(len(rgb_frames) - 1, int(round(frac * (len(rgb_frames) - 1)))))
    idxs = sorted(set(idxs))
    first = rgb_to_lab(rgb_frames[0])
    ctrl0 = control[0]
    median = float(np.median(ctrl0[ctrl0 > 8])) if np.any(ctrl0 > 8) else 0.0
    results = {}
    for idx in idxs[1:]:
        static = (ctrl0 > median) & (np.abs(control[idx] - ctrl0) <= 2.0)
        if not bool(static.any()):
            results[str(idx)] = None
            continue
        lab = rgb_to_lab(rgb_frames[idx])
        delta = np.linalg.norm(lab[static] - first[static], axis=-1)
        results[str(idx)] = float(delta.mean())
    usable = [v for v in results.values() if v is not None]
    return {"per_index": results, "mean": float(np.mean(usable)) if usable else None}


def _first_frame_delta(still: Path, frame0: np.ndarray) -> float:
    fitted = np.asarray(fit_to_clip(Image.open(still).convert("RGB")), dtype=np.float32)
    return float(np.mean(np.abs(fitted - frame0.astype(np.float32))))


def _castle_mask_from_control(gray: np.ndarray) -> np.ndarray:
    valid = gray > 8.0
    if not bool(valid.any()):
        return valid
    thresh = float(np.percentile(gray[valid], 60.0))
    return valid & (gray >= thresh)


def _drift(rgb_frames: list[np.ndarray], control: list[np.ndarray] | None, region: bool) -> dict:
    n = len(rgb_frames)
    idxs = [min(n - 1, int(round(p / 100.0 * (n - 1)))) for p in (0, 25, 50, 75, 100)]
    L_series = []
    C_series = []
    for i in idxs:
        lab = rgb_to_lab(rgb_frames[i])
        if region and control is not None:
            mask = _castle_mask_from_control(control[i])
            if not bool(mask.any()):
                L_series.append(None)
                C_series.append(None)
                continue
            L = lab[..., 0][mask]
            a = lab[..., 1][mask]
            b = lab[..., 2][mask]
        else:
            L = lab[..., 0].ravel()
            a = lab[..., 1].ravel()
            b = lab[..., 2].ravel()
        # OpenCV Lab a/b are 0вЂ“255 with neutral at 128.
        chroma = np.sqrt((a.astype(np.float32) - 128.0) ** 2 + (b.astype(np.float32) - 128.0) ** 2)
        L_series.append(float(L.mean()))
        C_series.append(float(chroma.mean()))
    L0 = L_series[0]
    C0 = C_series[0]
    L_max = max(abs(v - L0) for v in L_series if v is not None and L0 is not None) if L0 is not None else None
    C_max = max(abs(v - C0) for v in C_series if v is not None and C0 is not None) if C0 is not None else None
    return {
        "indexes": idxs,
        "L_mean": L_series,
        "chroma_mean": C_series,
        "L_max_abs_delta_from_0": L_max,
        "chroma_max_abs_delta_from_0": C_max,
    }


def measure_variant(
    scene_dir: Path,
    variant: str,
    still: Path,
    output_depth_da: Path | None,
    *,
    castle_region: bool,
) -> dict:
    control_path = scene_dir / VARIANT_CONTROL[variant]
    out_path = scene_dir / VARIANT_OUTPUT[variant]
    if not out_path.is_file():
        alt = scene_dir / VARIANT_OUTPUT[variant].replace(".mp4", "_dialogue.mp4")
        if alt.is_file():
            out_path = alt
    if not control_path.is_file() or not out_path.is_file():
        raise SystemExit(f"missing files for {variant}: {control_path} / {out_path}")

    control = decode_mp4_gray(control_path)
    assert_depth_levels(control[0], f"control {variant}")
    rgb = decode_mp4_rgb(out_path)
    if len(rgb) != len(control):
        raise SystemExit(f"{variant}: frame count mismatch control={len(control)} out={len(rgb)}")

    follow = {"mean": None, "min": None, "note": "missing output DA depth"}
    if output_depth_da and output_depth_da.is_file():
        out_depth = decode_mp4_gray(output_depth_da)
        if len(out_depth) != len(control):
            follow = {"mean": None, "min": None, "note": "DA depth frame mismatch"}
        else:
            follow = _camera_follow(control, out_depth)

    mads = _mad_series([f.mean(axis=2) for f in rgb])
    groups: list[list[float]] = [[], [], []]
    for i, v in enumerate(mads):
        groups[i % 3].append(v)
    mod3 = [float(np.mean(g)) if g else float("nan") for g in groups]
    appear = _appearance_delta(control, rgb)
    first = _first_frame_delta(still, rgb[0])
    run_meta = {}
    run_json = scene_dir / f"{out_path.stem}_run.json"
    if run_json.is_file():
        run_meta = json.loads(run_json.read_text(encoding="utf-8"))

    drift_full = _drift(rgb, control, region=False)
    drift_castle = _drift(rgb, control, region=True) if castle_region else None

    return {
        "variant": variant,
        "output": str(out_path),
        "control": str(control_path),
        "wall_seconds": run_meta.get("wall_seconds"),
        "peak_vram_mib": run_meta.get("peak_vram_mib"),
        "camera_follow": follow,
        "stutter_first12": mads[:12],
        "stutter_mod3": mod3,
        "appearance": appear,
        "first_frame_delta": first,
        "brightness_drift": drift_full,
        "saturation_drift": {
            "indexes": drift_full["indexes"],
            "chroma_mean": drift_full["chroma_mean"],
            "chroma_max_abs_delta_from_0": drift_full["chroma_max_abs_delta_from_0"],
        },
        "brightness_drift_castle": drift_castle,
        "saturation_drift_castle": (
            {
                "indexes": drift_castle["indexes"],
                "chroma_mean": drift_castle["chroma_mean"],
                "chroma_max_abs_delta_from_0": drift_castle["chroma_max_abs_delta_from_0"],
            }
            if drift_castle
            else None
        ),
        "frames": len(rgb),
    }


def write_contact_sheet_per_variant(scene_dir: Path, still: Path, variant: str) -> Path:
    """Rows: start still, control fed to LTX, output. Columns: 0/25/50/75/100%."""
    control_path = scene_dir / VARIANT_CONTROL[variant]
    out_path = scene_dir / VARIANT_OUTPUT[variant]
    if not out_path.is_file():
        alt = scene_dir / VARIANT_OUTPUT[variant].replace(".mp4", "_dialogue.mp4")
        if alt.is_file():
            out_path = alt
    if not control_path.is_file() or not out_path.is_file():
        raise SystemExit(f"contact sheet missing {control_path} or {out_path}")
    start = fit_to_clip(Image.open(still).convert("RGB"))
    control = decode_mp4_rgb(control_path)
    assert_depth_levels(control[0][:, :, 0].astype(np.float32), f"contact {variant}")
    frames = decode_mp4_rgb(out_path)
    n = len(frames)
    indexes = [min(n - 1, int(round(p / 100.0 * (n - 1)))) for p in (0, 25, 50, 75, 100)]
    sheet = Image.new("RGB", (CLIP_WIDTH * len(indexes), CLIP_HEIGHT * 3), (0, 0, 0))
    for col, i in enumerate(indexes):
        sheet.paste(start, (col * CLIP_WIDTH, 0))
        sheet.paste(Image.fromarray(control[i], "RGB"), (col * CLIP_WIDTH, CLIP_HEIGHT))
        sheet.paste(Image.fromarray(frames[i], "RGB"), (col * CLIP_WIDTH, CLIP_HEIGHT * 2))
    dest = scene_dir / f"contact_sheet_{variant}.png"
    sheet.save(dest)
    print(f"contact_sheet={dest}", flush=True)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", default="the-iron-bride")
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--scene", type=int, required=True)
    parser.add_argument("--variants", default="A,A_gamma,B,C")
    parser.add_argument("--da-outputs", default="", help="comma variant=path for DA-on-output mp4s")
    parser.add_argument(
        "--castle-region",
        action="store_true",
        help="Also compute brightness/saturation drift on nearest-40% castle mask (S2)",
    )
    args = parser.parse_args()
    show = load_show(discover_show_scripts(args.show)[0])
    still = start_still_path(show["id"], args.episode, args.scene)
    if not still.is_file():
        raise SystemExit(f"missing still {still}")
    scene_dir = scene_gate_dir(args.scene)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    da_map: dict[str, Path] = {}
    if args.da_outputs:
        for part in args.da_outputs.split(","):
            if "=" in part:
                k, p = part.split("=", 1)
                da_map[k.strip()] = Path(p.strip())

    results = []
    for v in variants:
        results.append(
            measure_variant(
                scene_dir,
                v,
                still,
                da_map.get(v),
                castle_region=args.castle_region,
            )
        )
        write_contact_sheet_per_variant(scene_dir, still, v)
    out = scene_dir / "metrics_3b.json"
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
