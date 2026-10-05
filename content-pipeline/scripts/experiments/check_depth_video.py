#!/usr/bin/env python3

"""Acceptance checks for 24 fps previs depth video export."""

from __future__ import annotations

import _scripts_path  # noqa: F401

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from clip_spec import CLIP_HEIGHT, CLIP_WIDTH, FPS, clip_crop_box, clip_frame_count
from ffmpeg_tools import find_ffmpeg
from pipeline_paths import (
    clay_24fps_path,
    depth_video_dir,
    depth_video_mp4_path,
    discover_show_scripts,
)
from spatial_previs import (
    load_show,
    render_clothes_cutout,
    scene_has_spatial_change,
)


def _decode_mp4_rgb(path: Path) -> list[np.ndarray]:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise SystemExit("ffmpeg missing")
    raw = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    ).stdout
    frame_bytes = CLIP_WIDTH * CLIP_HEIGHT * 3
    if len(raw) % frame_bytes:
        raise SystemExit(f"{path} decoded size {len(raw)} is not a multiple of {frame_bytes}")
    frames = []
    for offset in range(0, len(raw), frame_bytes):
        frames.append(
            np.frombuffer(raw, dtype=np.uint8, count=frame_bytes, offset=offset).reshape(
                (CLIP_HEIGHT, CLIP_WIDTH, 3)
            )
        )
    return frames


def _read_depth_gray8(path: Path) -> np.ndarray:
    """Load a depth PNG as 8-bit gray without Pillow's broken I;16в†’RGB convert."""
    arr = np.asarray(Image.open(path), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    if arr.max() > 255.0:
        arr = np.rint(np.clip(arr / 65535.0, 0.0, 1.0) * 255.0)
    return arr.astype(np.uint8)


def _fit_gray8_to_clip(gray8: np.ndarray) -> np.ndarray:
    rgb = np.stack((gray8, gray8, gray8), axis=-1)
    from clip_spec import fit_to_clip

    fitted = fit_to_clip(Image.fromarray(rgb, "RGB"))
    return np.asarray(fitted)[:, :, 0].astype(np.uint8)


def _contact_sheet_from_mp4(
    depth_frames: list[np.ndarray],
    clay_frames: list[np.ndarray],
    destination: Path,
    percents: list[float],
) -> None:
    """Depth (from decoded mp4) on the top row, clay on the bottom, one column per percent."""
    count = len(depth_frames)
    cols = len(percents)
    sheet = Image.new("RGB", (CLIP_WIDTH * cols, CLIP_HEIGHT * 2), (0, 0, 0))
    for col, pct in enumerate(percents):
        index = min(count - 1, max(0, int(round((pct / 100.0) * (count - 1)))))
        depth = Image.fromarray(depth_frames[index], "RGB")
        clay = Image.fromarray(clay_frames[index], "RGB")
        sheet.paste(depth, (col * CLIP_WIDTH, 0))
        sheet.paste(clay, (col * CLIP_WIDTH, CLIP_HEIGHT))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)
    print(f"contact_sheet={destination}")


def _assert_not_binary(gray: np.ndarray, label: str) -> None:
    """Fail when a depth frame looks like a silhouette mask."""
    if gray.ndim != 2:
        raise SystemExit(f"{label}: expected 2D gray, got shape {gray.shape}")
    gray_f = gray.astype(np.float32)
    valid = gray_f > 0
    if not bool(valid.any()):
        raise SystemExit(f"{label}: no valid (non-empty) depth pixels")
    vals = gray_f[valid]
    lo, hi = float(vals.min()), float(vals.max())
    frac_extreme = float(((vals == lo) | (vals == hi)).mean())
    distinct = int(np.unique(np.rint(vals).astype(np.int32)).size)
    if frac_extreme >= 0.40:
        raise SystemExit(
            f"{label}: {frac_extreme:.1%} of valid pixels are exactly min ({lo:g}) or max ({hi:g}) "
            f"(need < 40%)"
        )
    if distinct < 100:
        raise SystemExit(f"{label}: only {distinct} distinct levels over the frame (need >= 100)")

    # Center column, bottom в†’ horizon. Skip over local landmark bumps (fire pit) by
    # taking the longest prefix from the bottom that stays non-increasing within
    # 1-level noise after a 3-tap median.
    col = gray.shape[1] // 2
    column = gray_f[:, col]
    nonzero = np.where(column > 0)[0]
    if nonzero.size < 100:
        raise SystemExit(
            f"{label}: center-column floor profile has only {nonzero.size} non-empty rows (need >= 100)"
        )
    bottom = int(nonzero[-1])
    top = int(nonzero[0])
    profile = column[list(range(bottom, top - 1, -1))]
    keep = len(profile)
    for i in range(1, len(profile)):
        if profile[i] <= 2 and profile[i - 1] - profile[i] > 8:
            keep = i
            break
    profile = profile[:keep]

    def _median3(values: np.ndarray) -> np.ndarray:
        if values.size < 3:
            return values
        return np.median(
            np.stack(
                [
                    np.pad(values, (1, 1), mode="edge")[0:-2],
                    values,
                    np.pad(values, (1, 1), mode="edge")[2:],
                ],
                axis=0,
            ),
            axis=0,
        )

    # Grow from the bottom while the smoothed ramp does not brighten by >1.
    end = 1
    med = _median3(profile)
    for i in range(1, len(med)):
        if med[i] > med[i - 1] + 1.0 + 1e-6:
            break
        end = i + 1
    # If a landmark interrupts early, continue after the bump and keep the longer ramp.
    best_slice = profile[:end]
    start = end
    while start < len(med) - 100:
        # Skip upward bump until the value falls back to the pre-bump level.
        anchor = float(med[start - 1]) if start else float(med[0])
        j = start
        while j < len(med) and med[j] > anchor + 1.0:
            j += 1
        if j >= len(med) - 1:
            break
        run_start = j
        run_end = run_start + 1
        for i in range(run_start + 1, len(med)):
            if med[i] > med[i - 1] + 1.0 + 1e-6:
                break
            run_end = i + 1
        candidate = profile[run_start:run_end]
        if candidate.size > best_slice.size:
            best_slice = candidate
        start = run_end
        if start <= run_start:
            start = run_start + 1

    distinct_floor = int(np.unique(np.rint(best_slice).astype(np.int32)).size)
    if distinct_floor < 100:
        raise SystemExit(
            f"{label}: center-column floor ramp has {distinct_floor} distinct levels (need >= 100)"
        )
    med_best = _median3(best_slice)
    if bool(np.any(np.diff(med_best) > 1.0 + 1e-6)):
        raise SystemExit(
            f"{label}: center-column floor is not monotonically non-increasing "
            f"(allowing 1-level noise)"
        )
    print(
        f"{label}: ok binary-check "
        f"frac_extreme={frac_extreme:.4f} distinct={distinct} floor_distinct={distinct_floor}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--scene", type=int, required=True)
    parser.add_argument("--flicker-scene", type=int, default=None)
    parser.add_argument(
        "--save-mp4-frame",
        type=int,
        default=None,
        help="Save this mp4 frame index as 8-bit PNG under tmp/depth_fix/",
    )
    args = parser.parse_args()

    scripts = discover_show_scripts(args.show)
    if not scripts:
        raise SystemExit("show not found")
    show = load_show(scripts[0])
    episode = next(ep for ep in show["episodes"] if ep["episodeNumber"] == args.episode)
    scene = next(sc for sc in episode["scenes"] if sc["sceneNumber"] == args.scene)

    time_range = scene["timeRangeSeconds"]
    expected = clip_frame_count(time_range)
    depth_dir = depth_video_dir(show["id"], args.episode, args.scene)
    meta_path = depth_dir / "meta.json"
    depth_mp4 = depth_video_mp4_path(show["id"], args.episode, args.scene)
    clay_mp4 = clay_24fps_path(show["id"], args.episode, args.scene)
    pngs = sorted(depth_dir.glob("frame_*.png"))

    crop = clip_crop_box(768, 1360)
    print(f"fit_to_clip crop for 768x1360: {crop} (w={crop[2]-crop[0]}, h={crop[3]-crop[1]})")
    print(f"timeRangeSeconds={time_range}")
    print(f"clip_frame_count={expected}")
    print(f"png_count={len(pngs)}")
    print(f"meta_exists={meta_path.is_file()}")
    print(f"depth_mp4_exists={depth_mp4.is_file()} size={depth_mp4.stat().st_size if depth_mp4.is_file() else 0}")
    print(f"clay_mp4_exists={clay_mp4.is_file()} size={clay_mp4.stat().st_size if clay_mp4.is_file() else 0}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    print(f"meta={json.dumps(meta)}")
    if len(pngs) != expected or meta.get("frameCount") != expected:
        raise SystemExit("frame count mismatch")

    probe = Image.open(pngs[0])
    print(f"png_mode={probe.mode} size={probe.size}")

    rgb_frames = _decode_mp4_rgb(depth_mp4)
    frames = [frame.mean(axis=2).astype(np.float32) for frame in rgb_frames]
    print(f"depth_mp4_frames={len(frames)}")
    if len(frames) != expected:
        raise SystemExit("mp4 frame count mismatch")

    # Binary-look assertions on PNG (fit_to_clip) and mp4 for every frame.
    for index, png_path in enumerate(pngs):
        png_clip = _fit_gray8_to_clip(_read_depth_gray8(png_path))
        _assert_not_binary(png_clip, f"png[{index}]")
        _assert_not_binary(rgb_frames[index][:, :, 0], f"mp4[{index}]")

    diffs = [float(np.mean(np.abs(frames[i + 1] - frames[i]))) for i in range(len(frames) - 1)]
    first12 = diffs[:12]
    print("stutter_mad_first12=" + ",".join(f"{value:.6f}" for value in first12))
    if len(diffs) >= 9:
        groups: list[list[float]] = [[], [], []]
        for index, value in enumerate(diffs):
            groups[index % 3].append(value)
        means = [float(np.mean(group)) for group in groups]
        print("stutter_group_means_mod3=" + ",".join(f"{value:.6f}" for value in means))

    clay_rgb = _decode_mp4_rgb(clay_mp4)
    sheet_path = depth_dir / "contact_sheet.png"
    _contact_sheet_from_mp4(rgb_frames, clay_rgb, sheet_path, [0, 25, 50, 75, 100])
    print(f"near={meta['near']:.6f} far={meta['far']:.6f}")

    if args.save_mp4_frame is not None:
        index = int(args.save_mp4_frame)
        if index < 0 or index >= len(rgb_frames):
            raise SystemExit(f"--save-mp4-frame {index} out of range 0..{len(rgb_frames) - 1}")
        out = Path("tmp") / "depth_fix" / f"frame{index}_from_mp4.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb_frames[index], "RGB").save(out)
        print(f"saved_mp4_frame={out.resolve()}")

    # Far-field 8-bit headroom report (frame mid/late in clip).
    sample = rgb_frames[min(len(rgb_frames) - 1, max(0, 33 if expected > 33 else expected // 2))]
    valid = sample[:, :, 0] > 0
    dark = sample[:, :, 0][(sample[:, :, 0] > 0) & (sample[:, :, 0] <= 32)]
    print(
        f"far_field_u8: valid={int(valid.sum())} "
        f"levels_1_32={int(np.unique(dark).size) if dark.size else 0} "
        f"frac_valid_le_16={float(((sample[:, :, 0] > 0) & (sample[:, :, 0] <= 16)).sum()) / max(int(valid.sum()), 1):.4f}"
    )

    flicker_scene_number = args.flicker_scene
    if flicker_scene_number is None:
        return
    flicker_scene = next(
        sc for sc in episode["scenes"] if sc["sceneNumber"] == flicker_scene_number
    )
    if not scene_has_spatial_change(episode, flicker_scene):
        raise SystemExit(f"flicker scene {flicker_scene_number} has no spatial change")
    start, finish = (float(v) for v in flicker_scene["timeRangeSeconds"])
    count = clip_frame_count([start, finish])
    sample_indexes = sorted(
        {
            0,
            count - 1,
            *[min(count - 1, max(0, int(round(frac * (count - 1))))) for frac in (0.25, 0.5, 0.75)],
            *range(0, count, max(1, FPS // 2)),
        }
    )
    person_any = None
    for index in sample_indexes:
        time_seconds = start + index / FPS
        clothes = np.asarray(
            render_clothes_cutout(show, episode, flicker_scene, time_seconds).convert("RGB")
        )
        person = np.any(clothes > 0, axis=2)
        person_any = person if person_any is None else (person_any | person)
    if person_any is None:
        raise SystemExit("flicker person mask is empty")
    from PIL import ImageFilter

    mask_img = Image.fromarray((person_any.astype(np.uint8) * 255), mode="L")
    mask_img = mask_img.filter(ImageFilter.MaxFilter(size=25))
    person_any = np.asarray(mask_img) > 0
    flicker_dir = depth_video_dir(show["id"], args.episode, flicker_scene_number)
    flicker_pngs = sorted(flicker_dir.glob("frame_*.png"))
    if len(flicker_pngs) != count:
        raise SystemExit(
            f"flicker scene {flicker_scene_number} missing depth frames "
            f"(have {len(flicker_pngs)}, want {count}); run previs on that scene first"
        )
    depth_frames = []
    for path in flicker_pngs:
        depth_frames.append(_read_depth_gray8(path).astype(np.float32))
    background = ~person_any
    bg_count = int(background.sum())
    max_change = 0.0
    for index in range(len(depth_frames) - 1):
        delta = np.abs(depth_frames[index + 1] - depth_frames[index])
        if bg_count:
            max_change = max(max_change, float(delta[background].max()))
    print(f"flicker_scene={flicker_scene_number}")
    print(f"flicker_timeRangeSeconds={[start, finish]}")
    print(f"flicker_frame_count={count}")
    print(f"flicker_background_pixels={bg_count}")
    print(f"flicker_max_bg_change={max_change:.6f}")


if __name__ == "__main__":
    main()
