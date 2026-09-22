#!/usr/bin/env python3
"""Resolve a local ffmpeg binary (PATH or imageio-ffmpeg) for concat and thumbnails."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def find_ffmpeg() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
    except ImportError:
        install = subprocess.run(
            [sys.executable, "-m", "pip", "install", "imageio-ffmpeg"],
            check=False,
            capture_output=True,
            text=True,
        )
        if install.returncode != 0:
            return None
        try:
            import imageio_ffmpeg
        except ImportError:
            return None
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def concat_videos(scene_files: list[Path], dest: Path) -> None:
    if not scene_files:
        raise RuntimeError("No scene MP4s to concatenate")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if len(scene_files) == 1:
        shutil.copyfile(scene_files[0], dest)
        return
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg is missing. Re-run `pnpm run content:setup-comfy` so imageio-ffmpeg is installed."
        )
    list_file = dest.with_suffix(".concat.txt")
    lines = []
    for path in scene_files:
        escaped = path.resolve().as_posix().replace("'", r"'\''")
        lines.append(f"file '{escaped}'")
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        copy = subprocess.run(
            [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(dest)],
            check=False,
            capture_output=True,
            text=True,
        )
        if copy.returncode == 0 and dest.exists() and dest.stat().st_size > 1024:
            return
        encode = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_file),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                str(dest),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if encode.returncode != 0 or not dest.exists():
            detail = (encode.stderr or copy.stderr or "").strip()[-2000:]
            raise RuntimeError(f"Failed to concatenate scene MP4s into {dest}\n{detail}")
    finally:
        list_file.unlink(missing_ok=True)


def render_camera_move(
    still: Path,
    dest: Path,
    duration_seconds: float,
    horizontal_direction: float,
    vertical_direction: float = 0.0,
    frame_rate: int = 24,
) -> None:
    """Render a deterministic pan/creep for a silent anchor without generative drift."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg is missing. Re-run `pnpm run content:setup-comfy` so imageio-ffmpeg is installed."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    frames = max(2, round(duration_seconds * frame_rate))
    horizontal = max(-1.0, min(1.0, horizontal_direction))
    vertical = max(-1.0, min(1.0, vertical_direction))
    progress = f"on/{frames - 1}"
    zoom = f"min(1.14,1+0.14*{progress})"
    x = f"(iw-iw/zoom)/2*(1+({horizontal:.4f})*{progress})"
    y = f"(ih-ih/zoom)/2*(1+({vertical:.4f})*{progress})"
    # zoompan rounds crop coordinates to whole source pixels. Work at 4x and
    # downsample so the final movement advances in quarter-pixel increments
    # instead of visibly shaking between integer positions.
    video_filter = (
        "scale=iw*4:ih*4:flags=lanczos,"
        f"zoompan=z='{zoom}':x='{x}':y='{y}':d=1:s=1792x3200:fps={frame_rate},"
        "scale=448:800:flags=lanczos,format=yuv420p"
    )
    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loop",
            "1",
            "-i",
            str(still),
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=44100:cl=stereo",
            "-vf",
            video_filter,
            "-frames:v",
            str(frames),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            "-movflags",
            "+faststart",
            str(dest),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not dest.exists() or dest.stat().st_size <= 1024:
        detail = (result.stderr or "").strip()[-2000:]
        raise RuntimeError(f"Failed to render deterministic camera move {dest}\n{detail}")


def extract_thumbnail(mp4: Path, dest: Path) -> bool:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [ffmpeg, "-y", "-i", str(mp4), "-ss", "00:00:01", "-vframes", "1", str(dest)],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and dest.exists()
