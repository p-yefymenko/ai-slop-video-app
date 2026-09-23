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


def encode_rgb_frames(raw_rgb: bytes, width: int, height: int, fps: int, dest: Path) -> None:
    """Pack raw RGB24 frames into one H.264 MP4. Used for the model-free blockout."""
    if width % 2 or height % 2:
        raise RuntimeError(f"Blockout size {width}x{height} must be even for yuv420p")
    frame_bytes = width * height * 3
    if frame_bytes == 0 or len(raw_rgb) % frame_bytes:
        raise RuntimeError("Blockout frame buffer is not a whole number of frames")
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg is missing. Re-run `pnpm run content:setup-comfy` so imageio-ffmpeg is installed."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(dest),
        ],
        input=raw_rgb,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0 or not dest.exists() or dest.stat().st_size < 1024:
        detail = (result.stderr or b"").decode("utf-8", errors="replace").strip()[-2000:]
        raise RuntimeError(f"Failed to encode blockout {dest}\n{detail}")


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
