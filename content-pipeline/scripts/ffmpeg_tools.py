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
