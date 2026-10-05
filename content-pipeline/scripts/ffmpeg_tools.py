#!/usr/bin/env python3
"""Resolve a local ffmpeg binary (PATH or imageio-ffmpeg) for concat and thumbnails."""

from __future__ import annotations

import json
import re
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


def find_ffprobe() -> str | None:
    found = shutil.which("ffprobe")
    if found:
        return found
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return None
    sibling = Path(ffmpeg).with_name("ffprobe.exe" if sys.platform == "win32" else "ffprobe")
    if sibling.is_file():
        return str(sibling)
    return None


def probe_video_stream(path: Path) -> dict:
    """Return width, height, and fps for the first video stream."""
    ffprobe = find_ffprobe()
    ffmpeg = find_ffmpeg()
    if ffprobe:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate,avg_frame_rate",
                "-of",
                "json",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()[-1000:]
            raise RuntimeError(f"ffprobe failed for {path}\n{detail}")
        payload = json.loads(result.stdout or "{}")
        streams = payload.get("streams") or []
        if not streams:
            raise RuntimeError(f"No video stream in {path}")
        stream = streams[0]
        width = int(stream["width"])
        height = int(stream["height"])
        rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
        if isinstance(rate, str) and "/" in rate:
            num, den = rate.split("/", 1)
            fps = float(num) / float(den) if float(den) else 0.0
        else:
            fps = float(rate)
        return {"width": width, "height": height, "fps": fps}
    if ffmpeg:
        # Fallback: parse ffmpeg -i stderr (no ffprobe on PATH).
        result = subprocess.run(
            [ffmpeg, "-i", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        text = (result.stderr or "") + (result.stdout or "")
        match = re.search(r"(\d{2,5})x(\d{2,5}).*?(\d+(?:\.\d+)?)\s*fps", text, re.I | re.S)
        if not match:
            raise RuntimeError(f"Could not probe video stream for {path}")
        return {
            "width": int(match.group(1)),
            "height": int(match.group(2)),
            "fps": float(match.group(3)),
        }
    raise RuntimeError(
        "ffprobe/ffmpeg missing. Re-run `pnpm run content:setup-comfy` so imageio-ffmpeg is installed."
    )


def assert_clips_match_for_concat(scene_files: list[Path]) -> None:
    """Fail before concat if width/height/fps are not identical across clips."""
    if len(scene_files) < 2:
        return
    probed: list[tuple[Path, dict]] = []
    for path in scene_files:
        probed.append((path, probe_video_stream(path)))
    ref_path, ref = probed[0]
    ref_key = (ref["width"], ref["height"], round(ref["fps"], 3))
    offenders: list[str] = []
    for path, info in probed[1:]:
        key = (info["width"], info["height"], round(info["fps"], 3))
        if key != ref_key:
            offenders.append(
                f"{path.name}: {info['width']}x{info['height']} @ {info['fps']:.3f}fps"
            )
    if offenders:
        ref_line = f"{ref_path.name}: {ref['width']}x{ref['height']} @ {ref['fps']:.3f}fps"
        raise RuntimeError(
            "Cannot concatenate episode.mp4: clips do not share the same width, height, and fps.\n"
            f"Reference: {ref_line}\n"
            "Offending clips:\n- "
            + "\n- ".join(offenders)
            + "\nRegenerate the mismatched clips with "
            "`pnpm run content:generate -- --show <id> --episode <n> --scene <n> --force` "
            "(do not auto-regenerate)."
        )


def concat_videos(scene_files: list[Path], dest: Path) -> None:
    if not scene_files:
        raise RuntimeError("No scene MP4s to concatenate")
    dest.parent.mkdir(parents=True, exist_ok=True)
    assert_clips_match_for_concat(scene_files)
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


def encode_rgb_frames(
    raw_rgb: bytes,
    width: int,
    height: int,
    fps: int,
    dest: Path,
    *,
    crf: int | None = None,
) -> None:
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
    cmd = [
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
    ]
    if crf is not None:
        cmd.extend(["-crf", str(int(crf))])
    cmd.extend(["-movflags", "+faststart", str(dest)])
    result = subprocess.run(
        cmd,
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
