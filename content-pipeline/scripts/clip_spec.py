"""Shared LTX clip geometry and length helpers.

Union IC-LoRA ref0.5 needs latent spatial dims divisible by 2, so clip size is
448x768 (not 448x800). Previs/stills stay 768x1360; callers center-crop then
resize through fit_to_clip before feeding LTX.
"""

from __future__ import annotations

from typing import Sequence

from PIL import Image

CLIP_WIDTH = 448
CLIP_HEIGHT = 768
FPS = 24


def ltx_length_for_duration(duration_seconds: float, frame_rate: float) -> int:
    """LTX video length must be 8n+1 (9, 17, 25, ...)."""
    raw = max(1.0, float(duration_seconds) * float(frame_rate))
    n = max(1, round((raw - 1.0) / 8.0))
    return 8 * n + 1


def clip_frame_count(time_range_seconds: Sequence[float]) -> int:
    start, finish = (float(value) for value in time_range_seconds)
    return ltx_length_for_duration(max(0.0, finish - start), FPS)


def clip_crop_box(width: int, height: int) -> tuple[int, int, int, int]:
    """Center-crop box (left, top, right, bottom) matching CLIP_WIDTH:CLIP_HEIGHT."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size {width}x{height}")
    target_aspect = CLIP_WIDTH / CLIP_HEIGHT
    source_aspect = width / height
    if source_aspect > target_aspect:
        crop_w = max(1, int(round(height * target_aspect)))
        left = (width - crop_w) // 2
        return (left, 0, left + crop_w, height)
    crop_h = max(1, int(round(width / target_aspect)))
    top = (height - crop_h) // 2
    return (0, top, width, top + crop_h)


def fit_to_clip(image: Image.Image) -> Image.Image:
    """Center-crop to the clip aspect, then resize to CLIP_WIDTH x CLIP_HEIGHT."""
    rgb = image.convert("RGB")
    box = clip_crop_box(rgb.width, rgb.height)
    cropped = rgb.crop(box)
    if cropped.size == (CLIP_WIDTH, CLIP_HEIGHT):
        return cropped
    return cropped.resize((CLIP_WIDTH, CLIP_HEIGHT), Image.Resampling.LANCZOS)
