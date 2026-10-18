"""A picture of what made an output, saved beside its log for review: one row of labeled
pictures per stage, each picture scaled to the same width."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# The band above each step that holds its label.
LABEL_HEIGHT = 26
LABEL_SIZE = 16
BACKGROUND = (30, 30, 30)
# Where no one is: a color no person or backdrop is drawn in.
NOBODY = np.array([255, 0, 255], dtype=np.float32)


def over_nobody(picture: np.ndarray, alpha: np.ndarray) -> Image.Image:
    """``picture`` where ``alpha`` (0..1) says a person is, magenta where no one is."""
    weight = alpha[..., None].astype(np.float32)
    return Image.fromarray(np.rint(picture * weight + NOBODY * (1.0 - weight)).astype(np.uint8))


def write_collage(rows: list[list[tuple[str, Image.Image]]], dest: Path, width: int) -> Path:
    """Each row's (label, picture) side by side, every picture ``width`` wide, the rows one under another."""
    font = ImageFont.load_default(size=LABEL_SIZE)
    strips = []
    for row in rows:
        tiles = []
        for label, picture in row:
            picture = picture.convert("RGB")
            picture = picture.resize((width, max(1, round(picture.height * width / picture.width))), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (width, picture.height + LABEL_HEIGHT), BACKGROUND)
            tile.paste(picture, (0, LABEL_HEIGHT))
            ImageDraw.Draw(tile).text((6, 4), label, fill=(255, 255, 255), font=font)
            tiles.append(tile)
        strip = Image.new("RGB", (width * len(tiles), max(tile.height for tile in tiles)), BACKGROUND)
        for index, tile in enumerate(tiles):
            strip.paste(tile, (index * width, 0))
        strips.append(strip)
    sheet = Image.new("RGB", (max(strip.width for strip in strips), sum(strip.height for strip in strips)), BACKGROUND)
    top = 0
    for strip in strips:
        sheet.paste(strip, (0, top))
        top += strip.height
    dest.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dest)
    return dest
