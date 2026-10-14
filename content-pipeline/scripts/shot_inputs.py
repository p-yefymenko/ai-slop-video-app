"""What the video model gets from the 3D scene and the plates: a depth video and one reference picture.

The less the model is given, the more natural the take. Two inputs proved
necessary, and each is the least that does its job:

- **Depth video** (every shot): where things are and how big, so the framing is
  the script's. People are flat silhouettes at their own distance: the clay's
  face relief, copied, freezes the face. One fixed log scale for every shot, so a
  lone person far away is not stretched to white.
- **Reference sheet** (shots with a face the model must get right): each such
  person's reference photo, cut out onto pure white. VACE copies a reference's
  backdrop into the take unless it is white, and paints a face for anyone on the
  sheet, so a person whose face the shot does not show is left to the words
  (``describe.faces_shown``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ffmpeg_tools import encode_rgb_frames
from observe import ground_depth
from render import render
from world import camera_at, entities_at

# Depth gray: 0.5 m and nearer is brightest, 60 m and farther is darkest, in log distance.
DEPTH_NEAR_METERS = 0.5
DEPTH_FAR_METERS = 60.0
# Gray range of a surface; black is reserved for nothing at all.
DEPTH_GRAY_FAR, DEPTH_GRAY_NEAR = 48, 255
# A person fills at most this share of the sheet's height.
SHEET_FILL = 0.95


def depth_gray(depth: np.ndarray) -> np.ndarray:
    """Meters to gray on the fixed log scale; ``inf`` (nothing there) is black."""
    gray = np.zeros(depth.shape, dtype=np.uint8)
    valid = np.isfinite(depth)
    span = np.log(DEPTH_FAR_METERS) - np.log(DEPTH_NEAR_METERS)
    near = np.clip((np.log(DEPTH_FAR_METERS) - np.log(np.maximum(depth[valid], DEPTH_NEAR_METERS))) / span, 0.0, 1.0)
    gray[valid] = np.rint(DEPTH_GRAY_FAR + near * (DEPTH_GRAY_NEAR - DEPTH_GRAY_FAR)).astype(np.uint8)
    return gray


def flatten_people(depth: np.ndarray, entity_ids: np.ndarray, people: list[int]) -> np.ndarray:
    """Each person (entity number in ``entity_ids``) at one depth, their median: a flat silhouette."""
    flat = depth.copy()
    for number in people:
        body = (entity_ids == number) & np.isfinite(flat)
        if body.any():
            flat[body] = np.median(flat[body])
    return flat


def depth_frame(show: dict, episode: dict, scene: dict, time_seconds: float, width: int, height: int) -> np.ndarray:
    """The shot at one moment as a gray depth picture, rendered at the video's own size."""
    camera = camera_at(scene, time_seconds)
    entities = entities_at(show, episode, scene, time_seconds)
    frame = render(entities, camera, width, height)
    people = [number + 1 for number, entity in enumerate(entities) if entity.kind == "character"]
    depth = flatten_people(frame.depth, frame.entity, people)
    # Past every mesh the ground runs on to the horizon: shaded at its own distance, never empty.
    depth = np.where(np.isfinite(depth), depth, ground_depth(camera, depth.shape)[0])
    return depth_gray(depth)


def write_depth_video(show: dict, episode: dict, scene: dict, width: int, height: int, length: int, fps: int, dest: Path) -> Path:
    """The shot's first ``length`` frames at ``fps`` as a depth video."""
    start = float(scene["timeRangeSeconds"][0])
    frames = (depth_frame(show, episode, scene, start + index / fps, width, height) for index in range(length))
    raw = b"".join(np.repeat(gray[:, :, None], 3, axis=2).tobytes() for gray in frames)
    dest.parent.mkdir(parents=True, exist_ok=True)
    encode_rgb_frames(raw, width, height, fps, dest, crf=14)
    return dest


def reference_sheet(photos: list[Path], width: int, height: int, dest: Path) -> Path:
    """The reference photos side by side on pure white, as drawn, one picture: VACE takes a single reference."""
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    slot = width // len(photos)
    for index, path in enumerate(photos):
        person = Image.open(path).convert("RGB")
        scale = min(slot / person.width, height * SHEET_FILL / person.height)
        person = person.resize((max(1, int(person.width * scale)), max(1, int(person.height * scale))), Image.Resampling.LANCZOS)
        sheet.paste(person, (index * slot + (slot - person.width) // 2, (height - person.height) // 2))
    dest.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dest)
    return dest
