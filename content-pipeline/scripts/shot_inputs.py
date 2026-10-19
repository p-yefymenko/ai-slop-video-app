"""What the models get from the 3D scene and the plates.

- **Depth video** (a take VACE draws): where things are and how big, so the framing is the
  script's. People are flat silhouettes at their own distance: the clay's face relief, copied,
  freezes the face. One fixed log scale for every shot, so a lone person far away is not
  stretched to white.
- **Reference sheet** (the same takes): the reference photo of each of the shot's ``subjects``,
  side by side on white. VACE copies a reference's backdrop into the take unless it is white,
  and paints a face for anyone on the sheet, so only the people whose faces the shot shows are
  subjects (``describe.reference_people``).
- **Colored layout** (the start still, drawn by an image model): the shot's clay render at its
  start, each person, prop and landmark in it in their mesh's own colors (``colored_layout``), so
  the look of each comes from exactly the part of them the frame shows, not from words.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ffmpeg_tools import encode_rgb_frames
from observe import backdrop_depth
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
    """The shot at one moment as a gray depth picture, rendered at the video's own size, people
    flat: the clay's face relief, copied, froze the face."""
    camera = camera_at(scene, time_seconds)
    entities = entities_at(show, episode, scene, time_seconds)
    frame = render(entities, camera, width, height)
    depth = flatten_people(frame.depth, frame.entity, [number + 1 for number, entity in enumerate(entities) if entity.kind == "character"])
    # Past every mesh the ground runs on to the surround: shaded at its own distance, never empty.
    backdrop = show["locations"][scene["locationId"]]
    beyond = backdrop_depth(camera, depth.shape, backdrop["spatial"]["sizeMeters"], backdrop["backdrop"]["surroundHeightMeters"])[0]
    depth = np.where(np.isfinite(depth), depth, beyond)
    return depth_gray(depth)


def write_depth_video(show: dict, episode: dict, scene: dict, width: int, height: int, length: int, fps: int, dest: Path,
                      pace: float = 1.0) -> Path:
    """The shot's first ``length`` frames at ``fps`` as a depth video. ``pace`` is how many seconds
    of the script's timeline pass in one second of the take: below 1 when the scene was lengthened
    to hold its speech, so its blocking is spread over the longer take and never runs on into the
    next scene's."""
    start = float(scene["timeRangeSeconds"][0])
    frames = (depth_frame(show, episode, scene, start + index / fps * pace, width, height) for index in range(length))
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


# The clay's brightest value (``render.py``'s shader): a surface turned to the light keeps its own colors.
CLAY_LIGHTEST = 232.0


def colored_layout(clay: np.ndarray, albedo: np.ndarray, entity_ids: np.ndarray, colored: list[int],
                   floor: list[int], ground: tuple[int, int, int], sky: tuple[int, int, int]) -> np.ndarray:
    """The shot's render with each entity in ``colored`` (by entity number: those whose mesh has
    colors, from the picture it was made from) in its own colors, shaded as the clay is; the
    ``floor`` in the location's ``ground`` color, shaded the same; and where nothing is, the
    location's ``sky`` color. Both left one clay gray, the ground read as sky: from a camera
    looking down at nothing but ground, the model drew a horizon and trees."""
    out = clay.astype(np.float32)
    shade = out.mean(axis=2, keepdims=True) / CLAY_LIGHTEST
    where = np.isin(entity_ids, colored)
    out[where] = (albedo.astype(np.float32) * shade)[where]
    under = np.isin(entity_ids, floor)
    out[under] = (np.asarray(ground, dtype=np.float32) * shade)[under]
    out[entity_ids == 0] = np.asarray(sky, dtype=np.float32)
    return np.rint(out).clip(0, 255).astype(np.uint8)


def start_frame(show: dict, episode: dict, scene: dict, width: int, height: int):
    """The start frame at the still's size: (everyone with any pixel in it, in the script's order of
    characters; the entities; the render)."""
    start = float(scene["timeRangeSeconds"][0])
    entities = entities_at(show, episode, scene, start)
    frame = render(entities, camera_at(scene, start), width, height)
    numbers = {entity.id: number + 1 for number, entity in enumerate(entities) if entity.kind == "character"}
    seen = set(np.unique(frame.entity).tolist())
    return [cid for cid in show["characters"] if numbers.get(cid) in seen], entities, frame


def view_degrees(person, camera: dict) -> float:
    """How far round from a person's front ``camera`` stands, in degrees seen from above: 0 in front
    of them, 180 behind. A mesh faces its own +Y, turned by the person's yaw."""
    yaw = np.radians(person.yaw_degrees)
    facing = np.array([np.sin(yaw), np.cos(yaw)])
    toward = np.asarray(camera["position"], dtype=np.float64)[:2] - np.asarray(person.offset, dtype=np.float64)[:2]
    return float(np.degrees(np.arccos(np.clip(facing @ toward / np.linalg.norm(toward), -1.0, 1.0))))
