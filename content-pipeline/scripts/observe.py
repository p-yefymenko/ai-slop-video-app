"""What the camera sees at one moment: what is in the shot, and what is visible enough.

An observation lists every entity in the location with, per region, the pixels
that are actually the front surface (``visible``) and the pixels the region would
cover with nothing in front of it, not even the rest of the same body (``extent``).
Sky, ground, and surround are entities too, measured on the empty-space plate.

Two tests, for two different jobs:

- ``in_shot``: any pixel of it is in the frame. Everything in the shot is drawn
  (or, for sky, ground, and surround, said), so there is no size cutoff that
  could leave words without a place or a place without its picture.
- ``visible``: large enough for a purpose and not mostly hidden. Only the
  script checks use it (a performance part, a sound source, a speaker's face).
"""

from __future__ import annotations

import numpy as np

from render import HEIGHT, WIDTH, camera_basis, coverage, render
from world import camera_at, effect_entity, entities_at, settings

BACKDROP = ("sky", "ground", "surround")


def in_shot(observation: dict, entity_id: str) -> bool:
    """Any pixel of it is in the frame."""
    entry = observation["entities"].get(entity_id)
    return entry is not None and entry["pixels"][0] > 0


def passes(visible_px: int, extent_px: int | None, minimum: int) -> bool:
    """At least ``minimum`` pixels, and not mostly hidden behind something, itself included.

    Each check passes its own minimum (on screen, a mouth to lip-sync); the
    measurement and the hidden-share test stay the same. ``extent_px`` is None
    when the region was too small to be worth measuring.
    """
    config = settings()
    return (
        visible_px >= minimum
        and extent_px is not None
        and visible_px >= config["visibleMinShare"] * extent_px
    )


def visible(observation: dict, entity_id: str, regions, minimum: int) -> bool:
    """True when any of these regions (default: any region) passes."""
    entry = observation["entities"].get(entity_id)
    if entry is None:
        return False
    names = entry["regions"] if regions is None else [name for name in regions if name in entry["regions"]]
    return any(passes(*entry["regions"][name], minimum) for name in names)


def visible_share(entry: dict) -> float:
    shown, extent = entry["pixels"]
    return shown / extent if extent else 0.0


def describe_numbers(visible_px: int, extent_px: int | None) -> str:
    if extent_px is None:
        return f"{visible_px} px visible, too small to measure"
    return f"{visible_px} px visible of {extent_px} px"


def ground_depth(camera: dict, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where each pixel's ray meets the ground plane (z=0) out to the horizon: (camera-forward
    meters, inf above the horizon; world x; world y)."""
    height, width = shape
    position, right, up, forward, focal = camera_basis(camera, height)
    ys, xs = np.indices((height, width))
    vx = (xs + 0.5 - width / 2.0) / focal
    vy = (height / 2.0 - (ys + 0.5)) / focal
    ray = right[None, None, :] * vx[..., None] + up[None, None, :] * vy[..., None] + forward[None, None, :]
    down = ray[..., 2] < -1e-4
    t = np.where(down, -position[2] / np.where(down, ray[..., 2], -1.0), np.inf)
    t = np.where(t > 0, t, np.inf)
    return t, position[0] + np.where(np.isfinite(t), t, 0.0) * ray[..., 0], position[1] + np.where(np.isfinite(t), t, 0.0) * ray[..., 1]


def backdrop_labels(camera: dict, frame, entities, size) -> np.ndarray:
    """0 where a landmark, prop, or person is, else 1 sky, 2 ground, 3 surround.

    The floor is not an asset: it stays ground. A crack in a mesh is not sky, so
    small holes inside an asset stay asset.
    """
    from scipy import ndimage

    t, hx, hy = ground_depth(camera, frame.depth.shape)
    hit = np.isfinite(t)
    inside = hit & (np.abs(hx) <= size[0] / 2.0) & (np.abs(hy) <= size[1] / 2.0)
    labels = np.full(frame.depth.shape, 1, dtype=np.uint8)
    labels[hit & ~inside] = 3
    labels[inside] = 2
    floor = [index + 1 for index, entity in enumerate(entities) if entity.kind == "floor"]
    asset = (frame.entity > 0) & ~np.isin(frame.entity, floor)
    labels[ndimage.binary_fill_holes(ndimage.binary_closing(asset, iterations=3))] = 0
    return labels


def _seen_through_effects(entities, camera, frame):
    """The frame as the checks see it: fire, smoke, or steam in front hides what is behind.

    Effects are never in the guides, the clay, or the depth video; only what
    counts as visible accounts for them. Without effects this is the frame.
    """
    boxes = [effect_entity(entity, number) for entity in entities for number in range(len(entity.effects))]
    if not boxes:
        return frame
    blocked = render([*entities, *boxes], camera, full=False)
    hidden = blocked.entity > len(entities)
    blocked.entity[hidden] = 0
    blocked.region[hidden] = 0
    return blocked


def observe(show: dict, episode: dict, scene: dict, time_seconds: float, full: bool = True):
    """Render this moment and measure it. Returns (observation, frame, backdrop labels, entities).

    Every frame gets region visibility, which the checks need. ``full`` also
    measures where each entity sits (screen x, depth) and the empty space, which
    only a frame that becomes a still needs.
    """
    camera = camera_at(scene, time_seconds)
    entities = entities_at(show, episode, scene, time_seconds)
    frame = render(entities, camera, full=full)
    seen = _seen_through_effects(entities, camera, frame)
    codes = seen.entity.astype(np.int32) * 256 + seen.region
    counts = np.bincount(codes.ravel(), minlength=256 * (len(entities) + 1))
    # In shot at all (drawn into the still) counts what is behind an effect too.
    shown_total = np.bincount(frame.entity.ravel(), minlength=len(entities) + 1)
    if full:
        x_sum = np.bincount(frame.entity.ravel(), weights=np.tile(np.arange(WIDTH, dtype=np.float64), HEIGHT), minlength=len(entities) + 1)
    # Every check's minimum is at least this; smaller regions fail them all.
    minimum = settings()["onScreenMinPixels"]
    found: dict[str, dict] = {}
    for index, entity in enumerate(entities):
        if entity.kind == "floor":
            continue
        shown_by_region = {region: int(counts[(index + 1) * 256 + region + 1]) for region in range(len(entity.regions))}
        # A region under the minimum fails whatever its extent, so only the rest are measured.
        measured = {region for region, shown in shown_by_region.items() if shown >= minimum}
        extents, union = coverage(entity, camera, measured) if measured else ({}, None)
        regions = {
            name: [shown_by_region[region], extents.get(region)] for region, name in enumerate(entity.regions)
        }
        shown = int(shown_total[index + 1])
        found[entity.id] = {"kind": entity.kind, "pixels": [shown, union], "regions": regions}
        if full:
            found[entity.id]["screenX"] = round(float(x_sum[index + 1] / shown), 1) if shown else None
            found[entity.id]["depth"] = (
                round(float(np.median(frame.depth[frame.entity == index + 1])), 3) if shown else None
            )
    if not full:
        return {"timeSeconds": time_seconds, "camera": camera, "entities": found}, frame, None, entities
    labels = backdrop_labels(camera, frame, entities, show["locations"][scene["locationId"]]["spatial"]["sizeMeters"])
    for number, name in enumerate(BACKDROP, start=1):
        pixels = int(np.count_nonzero(labels == number))
        found[name] = {"kind": "backdrop", "pixels": [pixels, pixels], "regions": {"whole": [pixels, pixels]}}
    return {"timeSeconds": time_seconds, "camera": camera, "entities": found}, frame, labels, entities
