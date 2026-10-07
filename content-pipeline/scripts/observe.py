"""What the camera sees at one moment, and the one rule for "visible".

An observation lists every entity in the location with, per region, the pixels
that are actually the front surface (``visible``) and the pixels the region would
cover with nothing in front of it, not even the rest of the same body (``extent``).
Sky, ground, and surround are entities too, measured on the empty-space plate.

``visible`` is the only test anything uses: a still names a description, a
check accepts a performance part, a sound source, or a speaker's face, all by it.
"""

from __future__ import annotations

import numpy as np

from render import HEIGHT, WIDTH, camera_basis, coverage, render
from world import camera_at, entities_at, settings

BACKDROP = ("sky", "ground", "surround")


def passes(visible_px: int, extent_px: int | None, minimum: int | None = None) -> bool:
    """Large enough, and not mostly hidden behind something, itself included.

    ``minimum`` defaults to ``visibleMinPixels`` (large enough to name in a
    still). A purpose that needs more, such as a mouth to lip-sync, passes its
    own minimum; the measurement and the hidden-share test stay the same.
    ``extent_px`` is None when the region was too small to be worth measuring.
    """
    config = settings()
    return (
        visible_px >= (config["visibleMinPixels"] if minimum is None else minimum)
        and extent_px is not None
        and visible_px >= config["visibleMinShare"] * extent_px
    )


def visible(observation: dict, entity_id: str, regions=None, minimum: int | None = None) -> bool:
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


def backdrop_labels(camera: dict, frame, entities, size) -> np.ndarray:
    """0 where a landmark, prop, or person is, else 1 sky, 2 ground, 3 surround.

    The floor is not an asset: it stays ground. A crack in a mesh is not sky, so
    small holes inside an asset stay asset.
    """
    from scipy import ndimage

    height, width = frame.depth.shape
    position, right, up, forward, focal = camera_basis(camera, height)
    ys, xs = np.indices((height, width))
    vx = (xs + 0.5 - width / 2.0) / focal
    vy = (height / 2.0 - (ys + 0.5)) / focal
    ray = right[None, None, :] * vx[..., None] + up[None, None, :] * vy[..., None] + forward[None, None, :]
    down = ray[..., 2] < -1e-4
    t = np.where(down, -position[2] / np.where(down, ray[..., 2], -1.0), np.inf)
    hit = np.isfinite(t) & (t > 0)
    hx = position[0] + np.where(hit, t, 0.0) * ray[..., 0]
    hy = position[1] + np.where(hit, t, 0.0) * ray[..., 1]
    inside = hit & (np.abs(hx) <= size[0] / 2.0) & (np.abs(hy) <= size[1] / 2.0)
    labels = np.full((height, width), 1, dtype=np.uint8)
    labels[hit & ~inside] = 3
    labels[inside] = 2
    floor = [index + 1 for index, entity in enumerate(entities) if entity.kind == "floor"]
    asset = (frame.entity > 0) & ~np.isin(frame.entity, floor)
    labels[ndimage.binary_fill_holes(ndimage.binary_closing(asset, iterations=3))] = 0
    return labels


def observe(show: dict, episode: dict, scene: dict, time_seconds: float, full: bool = True):
    """Render this moment and measure it. Returns (observation, frame, backdrop labels, entities).

    Every frame gets region visibility, which the checks need. ``full`` also
    measures where each entity sits (screen x, depth) and the empty space, which
    only a frame that becomes a still needs.
    """
    camera = camera_at(scene, time_seconds)
    entities = entities_at(show, episode, scene, time_seconds)
    frame = render(entities, camera, full=full)
    codes = frame.entity.astype(np.int32) * 256 + frame.region
    counts = np.bincount(codes.ravel(), minlength=256 * (len(entities) + 1))
    shown_total = np.bincount(frame.entity.ravel(), minlength=len(entities) + 1)
    if full:
        x_sum = np.bincount(frame.entity.ravel(), weights=np.tile(np.arange(WIDTH, dtype=np.float64), HEIGHT), minlength=len(entities) + 1)
    # Every purpose's minimum is at least this; smaller regions fail them all.
    minimum = min(settings()["visibleMinPixels"], settings()["onScreenMinPixels"])
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
