"""What the camera sees at one moment: which entities are in the frame, and where.

An observation lists every entity in the location with the pixels of it in the
frame, its mean screen x, and its median distance; sky, ground, and surround are
entries too (pixels where no mesh is, by where the ray goes). Effects in the frame
are listed by their host. Nothing here is a threshold: an entity is in the shot
when any pixel of it is (``in_shot``).
"""

from __future__ import annotations

import numpy as np

from render import HEIGHT, WIDTH, camera_basis, render
from world import camera_at, effect_entity, entities_at

BACKDROP = ("sky", "ground", "surround")


def in_shot(observation: dict, entity_id: str) -> bool:
    """Any pixel of it is in the frame."""
    entry = observation["entities"].get(entity_id)
    return entry is not None and entry["pixels"] > 0


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
    """0 where a landmark, prop, or person is, else 1 sky, 2 ground (inside the location), 3 surround (beyond it)."""
    t, hx, hy = ground_depth(camera, frame.entity.shape)
    hit = np.isfinite(t)
    inside = hit & (np.abs(hx) <= size[0] / 2.0) & (np.abs(hy) <= size[1] / 2.0)
    labels = np.full(frame.entity.shape, 1, dtype=np.uint8)
    labels[hit & ~inside] = 3
    labels[inside] = 2
    floor = [index + 1 for index, entity in enumerate(entities) if entity.kind == "floor"]
    labels[(frame.entity > 0) & ~np.isin(frame.entity, floor)] = 0
    return labels


def effects_in_frame(entities, camera: dict) -> list[dict]:
    """Every effect whose box has a pixel in the frame, by its host."""
    found = []
    for entity in entities:
        for number, effect in enumerate(entity.effects):
            if (render([effect_entity(entity, number)], camera, full=False).entity > 0).any():
                found.append({"id": entity.id, "appearance": effect["appearance"]})
    return found


def observe(show: dict, episode: dict, scene: dict, time_seconds: float, full: bool = True):
    """Render this moment and measure it. Returns (observation, frame, entities).

    ``full`` also measures where each entity sits (screen x, distance), the empty
    space, and the effects: what a frame that is described needs.
    """
    camera = camera_at(scene, time_seconds)
    entities = entities_at(show, episode, scene, time_seconds)
    frame = render(entities, camera, full=full)
    shown = np.bincount(frame.entity.ravel(), minlength=len(entities) + 1)
    found: dict[str, dict] = {}
    if full:
        x_sum = np.bincount(frame.entity.ravel(), weights=np.tile(np.arange(WIDTH, dtype=np.float64), HEIGHT), minlength=len(entities) + 1)
    for index, entity in enumerate(entities):
        if entity.kind == "floor":
            continue
        pixels = int(shown[index + 1])
        found[entity.id] = {"kind": entity.kind, "pixels": pixels}
        if full and pixels:
            found[entity.id]["screenX"] = round(float(x_sum[index + 1] / pixels), 1)
            found[entity.id]["depth"] = round(float(np.median(frame.depth[frame.entity == index + 1])), 3)
    observation = {"timeSeconds": time_seconds, "camera": camera, "entities": found}
    if full:
        labels = backdrop_labels(camera, frame, entities, show["locations"][scene["locationId"]]["spatial"]["sizeMeters"])
        for number, name in enumerate(BACKDROP, start=1):
            found[name] = {"kind": "backdrop", "pixels": int(np.count_nonzero(labels == number))}
        observation["effects"] = effects_in_frame(entities, camera)
    return observation, frame, entities
