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


def _rays(camera: dict, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """The camera position and each pixel's ray, scaled so its camera-forward part is 1: a ray's
    parameter is then camera-forward meters, as in the rendered depth."""
    height, width = shape
    position, right, up, forward, focal = camera_basis(camera, height)
    ys, xs = np.indices((height, width))
    vx = (xs + 0.5 - width / 2.0) / focal
    vy = (height / 2.0 - (ys + 0.5)) / focal
    return np.asarray(position, dtype=np.float64), right[None, None, :] * vx[..., None] + up[None, None, :] * vy[..., None] + forward[None, None, :]


def backdrop_depth(camera: dict, shape: tuple[int, int], size, surround_height: float) -> tuple[np.ndarray, np.ndarray]:
    """Where each pixel's ray goes when no mesh is in the way: (camera-forward meters, inf for sky;
    label 1 sky, 2 ground inside the location, 3 surround).

    The surround stands on the location's edge as a wall ``surround_height`` meters tall (a forest,
    a city wall); a ray that leaves the location below its top meets it there. A ray over the wall
    that comes down meets the ground beyond, also surround; one that never comes down is sky.
    """
    position, ray = _rays(camera, shape)
    half = (float(size[0]) / 2.0, float(size[1]) / 2.0)
    down = ray[..., 2] < -1e-9
    ground = np.where(down, -position[2] / np.where(down, ray[..., 2], -1.0), np.inf)
    ground = np.where(ground > 0, ground, np.inf)
    # Where the ray enters and leaves the location's footprint (a slab per horizontal axis).
    enter = np.zeros(shape)
    leave = np.full(shape, np.inf)
    for axis in (0, 1):
        direction = ray[..., axis]
        moving = np.abs(direction) > 1e-12
        safe = np.where(moving, direction, 1.0)
        near = np.where(moving, (np.where(direction > 0, -half[axis], half[axis]) - position[axis]) / safe, np.where(abs(position[axis]) <= half[axis], -np.inf, np.inf))
        far = np.where(moving, (np.where(direction > 0, half[axis], -half[axis]) - position[axis]) / safe, np.where(abs(position[axis]) <= half[axis], np.inf, -np.inf))
        enter = np.maximum(enter, near)
        leave = np.minimum(leave, far)
    crosses = leave > enter
    wall_z = position[2] + np.where(np.isfinite(leave), leave, 0.0) * ray[..., 2]
    on_ground_inside = crosses & (ground >= enter) & (ground <= leave)
    on_wall = crosses & np.isfinite(leave) & ~on_ground_inside & (ground > leave) & (wall_z >= 0.0) & (wall_z <= surround_height)
    distance = np.where(on_wall, leave, ground)
    labels = np.where(np.isfinite(ground), 3, 1).astype(np.uint8)
    labels[on_wall] = 3
    labels[on_ground_inside] = 2
    return distance, labels


def backdrop_labels(camera: dict, frame, entities, size, surround_height: float) -> np.ndarray:
    """0 where a landmark, prop, or person is, else 1 sky, 2 ground (inside the location), 3 surround (beyond it)."""
    labels = backdrop_depth(camera, frame.entity.shape, size, surround_height)[1].copy()
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
        location = show["locations"][scene["locationId"]]
        labels = backdrop_labels(camera, frame, entities, location["spatial"]["sizeMeters"], location["backdrop"]["surroundHeightMeters"])
        for number, name in enumerate(BACKDROP, start=1):
            found[name] = {"kind": "backdrop", "pixels": int(np.count_nonzero(labels == number))}
        observation["effects"] = effects_in_frame(entities, camera)
    return observation, frame, entities
