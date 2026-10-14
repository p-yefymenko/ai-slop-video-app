"""Check a shot's script against its 3D scene. Only exact facts, no thresholds:

- the camera is never inside a mesh;
- everyone with a performance is in the frame at some moment of the shot;
- whatever an action is aimed at (its target) is in the start frame, so the
  words can name it by where the frame shows it.
"""

from __future__ import annotations

import numpy as np

from observe import in_shot


def check_scene(show: dict, episode: dict, scene: dict, samples: list[tuple[float, dict, list]]) -> list[str]:
    """``samples`` are (time, observation, entities) across the shot, start first."""
    where = f"scene {scene['sceneNumber']}"
    start = samples[0][1]
    errors: list[str] = []

    inside: set[str] = set()
    for time_seconds, observation, entities in samples:
        camera = np.asarray(observation["camera"]["position"])
        for entity in entities:
            if entity.kind == "floor" or entity.id in inside:
                continue
            low, high = entity.bounds()
            if np.all(camera > low) and np.all(camera < high) and _inside_mesh(entity, camera):
                inside.add(entity.id)
                errors.append(f"{where} at {time_seconds:g}s: the camera is inside {entity.id}'s mesh.")

    for cid, performance in (scene.get("performances") or {}).items():
        if not any(in_shot(observation, cid) for _t, observation, _e in samples):
            errors.append(f"{where}: performances.{cid} is set, but {cid} is never in the frame.")
            continue
        target = performance.get("target")
        if target and not in_shot(start, target):
            errors.append(
                f"{where}: performances.{cid}.target is {target}, which is not in the start frame; "
                f"frame it, or drop the target and keep the action to their own body."
            )
    return errors


def _inside_mesh(entity, point: np.ndarray) -> bool:
    """Odd crossings of a ray from the point through the mesh means the point is inside it."""
    vertices = entity.to_world(entity.mesh.vertices)
    triangles = vertices[entity.mesh.faces]
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    direction = np.array([0.0, 0.0, 1.0])
    edge1, edge2 = b - a, c - a
    h = np.cross(direction, edge2)
    det = np.einsum("ij,ij->i", edge1, h)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = point - a
    u = np.einsum("ij,ij->i", s, h) * inv
    q = np.cross(s, edge1)
    v = (q @ direction) * inv
    t = np.einsum("ij,ij->i", edge2, q) * inv
    hits = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
    return bool(np.count_nonzero(hits) % 2)
