"""Check a shot's script against what its camera actually sees.

Every check uses ``observe.visible``, the same test a still uses to name
something. A script passes only when what it says on camera is on camera:

- anyone visible during the shot has a performance, and nobody else does;
- every part a performance moves is visible at some moment of the shot;
- an expression is drawn only on a face visible in the start frame;
- the speaker's face is large enough to lip-sync in the start frame;
- every sound source is visible in the start frame;
- the camera is never inside a mesh.
"""

from __future__ import annotations

import numpy as np

from observe import describe_numbers, visible
from world import settings


def on_screen(observation: dict, entity_id: str, regions=None) -> bool:
    """In view at all: the video model will animate these pixels, however small."""
    return visible(observation, entity_id, regions, settings()["onScreenMinPixels"])


def _numbers(observation: dict, entity_id: str, region: str | None = None) -> str:
    entry = observation["entities"].get(entity_id)
    if entry is None:
        return "not in this location"
    return describe_numbers(*(entry["regions"][region] if region else entry["pixels"]))


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

    first_seen: dict[str, tuple[float, dict]] = {}
    for time_seconds, observation, _entities in samples:
        for entity_id, entry in observation["entities"].items():
            if entry["kind"] == "character" and entity_id not in first_seen and on_screen(observation, entity_id):
                first_seen[entity_id] = (time_seconds, observation)

    performances = scene.get("performances") or {}
    for cid, (time_seconds, _observation) in first_seen.items():
        if cid not in performances:
            errors.append(
                f"{where}: {cid} is visible from {time_seconds:g}s but has no performance, so the video "
                f"model invents their motion. Add performances.{cid}, or move them or the camera."
            )
    for cid, performance in performances.items():
        if cid not in first_seen:
            state = start["entities"].get(cid)
            errors.append(
                f"{where}: performances.{cid} is set, but {cid} is never visible "
                f"({'not in ' + scene['locationId'] if state is None else _numbers(start, cid)} at the start)."
            )
            continue
        for part in performance["parts"]:
            if not any(on_screen(observation, cid, [part]) for _t, observation, _e in samples):
                errors.append(
                    f"{where}: performances.{cid}.parts names {part}, but their {part} is never visible "
                    f"in the shot (start: {_numbers(start, cid, part)}). Reframe, re-block, or change the action."
                )
        if performance.get("expression") and not on_screen(start, cid, ["face"]):
            errors.append(
                f"{where}: performances.{cid}.expression is set, but their face is not visible in the start "
                f"frame ({_numbers(start, cid, 'face')}), so nobody would see it. Remove it or reframe."
            )

    speaker = scene.get("speakerId")
    # A talking face needs a talking shot: about an eighth of the frame height (a
    # medium close-up). Smaller, the video model has a few pixels of mouth to speak with.
    talking = settings()["speakerFaceMinPixels"]
    if speaker and not visible(start, speaker, ["face"], talking):
        errors.append(
            f"{where}: speaker {speaker}'s face is {_numbers(start, speaker, 'face')} in the start frame; "
            f"a speaking face needs at least {talking} px (about an eighth of the frame height, a medium "
            f"close-up). Bring the camera closer or use a longer lens."
        )

    for index, event in enumerate(scene["sound"]["events"]):
        source = event["source"]
        entity_id = source.get("characterId") or source.get("landmarkId") or source.get("propId")
        part = source.get("part")
        if not on_screen(start, entity_id, [part] if part else None):
            errors.append(
                f"{where}: sound.events[{index}] comes from {entity_id}{' ' + part if part else ''}, which is "
                f"not visible in the start frame ({_numbers(start, entity_id, part)})."
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
