"""The script as a physical world.

``entities_at`` is the one answer to "what exists in this location at this
moment, where, and turned which way". Characters, landmarks, and props are all
entities: a mesh, an offset, a yaw, and named regions. A character's regions
are body parts; anything else has one region, ``whole``. Every stage that draws,
measures, describes, or checks a shot starts from this list.

Presence comes from the timeline only. A character is in a shot when their
track puts them in the scene's location; nobody else decides it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from body_parts import BODY_PARTS, part_ids_for_vertices
from pipeline_paths import stage_dir

ROOT = Path(__file__).resolve().parents[1]
PROMPTS_PATH = ROOT / "prompts.json"
WHOLE = ("whole",)
FLOOR_ID = "floor"

Vec3 = tuple[float, float, float]


@lru_cache(maxsize=1)
def settings() -> dict:
    """``prompts.json``: templates and renderer numbers. Not show content."""
    return json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))


def load_show(path: Path) -> dict:
    """The script as authored. ``content:validate`` has already checked its shape."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clause(text: object) -> str:
    """Authored text as a clause: collapsed whitespace, no trailing punctuation."""
    return " ".join(str(text or "").split()).rstrip(".,;: ")


def _lerp(a, b, amount: float) -> list[float]:
    return [float(x) + (float(y) - float(x)) * amount for x, y in zip(a, b)]


def lerp_angle(a: float, b: float, amount: float) -> float:
    return a + ((b - a + 180.0) % 360.0 - 180.0) * amount


def _segment(track: list[dict], time_seconds: float) -> tuple[dict, dict | None, float]:
    """The keyframe at or before this time, the next one, and how far between them."""
    frames = sorted(track, key=lambda item: float(item["timeSeconds"]))
    before = frames[0]
    for after in frames[1:]:
        if float(after["timeSeconds"]) > time_seconds:
            span = float(after["timeSeconds"]) - float(before["timeSeconds"])
            amount = (time_seconds - float(before["timeSeconds"])) / span if span > 0 else 0.0
            return before, after, max(0.0, amount)
        before = after
    return before, None, 0.0


def character_state(track: list[dict], time_seconds: float) -> dict:
    """Location, feet position, yaw, and gaze target. Location ``None`` is off stage.

    Moving between two keyframes in one location is a straight walk. A change of
    location happens at the later keyframe.
    """
    before, after, amount = _segment(track, time_seconds)
    state = {
        "locationId": before.get("locationId"),
        "position": before.get("position"),
        "bodyYawDegrees": float(before.get("bodyYawDegrees") or 0.0),
        "lookAtId": before.get("lookAtId"),
    }
    if after is not None and after.get("locationId") == before.get("locationId") and before.get("position"):
        state["position"] = _lerp(before["position"], after["position"], amount)
        state["bodyYawDegrees"] = lerp_angle(
            state["bodyYawDegrees"], float(after.get("bodyYawDegrees") or 0.0), amount
        )
    return state


def prop_state(track: list[dict], time_seconds: float) -> dict:
    """Either held (``heldByCharacterId``) or placed (``locationId`` + ``position``)."""
    before, after, amount = _segment(track, time_seconds)
    state = dict(before)
    placed = lambda frame: frame is not None and frame.get("position") and not frame.get("heldByCharacterId")
    if placed(before) and placed(after) and after.get("locationId") == before.get("locationId"):
        state["position"] = _lerp(before["position"], after["position"], amount)
    return state


def camera_at(scene: dict, time_seconds: float) -> dict:
    """Interpolated camera pose. One keyframe holds the camera still."""
    before, after, amount = _segment(scene["camera"]["keyframes"], time_seconds)

    def pose(frame: dict) -> dict:
        return {
            "position": [float(value) for value in frame["position"]],
            "lookAt": [float(value) for value in frame["lookAt"]],
            "verticalFovDegrees": float(frame["verticalFovDegrees"]),
            "rollDegrees": float(frame.get("rollDegrees") or 0.0),
        }

    start = pose(before)
    if after is None:
        return start
    finish = pose(after)
    return {
        "position": _lerp(start["position"], finish["position"], amount),
        "lookAt": _lerp(start["lookAt"], finish["lookAt"], amount),
        "verticalFovDegrees": start["verticalFovDegrees"]
        + (finish["verticalFovDegrees"] - start["verticalFovDegrees"]) * amount,
        "rollDegrees": lerp_angle(start["rollDegrees"], finish["rollDegrees"], amount),
    }


def camera_moves(scene: dict) -> bool:
    """The camera pose differs between the first and last frame of the shot."""
    start, finish = (float(value) for value in scene["timeRangeSeconds"])
    first, last = camera_at(scene, start), camera_at(scene, finish)
    moved = lambda a, b: math.dist(a, b) >= 0.02
    return (
        moved(first["position"], last["position"])
        or moved(first["lookAt"], last["lookAt"])
        or abs(first["verticalFovDegrees"] - last["verticalFovDegrees"]) >= 0.5
        or abs((last["rollDegrees"] - first["rollDegrees"] + 180.0) % 360.0 - 180.0) >= 0.5
    )


def scene_has_spatial_change(episode: dict, scene: dict) -> bool:
    """The camera moves, or someone in this location moves or turns during the shot."""
    if camera_moves(scene):
        return True
    start, finish = (float(value) for value in scene["timeRangeSeconds"])
    for track in episode["spatialTimeline"]["characterTracks"].values():
        first, last = character_state(track, start), character_state(track, finish)
        if scene["locationId"] not in (first["locationId"], last["locationId"]):
            continue
        if first["locationId"] != last["locationId"] or math.dist(first["position"], last["position"]) >= 0.02:
            return True
        if abs((last["bodyYawDegrees"] - first["bodyYawDegrees"] + 180.0) % 360.0 - 180.0) >= 2.0:
            return True
    return False


@dataclass
class Mesh:
    """Schema-space triangles, faces grouped by region so each region draws as one range."""

    key: str
    vertices: np.ndarray
    faces: np.ndarray
    colors: np.ndarray | None
    # (region index, first face, face count), in face order.
    ranges: list[tuple[int, int, int]]
    # Named local points: "left_hand" and "right_hand" on a character.
    anchors: dict[str, np.ndarray]
    path: Path | None = None

    def __post_init__(self) -> None:
        # Local box, measured once; a mesh has millions of vertices.
        self.low = self.vertices.min(axis=0)
        self.high = self.vertices.max(axis=0)


@dataclass
class Entity:
    id: str
    kind: str  # character, landmark, prop, floor
    mesh: Mesh
    offset: Vec3
    yaw_degrees: float
    regions: tuple[str, ...]

    def to_world(self, local: np.ndarray) -> np.ndarray:
        yaw = math.radians(self.yaw_degrees)
        cosine, sine = math.cos(yaw), math.sin(yaw)
        points = np.atleast_2d(np.asarray(local, dtype=np.float64))
        world = np.empty_like(points)
        world[:, 0] = points[:, 0] * cosine + points[:, 1] * sine + self.offset[0]
        world[:, 1] = -points[:, 0] * sine + points[:, 1] * cosine + self.offset[1]
        world[:, 2] = points[:, 2] + self.offset[2]
        return world

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """World axis-aligned box around the mesh."""
        low, high = self.mesh.low, self.mesh.high
        corners = np.array(
            [[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])]
        )
        world = self.to_world(corners)
        return world.min(axis=0), world.max(axis=0)


class MissingMesh(Exception):
    pass


_MESHES: dict[str, Mesh] = {}


def _region_mesh(key: str, vertices: np.ndarray, faces: np.ndarray, colors, region_of_vertex, anchors=None) -> Mesh:
    face_regions = region_of_vertex[faces[:, 0]] if len(faces) else np.zeros(0, dtype=np.int16)
    order = np.argsort(face_regions, kind="stable")
    faces = np.ascontiguousarray(faces[order], dtype=np.uint32)
    face_regions = face_regions[order]
    ranges: list[tuple[int, int, int]] = []
    for region in np.unique(face_regions):
        hits = np.nonzero(face_regions == region)[0]
        ranges.append((int(region), int(hits[0]), int(len(hits))))
    return Mesh(key, np.ascontiguousarray(vertices, dtype=np.float32), faces, colors, ranges, anchors or {})


def load_mesh(path: Path, *, body: bool) -> Mesh:
    """A generated GLB in schema space. ``body`` labels the vertices with body parts."""
    from mesh_io import read_schema_mesh, read_vertex_colors

    if not path.is_file():
        raise MissingMesh(str(path))
    key = f"{path}:{path.stat().st_mtime_ns}"
    cached = _MESHES.get(str(path))
    if cached is not None and cached.key == key:
        return cached
    vertices, faces = read_schema_mesh(path)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.uint32)
    colors = read_vertex_colors(path)
    if colors is not None and len(colors) != len(vertices):
        colors = None
    if colors is not None:
        colors = np.ascontiguousarray(colors, dtype=np.float32)
    regions = part_ids_for_vertices(vertices) if body else np.zeros(len(vertices), dtype=np.int16)
    anchors = {}
    if body:
        hands = regions == BODY_PARTS.index("hands")
        for name, side in (("right_hand", vertices[:, 0] > 0), ("left_hand", vertices[:, 0] < 0)):
            if np.any(hands & side):
                anchors[name] = vertices[hands & side].mean(axis=0)
    mesh = _region_mesh(key, vertices, faces, colors, regions, anchors)
    mesh.path = path
    _MESHES[str(path)] = mesh
    return mesh


def _floor_mesh(size: Vec3) -> Mesh:
    width, depth = float(size[0]), float(size[1])
    vertices = np.array(
        [
            [-width / 2, -depth / 2, 0.0],
            [width / 2, -depth / 2, 0.0],
            [width / 2, depth / 2, 0.0],
            [-width / 2, depth / 2, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    return _region_mesh(f"floor:{width}x{depth}", vertices, faces, None, np.zeros(4, dtype=np.int16))


def mesh_path(show: dict, kind: str, entity_id: str, location_id: str | None = None) -> Path:
    """Where the generated mesh of one entity lives."""
    root = stage_dir("assets", str(show["id"]))
    if kind == "character":
        return root / "characters" / entity_id / "model.glb"
    if kind == "prop":
        return root / "props" / entity_id / "model.glb"
    return root / str(location_id) / entity_id / "model.glb"


def required_meshes(show: dict) -> list[Path]:
    """Every mesh the script needs. Previs cannot show the world without them."""
    paths = [mesh_path(show, "character", cid) for cid in show["characters"]]
    paths += [mesh_path(show, "prop", pid) for pid in show.get("props") or {}]
    for location_id, location in show["locations"].items():
        paths += [
            mesh_path(show, "landmark", landmark_id, location_id)
            for landmark_id in location["spatial"]["landmarks"]
        ]
    return paths


def camera_inside(camera: dict, size) -> bool:
    x, y, z = camera["position"]
    return abs(x) <= size[0] / 2 + 0.05 and abs(y) <= size[1] / 2 + 0.05 and -0.05 <= z <= size[2] + 0.05


def facing_yaw(position, body_yaw: float, target: Vec3 | None) -> float:
    """A mesh has one front. With a gaze target the whole body turns to it."""
    if target is not None and position is not None:
        dx, dy = target[0] - float(position[0]), target[1] - float(position[1])
        if math.hypot(dx, dy) > 1e-3:
            return math.degrees(math.atan2(dx, dy))
    return body_yaw


def entities_at(show: dict, episode: dict, scene: dict, time_seconds: float) -> list[Entity]:
    """Everything in the scene's location at this moment. A missing mesh raises MissingMesh."""
    location_id = scene["locationId"]
    spatial = show["locations"][location_id]["spatial"]
    timeline = episode["spatialTimeline"]
    landmarks = spatial["landmarks"]
    states = {cid: character_state(track, time_seconds) for cid, track in timeline["characterTracks"].items()}
    props = {pid: prop_state(track, time_seconds) for pid, track in timeline["propTracks"].items()}

    def here(cid: str) -> bool:
        return states.get(cid, {}).get("locationId") == location_id

    def target_point(target_id: str | None) -> Vec3 | None:
        if not target_id:
            return None
        if target_id in states and here(target_id):
            return tuple(states[target_id]["position"])
        if target_id in landmarks:
            return tuple(landmarks[target_id]["position"])
        prop = props.get(target_id) or {}
        holder = prop.get("heldByCharacterId")
        if holder and here(holder):
            return tuple(states[holder]["position"])
        if prop.get("locationId") == location_id and prop.get("position"):
            return tuple(prop["position"])
        return None

    found: list[Entity] = []
    for landmark_id, landmark in landmarks.items():
        found.append(
            Entity(
                landmark_id,
                "landmark",
                load_mesh(mesh_path(show, "landmark", landmark_id, location_id), body=False),
                tuple(float(v) for v in landmark["position"]),
                0.0,
                WHOLE,
            )
        )
    people: dict[str, Entity] = {}
    for cid, state in states.items():
        if not here(cid):
            continue
        yaw = facing_yaw(state["position"], state["bodyYawDegrees"], target_point(state.get("lookAtId")))
        people[cid] = Entity(
            cid,
            "character",
            load_mesh(mesh_path(show, "character", cid), body=True),
            tuple(float(v) for v in state["position"]),
            yaw,
            BODY_PARTS,
        )
    found.extend(people.values())
    for pid, state in props.items():
        holder = people.get(state.get("heldByCharacterId") or "")
        if holder is not None:
            hand = holder.mesh.anchors.get(f"{state.get('heldInHand') or 'right'}_hand")
            if hand is None:
                raise MissingMesh(f"{holder.id}'s mesh has no hand region to hold {pid}")
            offset = tuple(float(v) for v in holder.to_world(hand)[0])
            yaw = holder.yaw_degrees
        elif state.get("locationId") == location_id and state.get("position") and not state.get("heldByCharacterId"):
            offset, yaw = tuple(float(v) for v in state["position"]), 0.0
        else:
            continue
        found.append(Entity(pid, "prop", load_mesh(mesh_path(show, "prop", pid), body=False), offset, yaw, WHOLE))
    if camera_inside(camera_at(scene, time_seconds), spatial["sizeMeters"]):
        found.append(Entity(FLOOR_ID, "floor", _floor_mesh(spatial["sizeMeters"]), (0.0, 0.0, 0.0), 0.0, WHOLE))
    return found


def plate_clause(text: str) -> str:
    """Drop a trailing plate instruction such as 'a single object' or 'no walls'."""
    import re

    return re.split(r",\s*(?:a single |no )\b", text.strip(), maxsplit=1, flags=re.I)[0].strip(" .")


def descriptions(show: dict, scene: dict, kind: str, entity_id: str) -> list[tuple[str, tuple[str, ...]]]:
    """Words for a landmark, prop, or stretch of empty space, sent when it is visible.

    A person is not described in the shot: each is drawn whole in their own pass
    from ``character_appearance_text``.
    """
    if kind == "landmark":
        appearance = show["locations"][scene["locationId"]]["spatial"]["landmarks"][entity_id]["appearance"]
        return [(plate_clause(appearance), WHOLE)]
    if kind == "prop":
        return [(plate_clause(show["props"][entity_id]["appearance"]), WHOLE)]
    if kind == "backdrop":
        backdrop = show["locations"][scene["locationId"]]["backdrop"]
        return [(plate_clause(backdrop[entity_id]), WHOLE)]
    return []


def character_appearance_text(character: dict) -> str:
    """Plate text: body, then every attribute."""
    return ", ".join(clause(item) for item in [character["body"], *character["attributes"]])
