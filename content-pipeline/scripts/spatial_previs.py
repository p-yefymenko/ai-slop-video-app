#!/usr/bin/env python3
"""Render a blocked scene from the stage timeline, with no image or video model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from pipeline_paths import (
    blockout_video_path,
    clay_frame_path,
    contact_sheet_path,
    discover_show_scripts,
    episode_blockout_path,
    guide_path,
)

PROXY_WIDTH = 768
PROXY_HEIGHT = 1360
NEAR_CLIP = 0.05
# Empty viewport, matching a clay playblast: gray where no surface exists.
VIEWPORT_GRAY = (148, 149, 152)
IDENTITY_FACE_LIMIT = 2
BLOCKOUT_FPS = 8

Vec3 = tuple[float, float, float]


def vec(raw: Iterable[float]) -> Vec3:
    values = tuple(float(value) for value in raw)
    if len(values) != 3:
        raise ValueError(f"Expected a 3D vector, got {values}")
    return values  # type: ignore[return-value]


def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def mul(a: Vec3, scale: float) -> Vec3:
    return (a[0] * scale, a[1] * scale, a[2] * scale)


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def length(a: Vec3) -> float:
    return math.sqrt(dot(a, a))


def normalize(a: Vec3) -> Vec3:
    magnitude = length(a)
    if magnitude < 1e-6:
        raise ValueError("Cannot normalize a zero-length vector")
    return mul(a, 1.0 / magnitude)


def lerp(a: Vec3, b: Vec3, amount: float) -> Vec3:
    return add(a, mul(sub(b, a), amount))


def lerp_angle(a: float, b: float, amount: float) -> float:
    delta = (b - a + 180.0) % 360.0 - 180.0
    return a + delta * amount


def load_show(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def timeline_state(track: list[dict], time_seconds: float) -> dict:
    if not track:
        raise ValueError("Spatial track cannot be empty")
    frames = sorted(track, key=lambda item: float(item["timeSeconds"]))
    if time_seconds <= float(frames[0]["timeSeconds"]):
        return dict(frames[0])
    if time_seconds >= float(frames[-1]["timeSeconds"]):
        return dict(frames[-1])
    for frame in frames:
        if abs(time_seconds - float(frame["timeSeconds"])) < 1e-6:
            return dict(frame)
    for before, after in zip(frames, frames[1:]):
        start = float(before["timeSeconds"])
        finish = float(after["timeSeconds"])
        if start <= time_seconds <= finish:
            if before["locationId"] != after["locationId"]:
                return dict(before if time_seconds < finish else after)
            amount = (time_seconds - start) / max(finish - start, 1e-6)
            state = dict(before)
            state["timeSeconds"] = time_seconds
            state["position"] = list(
                lerp(vec(before["position"]), vec(after["position"]), amount)
            )
            state["bodyYawDegrees"] = lerp_angle(
                float(before["bodyYawDegrees"]),
                float(after["bodyYawDegrees"]),
                amount,
            )
            for discrete in (
                "lookAtId",
                "stance",
                "leftHandTargetId",
                "rightHandTargetId",
            ):
                state[discrete] = before.get(discrete)
            return state
    return dict(frames[-1])


def episode_character_state(episode: dict, character_id: str, time_seconds: float) -> dict:
    timeline = episode.get("spatialTimeline") or {}
    tracks = timeline.get("characterTracks") or {}
    if character_id not in tracks:
        raise ValueError(f"Spatial timeline has no track for character {character_id!r}")
    return timeline_state(tracks[character_id], time_seconds)


def _camera_pose(frame: dict) -> dict:
    return {
        "position": list(vec(frame["position"])),
        "lookAt": list(vec(frame["lookAt"])),
        "verticalFovDegrees": float(frame["verticalFovDegrees"]),
        "rollDegrees": float(frame.get("rollDegrees") or 0.0),
    }


def camera_keyframes(scene: dict) -> list[dict]:
    camera = scene.get("camera") or {}
    frames = camera.get("keyframes")
    if not isinstance(frames, list) or not frames:
        raise ValueError(
            f"scene {scene.get('sceneNumber', '?')} requires camera.keyframes"
        )
    return sorted(frames, key=lambda item: float(item["timeSeconds"]))


def camera_at(scene: dict, time_seconds: float) -> dict:
    frames = camera_keyframes(scene)
    if time_seconds <= float(frames[0]["timeSeconds"]):
        return _camera_pose(frames[0])
    if time_seconds >= float(frames[-1]["timeSeconds"]):
        return _camera_pose(frames[-1])
    for before, after in zip(frames, frames[1:]):
        start = float(before["timeSeconds"])
        finish = float(after["timeSeconds"])
        if start <= time_seconds <= finish:
            amount = (time_seconds - start) / max(finish - start, 1e-6)
            start_pose = _camera_pose(before)
            end_pose = _camera_pose(after)
            return {
                "position": list(
                    lerp(vec(start_pose["position"]), vec(end_pose["position"]), amount)
                ),
                "lookAt": list(
                    lerp(vec(start_pose["lookAt"]), vec(end_pose["lookAt"]), amount)
                ),
                "verticalFovDegrees": start_pose["verticalFovDegrees"]
                + (end_pose["verticalFovDegrees"] - start_pose["verticalFovDegrees"])
                * amount,
                "rollDegrees": lerp_angle(
                    start_pose["rollDegrees"],
                    end_pose["rollDegrees"],
                    amount,
                ),
            }
    return _camera_pose(frames[-1])


def _camera_basis(
    camera: dict, *, viewport_height: float | None = None
) -> tuple[Vec3, Vec3, Vec3, Vec3, float]:
    position = vec(camera["position"])
    forward = normalize(sub(vec(camera["lookAt"]), position))
    world_up: Vec3 = (0.0, 0.0, 1.0)
    if abs(dot(forward, world_up)) > 0.999:
        world_up = (0.0, 1.0, 0.0)
    right = normalize(cross(forward, world_up))
    up = normalize(cross(right, forward))
    roll = math.radians(float(camera.get("rollDegrees") or 0.0))
    if abs(roll) > 1e-8:
        cos_r, sin_r = math.cos(roll), math.sin(roll)
        right, up = (
            add(mul(right, cos_r), mul(up, sin_r)),
            add(mul(mul(right, -1.0), sin_r), mul(up, cos_r)),
        )
    height = PROXY_HEIGHT if viewport_height is None else float(viewport_height)
    focal = (height / 2.0) / math.tan(math.radians(float(camera["verticalFovDegrees"])) / 2.0)
    return position, right, up, forward, focal


def _view_point(point: Vec3, basis: tuple[Vec3, Vec3, Vec3, Vec3, float]) -> Vec3:
    position, right, up, forward, _focal = basis
    relative = sub(point, position)
    return (dot(relative, right), dot(relative, up), dot(relative, forward))


def _screen_point(x: float, y: float, depth: float, focal: float) -> tuple[float, float, float]:
    return (
        PROXY_WIDTH / 2.0 + focal * x / depth,
        PROXY_HEIGHT / 2.0 - focal * y / depth,
        depth,
    )


def project(
    point: Vec3,
    camera: dict,
    *,
    width: float | None = None,
    height: float | None = None,
) -> tuple[float, float, float] | None:
    view_w = PROXY_WIDTH if width is None else float(width)
    view_h = PROXY_HEIGHT if height is None else float(height)
    basis = _camera_basis(camera, viewport_height=view_h)
    x, y, depth = _view_point(point, basis)
    if depth <= NEAR_CLIP:
        return None
    focal = basis[4]
    return (
        view_w / 2.0 + focal * x / depth,
        view_h / 2.0 - focal * y / depth,
        depth,
    )


def unproject(
    pixel_x: float,
    pixel_y: float,
    depth: float,
    camera: dict,
    *,
    width: float,
    height: float,
) -> Vec3:
    """Inverse of ``project`` for one finite depth sample. Schema space, meters."""
    basis = _camera_basis(camera, viewport_height=height)
    position, right, up, forward, focal = basis
    view_x = (float(pixel_x) - float(width) / 2.0) * float(depth) / focal
    view_y = (float(height) / 2.0 - float(pixel_y)) * float(depth) / focal
    return add(position, add(mul(right, view_x), add(mul(up, view_y), mul(forward, float(depth)))))


def _clip_near(points: list[Vec3], near: float) -> list[Vec3]:
    """Keep the part of a polygon in front of the near plane."""
    clipped: list[Vec3] = []
    for index, current in enumerate(points):
        previous = points[index - 1]
        previous_in = previous[2] >= near
        current_in = current[2] >= near
        if current_in != previous_in:
            span = current[2] - previous[2]
            amount = 0.0 if abs(span) < 1e-8 else (near - previous[2]) / span
            clipped.append(
                (
                    previous[0] + (current[0] - previous[0]) * amount,
                    previous[1] + (current[1] - previous[1]) * amount,
                    near,
                )
            )
        if current_in:
            clipped.append(current)
    return clipped


def _projected_triangles(
    triangle: tuple[Vec3, Vec3, Vec3],
    basis: tuple[Vec3, Vec3, Vec3, Vec3, float],
) -> list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]]:
    view = [_view_point(point, basis) for point in triangle]
    polygon = _clip_near(view, NEAR_CLIP)
    if len(polygon) < 3:
        return []
    focal = basis[4]
    screen = [_screen_point(x, y, depth, focal) for x, y, depth in polygon]
    return [(screen[0], screen[index], screen[index + 1]) for index in range(1, len(screen) - 1)]


def _line3d(
    draw: ImageDraw.ImageDraw,
    camera: dict,
    start: Vec3,
    finish: Vec3,
    fill: tuple[int, int, int],
    width: int = 2,
) -> None:
    projected_start = project(start, camera)
    projected_finish = project(finish, camera)
    if projected_start and projected_finish:
        draw.line(
            (
                projected_start[0],
                projected_start[1],
                projected_finish[0],
                projected_finish[1],
            ),
            fill=fill,
            width=width,
        )


def _box_edges(position: Vec3, size: Vec3) -> list[tuple[Vec3, Vec3]]:
    half_x, half_y = size[0] / 2.0, size[1] / 2.0
    x0, x1 = position[0] - half_x, position[0] + half_x
    y0, y1 = position[1] - half_y, position[1] + half_y
    z0, z1 = position[2], position[2] + size[2]
    corners = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    indexes = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    return [(corners[a], corners[b]) for a, b in indexes]


def _character_color(character_id: str) -> tuple[int, int, int]:
    digest = hashlib.blake2s(character_id.encode("utf-8"), digest_size=3).digest()
    return tuple(90 + byte % 150 for byte in digest)  # type: ignore[return-value]


def _stance_heights(stance: str, scale: float = 1.0) -> tuple[float, float, float]:
    if stance == "sitting":
        base = (0.75, 1.12, 1.35)
    elif stance == "kneeling":
        base = (0.55, 0.92, 1.18)
    else:
        base = (0.95, 1.42, 1.72)
    return tuple(value * scale for value in base)  # type: ignore[return-value]


def _height_scale(show: dict, character_id: str | None) -> float:
    """Standing head height is 1.72m when a character has no proxy."""
    if not character_id:
        return 1.0
    proxy = ((show.get("characters") or {}).get(character_id) or {}).get("proxy") or {}
    try:
        height = float(proxy.get("heightMeters"))
    except (TypeError, ValueError):
        return 1.0
    if height <= 0:
        return 1.0
    return height / 1.72


def _build_factor(show: dict, character_id: str | None) -> float:
    if not character_id:
        return 1.0
    proxy = ((show.get("characters") or {}).get(character_id) or {}).get("proxy") or {}
    return {"slim": 0.85, "average": 1.0, "broad": 1.15}.get(str(proxy.get("build") or "average"), 1.0)


OPENPOSE_NAMES = (
    "nose",
    "neck",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
)
OPENPOSE_LIMBS = (
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
)
OPENPOSE_COLORS = (
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255),
    (170, 0, 255),
    (255, 0, 255),
    (255, 0, 170),
    (255, 0, 85),
)
BODY_LIMBS = (
    ("neck", "right_shoulder"),
    ("neck", "left_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("neck", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("neck", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("neck", "nose"),
)
_VOLUME_LIMBS = (
    ("neck", "right_hip", 0.12),
    ("neck", "left_hip", 0.12),
    ("right_shoulder", "left_shoulder", 0.06),
    ("right_shoulder", "right_elbow", 0.05),
    ("right_elbow", "right_wrist", 0.04),
    ("left_shoulder", "left_elbow", 0.05),
    ("left_elbow", "left_wrist", 0.04),
    ("right_hip", "right_knee", 0.07),
    ("right_knee", "right_ankle", 0.055),
    ("left_hip", "left_knee", 0.07),
    ("left_knee", "left_ankle", 0.055),
)


def _yaw_axes(yaw_degrees: float) -> tuple[Vec3, Vec3]:
    yaw = math.radians(float(yaw_degrees))
    return (math.sin(yaw), math.cos(yaw), 0.0), (math.cos(yaw), -math.sin(yaw), 0.0)


def _anchor_point(
    show: dict,
    episode: dict,
    scene: dict,
    target_id: str | None,
    time_seconds: float,
) -> Vec3 | None:
    if not target_id:
        return None
    tracks = (episode.get("spatialTimeline") or {}).get("characterTracks") or {}
    if target_id in tracks:
        state = episode_character_state(episode, target_id, time_seconds)
        _, _, head_height = _stance_heights(state["stance"], _height_scale(show, target_id))
        feet = vec(state["position"])
        return add(feet, (0.0, 0.0, head_height))
    prop_frames = ((episode.get("spatialTimeline") or {}).get("propTracks") or {}).get(target_id)
    if prop_frames:
        state = timeline_state(prop_frames, time_seconds)
        if state.get("position") is not None:
            return vec(state["position"])
        holder_id = state.get("heldByCharacterId")
        if holder_id:
            holder = episode_character_state(episode, holder_id, time_seconds)
            feet = vec(holder["position"])
            return add(feet, (0.0, 0.0, 1.1 * _height_scale(show, holder_id)))
    landmark = (
        show["locations"][scene["locationId"]].get("spatial", {}).get("landmarks", {}).get(target_id)
    )
    if isinstance(landmark, dict) and landmark.get("position") is not None:
        return vec(landmark["position"])
    return None


def character_pose_joints(
    show: dict,
    episode: dict,
    scene: dict,
    state: dict,
    time_seconds: float,
    character_id: str | None = None,
) -> dict[str, Vec3]:
    """OpenPose-18 joints for one blocking state. Right/left are the character's."""
    scale = _height_scale(show, character_id)
    feet = vec(state["position"])
    forward, right = _yaw_axes(state["bodyYawDegrees"])
    hip_z, shoulder_z, head_z = _stance_heights(state["stance"], scale)
    neck = add(feet, (0.0, 0.0, shoulder_z + (head_z - shoulder_z) * 0.45))
    hip = add(feet, (0.0, 0.0, hip_z))
    shoulder = add(feet, (0.0, 0.0, shoulder_z))
    gaze = forward
    look_target = _anchor_point(show, episode, scene, state.get("lookAtId"), time_seconds)
    if look_target is not None:
        aim = sub(look_target, neck)
        flat = (aim[0], aim[1], 0.0)
        if length(flat) > 1e-4:
            gaze = normalize(flat)
    nose = add(add(feet, (0.0, 0.0, head_z - 0.04 * scale)), mul(gaze, 0.06 * scale))
    right_shoulder = add(shoulder, mul(right, 0.20 * scale))
    left_shoulder = add(shoulder, mul(right, -0.20 * scale))
    right_hip = add(hip, mul(right, 0.11 * scale))
    left_hip = add(hip, mul(right, -0.11 * scale))

    def arm(shoulder_point: Vec3, side_sign: float, target_id: str | None) -> tuple[Vec3, Vec3]:
        target = _anchor_point(show, episode, scene, target_id, time_seconds)
        if target is not None:
            wrist = target
        else:
            wrist = add(shoulder_point, (0.0, 0.0, -(shoulder_z - hip_z) * 0.92))
            wrist = add(wrist, mul(forward, 0.06 * scale))
            wrist = add(wrist, mul(right, side_sign * 0.04 * scale))
        elbow = add(lerp(shoulder_point, wrist, 0.48), mul(right, side_sign * 0.05 * scale))
        return elbow, wrist

    right_elbow, right_wrist = arm(right_shoulder, 1.0, state.get("rightHandTargetId"))
    left_elbow, left_wrist = arm(left_shoulder, -1.0, state.get("leftHandTargetId"))

    def leg(hip_point: Vec3, side_sign: float) -> tuple[Vec3, Vec3]:
        ankle = add(add(feet, mul(right, side_sign * 0.09 * scale)), (0.0, 0.0, 0.06 * scale))
        knee_forward = 0.02 * scale
        stance = state["stance"]
        if stance == "sitting":
            knee_forward = 0.30 * scale
        elif stance == "kneeling":
            knee_forward = 0.12 * scale
            ankle = add(ankle, mul(forward, -0.18 * scale))
        elif stance == "walking":
            ankle = add(ankle, mul(forward, (0.22 if side_sign < 0 else -0.16) * scale))
        knee = add(lerp(hip_point, ankle, 0.52), mul(forward, knee_forward))
        return knee, ankle

    right_knee, right_ankle = leg(right_hip, 1.0)
    left_knee, left_ankle = leg(left_hip, -1.0)
    return {
        "nose": nose,
        "neck": neck,
        "right_shoulder": right_shoulder,
        "right_elbow": right_elbow,
        "right_wrist": right_wrist,
        "left_shoulder": left_shoulder,
        "left_elbow": left_elbow,
        "left_wrist": left_wrist,
        "right_hip": right_hip,
        "right_knee": right_knee,
        "right_ankle": right_ankle,
        "left_hip": left_hip,
        "left_knee": left_knee,
        "left_ankle": left_ankle,
        "right_eye": add(add(nose, mul(right, 0.032 * scale)), (0.0, 0.0, 0.04 * scale)),
        "left_eye": add(add(nose, mul(right, -0.032 * scale)), (0.0, 0.0, 0.04 * scale)),
        "right_ear": add(add(nose, mul(right, 0.08 * scale)), add(mul(gaze, -0.04 * scale), (0.0, 0.0, -0.02 * scale))),
        "left_ear": add(add(nose, mul(right, -0.08 * scale)), add(mul(gaze, -0.04 * scale), (0.0, 0.0, -0.02 * scale))),
    }


def _draw_character(
    draw: ImageDraw.ImageDraw,
    camera: dict,
    character_id: str,
    joints: dict[str, Vec3],
    debug: bool,
) -> None:
    color = _character_color(character_id)
    for start_name, end_name in BODY_LIMBS:
        _line3d(draw, camera, joints[start_name], joints[end_name], color, 8)
    nose = project(joints["nose"], camera)
    neck = project(joints["neck"], camera)
    if nose and neck:
        radius = max(7, int(math.hypot(nose[0] - neck[0], nose[1] - neck[1]) * 0.9))
        draw.ellipse(
            (nose[0] - radius, nose[1] - radius, nose[0] + radius, nose[1] + radius),
            fill=color,
            outline=(255, 255, 255),
            width=3,
        )
        if debug:
            facing = sub(joints["nose"], joints["neck"])
            _line3d(draw, camera, joints["nose"], add(joints["nose"], mul(facing, 2.0)), (255, 238, 80), 5)
            draw.text((nose[0] + radius + 5, nose[1] - radius), character_id, fill=(255, 255, 255))


def _quad(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> list[tuple[Vec3, Vec3, Vec3]]:
    return [(a, b, c), (a, c, d)]


def _capsule_triangles(start: Vec3, finish: Vec3, radius: float, sides: int = 8) -> list[tuple[Vec3, Vec3, Vec3]]:
    delta = sub(finish, start)
    if length(delta) < 1e-4:
        finish = add(start, (0.0, 0.0, max(radius, 0.05)))
        delta = sub(finish, start)
    axis = normalize(delta)
    helper = (0.0, 0.0, 1.0) if abs(axis[2]) < 0.9 else (1.0, 0.0, 0.0)
    across = normalize(cross(axis, helper))
    around = cross(axis, across)
    rings: list[list[Vec3]] = []
    for center in (start, finish):
        ring: list[Vec3] = []
        for index in range(sides):
            angle = 2.0 * math.pi * index / sides
            offset = add(mul(across, math.cos(angle) * radius), mul(around, math.sin(angle) * radius))
            ring.append(add(center, offset))
        rings.append(ring)
    triangles: list[tuple[Vec3, Vec3, Vec3]] = []
    for index in range(sides):
        nxt = (index + 1) % sides
        triangles.append((rings[0][index], rings[0][nxt], rings[1][nxt]))
        triangles.append((rings[0][index], rings[1][nxt], rings[1][index]))
    return triangles


def _character_triangles(
    joints: dict[str, Vec3], thickness: float = 1.0
) -> list[tuple[Vec3, Vec3, Vec3]]:
    top = add(joints["nose"], (0.0, 0.0, 0.10 * thickness))
    triangles = _capsule_triangles(joints["neck"], top, 0.11 * thickness, sides=10)
    for start_name, end_name, radius in _VOLUME_LIMBS:
        triangles.extend(
            _capsule_triangles(joints[start_name], joints[end_name], radius * thickness)
        )
    return triangles


def _triangle_normal(triangle: tuple[Vec3, Vec3, Vec3]) -> Vec3:
    return cross(sub(triangle[1], triangle[0]), sub(triangle[2], triangle[0]))


def _shade_gray(normal: Vec3, base: int) -> tuple[int, int, int]:
    """Reference clay gray. The GPU shader in clay_gpu.py uses this same formula."""
    try:
        normal = normalize(normal)
    except ValueError:
        value = base
    else:
        key = normalize((0.28, 0.42, 0.86))
        facing = abs(dot(normal, key))
        sky = max(0.0, normal[2])
        value = int(round(base * (0.62 + 0.38 * facing) + 18.0 * sky))
    value = max(88, min(232, value))
    return (value, value, value)


def _floor_triangles(spatial: dict) -> list[tuple[Vec3, Vec3, Vec3]]:
    width, depth, _height = vec(spatial["sizeMeters"])
    x0, x1 = -width / 2.0, width / 2.0
    y0, y1 = -depth / 2.0, depth / 2.0
    return _quad((x0, y0, 0.0), (x1, y0, 0.0), (x1, y1, 0.0), (x0, y1, 0.0))


def _camera_inside_stage(camera: dict, spatial: dict) -> bool:
    width, depth, height = vec(spatial["sizeMeters"])
    x, y, z = vec(camera["position"])
    return abs(x) <= width / 2.0 + 0.05 and abs(y) <= depth / 2.0 + 0.05 and -0.05 <= z <= height + 0.05


def _raster_clay(batches: list, camera: dict) -> tuple[Image.Image, np.ndarray]:
    """One GPU draw of the clay batches. The depth buffer is camera-forward meters."""
    from clay_gpu import raster_clay

    return raster_clay(
        batches,
        _camera_basis(camera),
        width=PROXY_WIDTH,
        height=PROXY_HEIGHT,
        near=NEAR_CLIP,
        background=VIEWPORT_GRAY,
    )


def render_mesh_thumbnail(vertices: np.ndarray, faces: np.ndarray, destination: Path, width: int = 256) -> None:
    """Clay thumbnail of one mesh, using the previs rasterizer."""
    from clay_gpu import ClayBatch

    if len(faces) == 0:
        raise ValueError("No triangles to thumbnail")
    points = np.asarray(vertices, dtype=np.float64)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2.0
    span = max(float(np.max(maximum - minimum)), 0.1)
    camera = {
        "position": [center[0] + span, center[1] - span * 2.2, center[2] + span * 0.85],
        "lookAt": [float(center[0]), float(center[1]), float(center[2])],
        "verticalFovDegrees": 35.0,
        "rollDegrees": 0.0,
    }
    image, _zbuf = _raster_clay([ClayBatch(points, faces, 176)], camera)
    thumb_height = max(1, round(width * PROXY_HEIGHT / PROXY_WIDTH))
    image.resize((width, thumb_height), Image.Resampling.LANCZOS).save(destination)


def _depth_image(zbuf: np.ndarray) -> Image.Image:
    valid = np.isfinite(zbuf)
    gray = np.zeros(zbuf.shape, dtype=np.uint8)
    if bool(valid.any()):
        near = float(np.percentile(zbuf[valid], 1))
        far = float(np.percentile(zbuf[valid], 99))
        span = max(far - near, 1e-3)
        normalized = np.clip((far - zbuf) / span, 0.0, 1.0)
        # Far surfaces stay gray, not black, so the control image is a depth
        # drawing rather than an underexposed frame.
        gray[valid] = np.rint(48.0 + normalized[valid] * 207.0).astype(np.uint8)
    return Image.fromarray(np.stack((gray, gray, gray), axis=-1), "RGB")


def _draw_openpose(draw: ImageDraw.ImageDraw, camera: dict, people: list[dict[str, Vec3]]) -> None:
    ordered: list[tuple[float, dict[str, Vec3]]] = []
    for joints in people:
        neck = project(joints["neck"], camera)
        ordered.append((neck[2] if neck else 1e9, joints))
    for _, joints in sorted(ordered, reverse=True):
        projected: dict[str, tuple[float, float, float]] = {}
        for name in OPENPOSE_NAMES:
            screen = project(joints[name], camera)
            if screen is not None:
                projected[name] = screen
        if "neck" not in projected:
            continue
        neck = projected["neck"]
        hip = projected.get("right_hip") or projected.get("left_hip")
        torso = 40.0
        if hip is not None:
            torso = math.hypot(neck[0] - hip[0], neck[1] - hip[1])
        width = max(4, min(36, int(torso * 0.18)))
        radius = max(3, width // 2)
        for index, (start_index, end_index) in enumerate(OPENPOSE_LIMBS):
            start_name = OPENPOSE_NAMES[start_index]
            end_name = OPENPOSE_NAMES[end_index]
            if start_name not in projected or end_name not in projected:
                continue
            start = projected[start_name]
            finish = projected[end_name]
            draw.line(
                (start[0], start[1], finish[0], finish[1]),
                fill=OPENPOSE_COLORS[index % len(OPENPOSE_COLORS)],
                width=width,
            )
        for index, name in enumerate(OPENPOSE_NAMES):
            if name not in projected:
                continue
            point = projected[name]
            draw.ellipse(
                (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
                fill=OPENPOSE_COLORS[index],
            )


_MESH_CACHE: dict[str, tuple[int, np.ndarray, np.ndarray]] = {}


def _cached_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, tuple]:
    """Schema-space vertices and faces. The GPU buffer key is the path and mtime."""
    from mesh_io import read_schema_mesh

    stamp = path.stat().st_mtime_ns
    key = (str(path), stamp)
    cached = _MESH_CACHE.get(str(path))
    if cached is not None and cached[0] == stamp:
        return cached[1], cached[2], key
    vertices, faces = read_schema_mesh(path)
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.uint32)
    _MESH_CACHE[str(path)] = (stamp, vertices, faces)
    return vertices, faces, key


def _location_set_mesh(show_id: str | None, location_id: str):
    """Schema-space location mesh. A missing file draws no location."""
    if not show_id:
        return None
    from pipeline_paths import location_dir

    path = location_dir(show_id, location_id) / "model.glb"
    if not path.is_file():
        return None
    return _cached_mesh(path)


def _landmark_batches(show: dict, location_id: str):
    """Landmark meshes placed at their script positions. Missing files are skipped."""
    from clay_gpu import ClayBatch
    from pipeline_paths import location_dir

    spatial = show["locations"][location_id].get("spatial") or {}
    root = location_dir(str(show.get("id") or ""), location_id)
    batches = []
    for landmark_id, landmark in (spatial.get("landmarks") or {}).items():
        if not isinstance(landmark, dict) or landmark.get("position") is None:
            continue
        glb = root / str(landmark_id) / "model.glb"
        if not glb.is_file():
            continue
        vertices, faces, key = _cached_mesh(glb)
        batches.append(
            ClayBatch(vertices, faces, 176, offset=vec(landmark["position"]), key=key)
        )
    return batches


def _batch_from_triangles(triangles: list[tuple[Vec3, Vec3, Vec3]], base: int):
    from clay_gpu import ClayBatch

    if not triangles:
        return None
    points = np.asarray(triangles, dtype=np.float32).reshape(-1, 3)
    faces = np.arange(points.shape[0], dtype=np.uint32).reshape(-1, 3)
    return ClayBatch(points, faces, base)


def _prop_frame(track: list[dict], time_seconds: float) -> dict:
    frames = sorted(track, key=lambda item: float(item["timeSeconds"]))
    if all(frame.get("position") is not None for frame in frames):
        return timeline_state(frames, time_seconds)
    chosen = frames[0]
    for frame in frames:
        if float(frame["timeSeconds"]) <= time_seconds:
            chosen = frame
    return dict(chosen)


def _prop_surfaces(
    show: dict,
    episode: dict,
    scene: dict,
    time_seconds: float,
):
    from clay_gpu import ClayBatch
    from pipeline_paths import stage_dir

    tracks = ((episode.get("spatialTimeline") or {}).get("propTracks")) or {}
    batches = []
    for prop_id, track in tracks.items():
        glb = stage_dir("assets", str(show.get("id") or "")) / "props" / str(prop_id) / "model.glb"
        if not glb.is_file() or not track:
            continue
        state = _prop_frame(track, time_seconds)
        if state.get("locationId") and state["locationId"] != scene["locationId"]:
            continue
        position = state.get("position")
        holder_id = state.get("heldByCharacterId")
        if position is None and holder_id:
            holder = episode_character_state(episode, holder_id, time_seconds)
            if holder.get("locationId") and holder["locationId"] != scene["locationId"]:
                continue
            joints = character_pose_joints(
                show, episode, scene, holder, time_seconds, holder_id
            )
            hand = "left_wrist" if state.get("heldInHand") == "left" else "right_wrist"
            position = joints[hand]
        if position is None:
            continue
        vertices, faces, key = _cached_mesh(glb)
        batches.append(ClayBatch(vertices, faces, 190, offset=vec(position), key=key))
    return batches


def _scene_surfaces(
    show: dict,
    episode: dict,
    scene: dict,
    time_seconds: float,
):
    """Clay batches for one instant. The stage box is bounds, not a room mesh.

    A location with no people draws its one generated mesh. A location with
    people draws the open floor and each landmark mesh at the script position.
    Cameras and people stay on the marks in the script. Open sky stays the
    viewport gray.
    """
    from asset_resolver import location_has_people
    from clay_gpu import ClayBatch

    camera = camera_at(scene, time_seconds)
    spatial = show["locations"][scene["locationId"]]["spatial"]
    batches: list[ClayBatch] = []
    if location_has_people(show, scene["locationId"]):
        if _camera_inside_stage(camera, spatial):
            floor = _batch_from_triangles(_floor_triangles(spatial), 156)
            if floor is not None:
                batches.append(floor)
        batches.extend(_landmark_batches(show, scene["locationId"]))
    else:
        set_mesh = _location_set_mesh(show.get("id"), scene["locationId"])
        if set_mesh is None and _camera_inside_stage(camera, spatial):
            floor = _batch_from_triangles(_floor_triangles(spatial), 156)
            if floor is not None:
                batches.append(floor)
        if set_mesh is not None:
            vertices, faces, key = set_mesh
            batches.append(ClayBatch(vertices, faces, 176, key=key))
    batches.extend(_prop_surfaces(show, episode, scene, time_seconds))
    people: list[tuple[str, dict[str, Vec3]]] = []
    character_triangles: list[tuple[Vec3, Vec3, Vec3]] = []
    for character_id in scene["characterIds"]:
        state = episode_character_state(episode, character_id, time_seconds)
        joints = character_pose_joints(
            show, episode, scene, state, time_seconds, character_id
        )
        people.append((character_id, joints))
        thickness = _height_scale(show, character_id) * _build_factor(show, character_id)
        character_triangles.extend(_character_triangles(joints, thickness))
    characters = _batch_from_triangles(character_triangles, 214)
    if characters is not None:
        batches.append(characters)
    return camera, batches, people


def _face_mask(
    camera: dict,
    people: list[tuple[str, dict[str, Vec3]]],
    identity_ids: list[str],
    zbuf: np.ndarray,
) -> Image.Image:
    mask = Image.new("L", (PROXY_WIDTH, PROXY_HEIGHT), 0)
    draw = ImageDraw.Draw(mask)
    allowed = set(identity_ids)
    for character_id, joints in people:
        if character_id not in allowed:
            continue
        nose = project(joints["nose"], camera)
        neck = project(joints["neck"], camera)
        if nose is None:
            continue
        x = int(round(nose[0]))
        y = int(round(nose[1]))
        if not (0 <= x < PROXY_WIDTH and 0 <= y < PROXY_HEIGHT):
            continue
        if np.isfinite(zbuf[y, x]) and zbuf[y, x] + 0.2 < nose[2]:
            continue
        if neck is None:
            radius = 18
        else:
            radius = max(14, int(math.hypot(nose[0] - neck[0], nose[1] - neck[1]) * 0.85))
        draw.ellipse(
            (x - radius, y - int(radius * 1.25), x + radius, y + int(radius * 0.85)),
            fill=255,
        )
    blurred = mask.filter(ImageFilter.GaussianBlur(radius=8))
    return Image.merge("RGB", (blurred, blurred, blurred))


def render_blocked_frame(
    show: dict,
    episode: dict,
    scene: dict,
    time_seconds: float,
) -> tuple[Image.Image, np.ndarray, list[tuple[str, dict[str, Vec3]]]]:
    """One clay viewport frame. No diffusion model is involved."""
    camera, batches, people = _scene_surfaces(show, episode, scene, time_seconds)
    image, zbuf = _raster_clay(batches, camera)
    return image, zbuf, people


def render_structure_maps(
    show: dict,
    episode: dict,
    scene: dict,
    time_seconds: float,
    depth_destination: Path,
    pose_destination: Path,
    zbuf: np.ndarray | None = None,
) -> None:
    """Depth and pose previews of the same clay surfaces. Not sent to a model.

    Pass the clay frame's depth buffer so the set is not drawn a second time.
    """
    camera, batches, people = _scene_surfaces(show, episode, scene, time_seconds)
    depth_destination.parent.mkdir(parents=True, exist_ok=True)
    if zbuf is None:
        _image, zbuf = _raster_clay(batches, camera)
    _depth_image(zbuf).save(depth_destination)
    pose = Image.new("RGB", (PROXY_WIDTH, PROXY_HEIGHT), (0, 0, 0))
    _draw_openpose(ImageDraw.Draw(pose), camera, [joints for _character_id, joints in people])
    pose.save(pose_destination)


def render_scene_proxy(
    show: dict,
    episode: dict,
    scene: dict,
    time_seconds: float,
    destination: Path,
    debug: bool = True,
) -> None:
    camera = camera_at(scene, time_seconds)
    location = show["locations"][scene["locationId"]]
    spatial = location["spatial"]
    image = Image.new("RGB", (PROXY_WIDTH, PROXY_HEIGHT), (13, 17, 25))
    draw = ImageDraw.Draw(image)

    width, depth, _ = vec(spatial["sizeMeters"])
    if debug:
        for x in range(math.floor(-width / 2), math.ceil(width / 2) + 1):
            _line3d(draw, camera, (float(x), -depth / 2, 0.0), (float(x), depth / 2, 0.0), (39, 51, 66))
        for y in range(math.floor(-depth / 2), math.ceil(depth / 2) + 1):
            _line3d(draw, camera, (-width / 2, float(y), 0.0), (width / 2, float(y), 0.0), (39, 51, 66))

    for landmark_id, landmark in spatial.get("landmarks", {}).items():
        if not isinstance(landmark, dict) or landmark.get("position") is None:
            continue
        point = vec(landmark["position"])
        _line3d(draw, camera, point, add(point, (0.0, 0.0, 0.6)), (70, 115, 145), 3)
        label_point = project(add(point, (0.0, 0.0, 0.6)), camera)
        if debug and label_point:
            draw.text(
                (label_point[0] + 4, label_point[1]),
                landmark_id,
                fill=(110, 180, 210),
            )

    states: list[tuple[float, str, dict]] = []
    for character_id in scene["characterIds"]:
        state = dict(episode_character_state(episode, character_id, time_seconds))
        if state["locationId"] != scene["locationId"]:
            raise ValueError(
                f"Scene {scene['sceneNumber']} shows {character_id} at "
                f"{scene['locationId']}, but timeline places them at {state['locationId']}"
            )
        depth_from_camera = length(sub(vec(state["position"]), vec(camera["position"])))
        states.append((depth_from_camera, character_id, state))
    for _, character_id, state in sorted(states, reverse=True):
        joints = character_pose_joints(
            show, episode, scene, state, time_seconds, character_id
        )
        _draw_character(draw, camera, character_id, joints, debug)

    title = (
        f"scene {scene['sceneNumber']:02d}  t={time_seconds:.2f}s  "
        f"{scene['locationId']}  fov={float(camera['verticalFovDegrees']):.1f}"
    )
    if debug:
        draw.rectangle((0, 0, PROXY_WIDTH, 54), fill=(0, 0, 0))
        draw.text((18, 17), title, fill=(255, 255, 255), font=ImageFont.load_default())
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def validate_spatial_episode(show: dict, episode: dict) -> list[str]:
    errors: list[str] = []
    timeline = episode.get("spatialTimeline")
    if not timeline:
        return [
            f"episode {episode.get('episodeNumber', '?')}: spatialTimeline is required"
        ]
    for character_id, track in timeline.get("characterTracks", {}).items():
        previous: dict | None = None
        for frame in sorted(track, key=lambda item: float(item["timeSeconds"])):
            location_id = frame["locationId"]
            location = show["locations"].get(location_id)
            if not location or not location.get("spatial"):
                errors.append(f"{character_id}: location {location_id!r} has no spatial stage")
                continue
            width, depth, _ = vec(location["spatial"]["sizeMeters"])
            position = vec(frame["position"])
            if abs(position[0]) > width / 2 or abs(position[1]) > depth / 2 or position[2] < 0:
                errors.append(f"{character_id}: position {position} is outside {location_id}")
            if previous and previous["locationId"] == location_id:
                elapsed = float(frame["timeSeconds"]) - float(previous["timeSeconds"])
                if elapsed <= 0:
                    errors.append(f"{character_id}: keyframe times must increase")
            previous = frame

    for scene in episode["scenes"]:
        if not scene.get("camera") or not scene.get("timeRangeSeconds"):
            errors.append(
                f"scene {scene['sceneNumber']}: every scene in a spatial episode requires "
                "timeRangeSeconds and camera"
            )
            continue
        start, finish = (float(value) for value in scene["timeRangeSeconds"])
        if finish <= start:
            errors.append(f"scene {scene['sceneNumber']}: invalid time range")
            continue
        location = show["locations"].get(scene["locationId"]) or {}
        if not location.get("spatial"):
            errors.append(
                f"scene {scene['sceneNumber']}: location {scene['locationId']!r} "
                "has no spatial stage"
            )
            continue
        try:
            frames = camera_keyframes(scene)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        previous_time: float | None = None
        pose_ok = True
        for frame in frames:
            frame_time = float(frame["timeSeconds"])
            if previous_time is not None and frame_time <= previous_time:
                errors.append(f"scene {scene['sceneNumber']}: camera keyframe times must increase")
                pose_ok = False
                break
            previous_time = frame_time
            try:
                _camera_basis(_camera_pose(frame))
            except ValueError as exc:
                errors.append(f"scene {scene['sceneNumber']}: {exc}")
                pose_ok = False
                break
        if not pose_ok:
            continue
        camera = camera_at(scene, start)
        for character_id in scene["characterIds"]:
            try:
                state = episode_character_state(episode, character_id, start)
            except ValueError as exc:
                errors.append(f"scene {scene['sceneNumber']}: {exc}")
                continue
            if state["locationId"] != scene["locationId"]:
                errors.append(
                    f"scene {scene['sceneNumber']}: {character_id} is in "
                    f"{state['locationId']}, not {scene['locationId']}"
                )
                continue
            if length(sub(vec(state["position"]), vec(camera["position"]))) < 0.3:
                errors.append(f"scene {scene['sceneNumber']}: camera intersects {character_id}")
    return errors


def compile_spatial_video_prompt(scene: dict) -> str:
    parts: list[str] = []
    if scene.get("speakerId"):
        parts.append(
            f"LIP SYNC: {scene['speakerId']} visibly lip-syncs every spoken word; "
            "the lips open on the first syllable and articulate continuously through the line."
        )
    if scene.get("videoPrompt"):
        parts.append(scene["videoPrompt"])
    return " ".join(parts)


def scene_has_spatial_change(episode: dict, scene: dict) -> bool:
    if not scene.get("timeRangeSeconds") or not scene.get("camera"):
        return False
    start, finish = (float(value) for value in scene["timeRangeSeconds"])
    try:
        first_camera = camera_at(scene, start)
        last_camera = camera_at(scene, finish)
    except ValueError:
        return False
    if length(sub(vec(last_camera["position"]), vec(first_camera["position"]))) >= 0.02:
        return True
    if length(sub(vec(last_camera["lookAt"]), vec(first_camera["lookAt"]))) >= 0.02:
        return True
    if abs(last_camera["verticalFovDegrees"] - first_camera["verticalFovDegrees"]) >= 0.5:
        return True
    if (
        abs(
            (last_camera["rollDegrees"] - first_camera["rollDegrees"] + 180.0) % 360.0
            - 180.0
        )
        >= 0.5
    ):
        return True
    for character_id in scene["characterIds"]:
        first = episode_character_state(episode, character_id, start)
        last = episode_character_state(episode, character_id, finish)
        if length(sub(vec(last["position"]), vec(first["position"]))) >= 0.02:
            return True
        if abs(
            (float(last["bodyYawDegrees"]) - float(first["bodyYawDegrees"]) + 180.0)
            % 360.0
            - 180.0
        ) >= 2.0:
            return True
        if first["stance"] != last["stance"]:
            return True
    return False


def character_facing_direction(
    episode: dict,
    scene: dict,
    character_id: str,
    show: dict | None = None,
) -> str | None:
    """Return the frame edge the character faces from authoritative blocking."""
    if not scene.get("timeRangeSeconds") or not scene.get("camera"):
        return None
    start = float(scene["timeRangeSeconds"][0])
    camera = camera_at(scene, start)
    state = episode_character_state(episode, character_id, start)
    feet = vec(state["position"])
    _, _, head_height = _stance_heights(state["stance"], _height_scale(show or {}, character_id))
    head = add(feet, (0.0, 0.0, head_height))
    projected = project(head, camera)
    if projected is None:
        return None
    target_id = state.get("lookAtId")
    tracks = (episode.get("spatialTimeline") or {}).get("characterTracks") or {}
    if target_id in tracks:
        target = episode_character_state(episode, target_id, start)
        _, _, target_head_height = _stance_heights(
            target["stance"], _height_scale(show or {}, target_id)
        )
        target_feet = vec(target["position"])
        target_projected = project(
            add(target_feet, (0.0, 0.0, target_head_height)),
            camera,
        )
    else:
        yaw = math.radians(float(state["bodyYawDegrees"]))
        target_projected = project(
            add(head, (math.sin(yaw), math.cos(yaw), 0.0)),
            camera,
        )
    if target_projected is None:
        return None
    return "left" if target_projected[0] < projected[0] else "right"


def spatial_target_screen_position(
    show: dict,
    episode: dict,
    scene: dict,
    target_id: str,
) -> tuple[float, float]:
    start = float(scene["timeRangeSeconds"][0])
    timeline = episode.get("spatialTimeline") or {}
    prop_frames = (timeline.get("propTracks") or {}).get(target_id)
    if prop_frames:
        state = timeline_state(prop_frames, start)
        position = state.get("position")
        if position is None and state.get("heldByCharacterId"):
            holder = episode_character_state(
                episode, state["heldByCharacterId"], start
            )
            position = add(vec(holder["position"]), (0.0, 0.0, 1.1 * _height_scale(show, state["heldByCharacterId"])))
    else:
        landmark = (
            show["locations"][scene["locationId"]]
            .get("spatial", {})
            .get("landmarks", {})
            .get(target_id)
        )
        position = landmark.get("position") if isinstance(landmark, dict) else None
        if isinstance(landmark, dict) and position is None:
            raise ValueError(
                f"Landmark {target_id!r} in scene {scene['sceneNumber']} has no position"
            )
        if position is not None:
            position = vec(position)
    if position is None:
        raise ValueError(
            f"Unknown spatial focus target {target_id!r} in scene {scene['sceneNumber']}"
        )
    projected = project(vec(position), camera_at(scene, start))
    if not projected:
        raise ValueError(
            f"Spatial focus target {target_id!r} is behind scene "
            f"{scene['sceneNumber']} camera"
        )
    return projected[0], projected[1]


def blockout_sample_times(start: float, finish: float, fps: int = BLOCKOUT_FPS) -> list[float]:
    """Sample the shot so the clay playblast covers the whole camera move."""
    span = max(0.0, finish - start)
    count = max(2, int(round(span * fps)) + 1)
    return [start + span * index / (count - 1) for index in range(count)]


def write_scene_description(show: dict, episode: dict, scene: dict) -> Path:
    """Y-up scene description. Positions are glTF, with the schema point kept beside them."""
    from coords import schema_to_gltf
    from pipeline_paths import location_dir, scene_description_path

    start, finish = (float(value) for value in scene["timeRangeSeconds"])

    def gltf_point(point: Vec3) -> list[float]:
        return list(schema_to_gltf((float(point[0]), float(point[1]), float(point[2]))))

    camera_frames = []
    for frame in camera_keyframes(scene):
        pose = camera_at(
            {**scene, "camera": {"keyframes": [frame]}},
            float(frame["timeSeconds"]),
        )
        camera_frames.append(
            {
                "timeSeconds": float(frame["timeSeconds"]),
                "position": gltf_point(pose["position"]),
                "schemaPosition": pose["position"],
                "lookAt": gltf_point(pose["lookAt"]),
                "schemaLookAt": pose["lookAt"],
                "verticalFovDegrees": pose["verticalFovDegrees"],
                "rollDegrees": pose["rollDegrees"],
            }
        )
    characters = []
    tracks = ((episode.get("spatialTimeline") or {}).get("characterTracks")) or {}
    for character_id in scene.get("characterIds") or []:
        keyframes = []
        for frame in tracks.get(character_id) or []:
            time_seconds = float(frame["timeSeconds"])
            if time_seconds < start - 1e-6 or time_seconds > finish + 1e-6:
                continue
            position = [float(value) for value in frame["position"]]
            keyframes.append(
                {
                    "timeSeconds": time_seconds,
                    "position": gltf_point(position),  # type: ignore[arg-type]
                    "schemaPosition": position,
                    "bodyYawDegrees": float(frame.get("bodyYawDegrees") or 0.0),
                    "stance": frame.get("stance") or "standing",
                }
            )
        if not keyframes and character_id in tracks:
            state = episode_character_state(episode, character_id, start)
            position = [float(value) for value in state["position"]]
            keyframes.append(
                {
                    "timeSeconds": start,
                    "position": gltf_point(position),  # type: ignore[arg-type]
                    "schemaPosition": position,
                    "bodyYawDegrees": float(state.get("bodyYawDegrees") or 0.0),
                    "stance": state.get("stance") or "standing",
                }
            )
        proxy = ((show.get("characters") or {}).get(character_id) or {}).get("proxy")
        characters.append({"id": character_id, "proxy": proxy, "keyframes": keyframes})
    props = []
    prop_tracks = ((episode.get("spatialTimeline") or {}).get("propTracks")) or {}
    for prop_id, track in prop_tracks.items():
        keyframes = []
        for frame in track:
            time_seconds = float(frame["timeSeconds"])
            if time_seconds < start - 1e-6 or time_seconds > finish + 1e-6:
                continue
            if frame.get("locationId") and frame["locationId"] != scene["locationId"]:
                continue
            entry: dict = {
                "timeSeconds": time_seconds,
                "heldByCharacterId": frame.get("heldByCharacterId"),
            }
            if frame.get("position") is not None:
                position = [float(value) for value in frame["position"]]
                entry["position"] = gltf_point(position)  # type: ignore[arg-type]
                entry["schemaPosition"] = position
            keyframes.append(entry)
        if not keyframes:
            continue
        props.append({"id": prop_id, "keyframes": keyframes})
    from asset_resolver import location_has_people

    model = location_dir(str(show.get("id") or ""), scene["locationId"]) / "model.glb"
    occupied = location_has_people(show, scene["locationId"])
    spatial = show["locations"][scene["locationId"]].get("spatial") or {}
    landmarks = []
    for landmark_id, landmark in (spatial.get("landmarks") or {}).items():
        if not isinstance(landmark, dict) or landmark.get("position") is None:
            continue
        position = [float(value) for value in landmark["position"]]
        entry = {
            "id": landmark_id,
            "position": gltf_point(position),  # type: ignore[arg-type]
            "schemaPosition": position,
        }
        landmark_model = (
            location_dir(str(show.get("id") or ""), scene["locationId"])
            / str(landmark_id)
            / "model.glb"
        )
        if landmark_model.is_file():
            entry["model"] = (
                f"assets/{show.get('id')}/{scene['locationId']}/{landmark_id}/model.glb"
            )
        landmarks.append(entry)
    payload = {
        "space": "gltf-y-up",
        "showId": show.get("id"),
        "episodeNumber": episode.get("episodeNumber"),
        "sceneNumber": scene.get("sceneNumber"),
        "locationId": scene["locationId"],
        "timeRangeSeconds": [start, finish],
        "location": None
        if occupied or not model.is_file()
        else {
            "locationId": scene["locationId"],
            "model": f"assets/{show.get('id')}/{scene['locationId']}/model.glb",
        },
        "landmarks": landmarks,
        "camera": {"keyframes": camera_frames},
        "characters": characters,
        "props": props,
    }
    destination = scene_description_path(
        str(show["id"]), int(episode["episodeNumber"]), int(scene["sceneNumber"])
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def render_blocked_scene(
    show: dict,
    episode: dict,
    scene: dict,
) -> list[Path]:
    """Write the clay playblast and the start/end frames the realism pass edits."""
    from ffmpeg_tools import encode_rgb_frames
    from pipeline_paths import (
        blockout_video_path,
        clay_frame_path,
        guide_path,
    )

    start, finish = (float(value) for value in scene["timeRangeSeconds"])
    times = blockout_sample_times(start, finish)
    frames: list[Image.Image] = []
    written: list[Path] = [write_scene_description(show, episode, scene)]
    identity_ids = list(scene["characterIds"][:IDENTITY_FACE_LIMIT])
    guides = {times[0]: "start", times[-1]: "end"}
    for time_seconds in times:
        image, zbuf, people = render_blocked_frame(show, episode, scene, time_seconds)
        frames.append(image)
        label = guides.get(time_seconds)
        if label is None:
            continue
        blockout_path = clay_frame_path(
            show["id"],
            episode["episodeNumber"],
            scene["sceneNumber"],
            label,
        )
        mask_path = guide_path(
            show["id"],
            episode["episodeNumber"],
            scene["sceneNumber"],
            label,
            "faces",
        )
        depth_path = guide_path(
            show["id"],
            episode["episodeNumber"],
            scene["sceneNumber"],
            label,
            "depth",
        )
        pose_path = guide_path(
            show["id"],
            episode["episodeNumber"],
            scene["sceneNumber"],
            label,
            "pose",
        )
        blockout_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(blockout_path)
        camera = camera_at(scene, time_seconds)
        _face_mask(camera, people, identity_ids, zbuf).save(mask_path)
        render_structure_maps(
            show, episode, scene, time_seconds, depth_path, pose_path, zbuf
        )
        written.extend((blockout_path, mask_path, depth_path, pose_path))
    video_path = blockout_video_path(
        show["id"], episode["episodeNumber"], scene["sceneNumber"]
    )
    raw = b"".join(frame.convert("RGB").tobytes() for frame in frames)
    encode_rgb_frames(raw, PROXY_WIDTH, PROXY_HEIGHT, BLOCKOUT_FPS, video_path)
    written.append(video_path)
    return written


def scene_blockouts(show: dict, episode: dict) -> tuple[list[Path], list[int]]:
    """Scene playblasts in script order, plus scene numbers that are not on disk yet."""
    ready: list[Path] = []
    missing: list[int] = []
    for scene in episode["scenes"]:
        if not scene.get("camera") or not scene.get("timeRangeSeconds"):
            continue
        path = blockout_video_path(show["id"], episode["episodeNumber"], scene["sceneNumber"])
        if path.is_file() and path.stat().st_size >= 1024:
            ready.append(path)
        else:
            missing.append(int(scene["sceneNumber"]))
    return ready, missing


def write_episode_blockout(show: dict, episode: dict) -> Path | None:
    """Join scene blockouts into one episode playblast. Skip when a scene is missing."""
    ready, missing = scene_blockouts(show, episode)
    if missing or not ready:
        return None
    from ffmpeg_tools import concat_videos

    destination = episode_blockout_path(show["id"], episode["episodeNumber"])
    concat_videos(ready, destination)
    return destination


def generate_episode_previs(show: dict, episode: dict, scene_number: int | None = None) -> list[Path]:
    errors = validate_spatial_episode(show, episode)
    if errors:
        raise SystemExit("Spatial validation failed:\n- " + "\n- ".join(errors))
    generated: list[Path] = []
    for scene in episode["scenes"]:
        if scene_number is not None and scene["sceneNumber"] != scene_number:
            continue
        if not scene.get("camera") or not scene.get("timeRangeSeconds"):
            continue
        from asset_resolver import location_has_people

        if location_has_people(show, scene["locationId"]):
            where = "landmark meshes"
        elif _location_set_mesh(show.get("id"), scene["locationId"]) is not None:
            where = "location mesh"
        else:
            where = "empty stage"
        print(f"Blocking scene {scene['sceneNumber']:02d} on the {where}...", flush=True)
        generated.extend(render_blocked_scene(show, episode, scene))
    return generated


def write_contact_sheet(paths: list[Path], destination: Path) -> None:
    starts = [path for path in paths if path.name == "start.png"]
    if not starts:
        return
    thumb_width = 240
    thumb_height = round(thumb_width * PROXY_HEIGHT / PROXY_WIDTH)
    columns = 4
    rows = math.ceil(len(starts) / columns)
    sheet = Image.new("RGB", (columns * thumb_width, rows * thumb_height), (0, 0, 0))
    for index, path in enumerate(starts):
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_width, thumb_height))
        sheet.paste(image, ((index % columns) * thumb_width, (index // columns) * thumb_height))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render deterministic spatial previs guides.")
    parser.add_argument("--show", help="Generate only this show id")
    parser.add_argument("--episode", type=int, help="Generate only this episode")
    parser.add_argument("--scene", type=int, help="Generate only this scene")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Accepted for compatibility with the unified content:render command",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Accepted for compatibility with the unified content:render command",
    )
    args = parser.parse_args()
    if args.scene is not None and args.episode is None:
        parser.error("--scene requires --episode")

    scripts = discover_show_scripts(args.show)
    if not scripts:
        raise SystemExit("No matching show scripts")
    for script in scripts:
        show = load_show(script)
        for episode in show["episodes"]:
            if args.episode is not None and episode["episodeNumber"] != args.episode:
                continue
            paths = generate_episode_previs(show, episode, args.scene)
            contact_sheet = contact_sheet_path(show["id"], episode["episodeNumber"])
            write_contact_sheet(paths, contact_sheet)
            print(
                f"Wrote {len(paths)} blocked-scene files for {show['id']}/{episode['episodeNumber']} "
                f"to {contact_sheet.parent}",
                flush=True,
            )
            episode_blockout = write_episode_blockout(show, episode)
            if episode_blockout is not None:
                print(
                    f"Wrote {episode_blockout} ({episode_blockout.stat().st_size} bytes)",
                    flush=True,
                )
            else:
                _ready, missing = scene_blockouts(show, episode)
                if missing:
                    waiting = ", ".join(f"{number:02d}" for number in missing)
                    print(
                        f"Episode blockout waiting on scenes {waiting}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
