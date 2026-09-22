#!/usr/bin/env python3
"""Project deterministic stage blocking into vertical proxy frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts_input"
OUTPUT_DIR = ROOT / "output"
PROXY_WIDTH = 768
PROXY_HEIGHT = 1360

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


def _camera_basis(camera: dict) -> tuple[Vec3, Vec3, Vec3, Vec3, float]:
    position = vec(camera["position"])
    forward = normalize(sub(vec(camera["lookAt"]), position))
    right = normalize(cross(forward, (0.0, 0.0, 1.0)))
    up = normalize(cross(right, forward))
    focal = (PROXY_HEIGHT / 2.0) / math.tan(
        math.radians(float(camera["verticalFovDegrees"])) / 2.0
    )
    return position, right, up, forward, focal


def project(point: Vec3, camera: dict) -> tuple[float, float, float] | None:
    position, right, up, forward, focal = _camera_basis(camera)
    relative = sub(point, position)
    depth = dot(relative, forward)
    if depth <= 0.05:
        return None
    return (
        PROXY_WIDTH / 2.0 + focal * dot(relative, right) / depth,
        PROXY_HEIGHT / 2.0 - focal * dot(relative, up) / depth,
        depth,
    )


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


def _stance_heights(stance: str) -> tuple[float, float, float]:
    if stance == "sitting":
        return 0.75, 1.12, 1.35
    if stance == "kneeling":
        return 0.55, 0.92, 1.18
    return 0.95, 1.42, 1.72


def _draw_character(
    draw: ImageDraw.ImageDraw,
    camera: dict,
    character_id: str,
    state: dict,
    debug: bool,
) -> None:
    feet = vec(state["position"])
    hip_height, shoulder_height, head_height = _stance_heights(state["stance"])
    hip = add(feet, (0.0, 0.0, hip_height))
    shoulder = add(feet, (0.0, 0.0, shoulder_height))
    head = add(feet, (0.0, 0.0, head_height))
    color = _character_color(character_id)
    _line3d(draw, camera, feet, hip, color, 9)
    _line3d(draw, camera, hip, shoulder, color, 14)
    yaw = math.radians(float(state["bodyYawDegrees"]))
    side = (math.cos(yaw) * 0.24, -math.sin(yaw) * 0.24, 0.0)
    left_shoulder, right_shoulder = add(shoulder, side), sub(shoulder, side)
    left_hand = add(hip, side)
    right_hand = sub(hip, side)
    _line3d(draw, camera, left_shoulder, left_hand, color, 7)
    _line3d(draw, camera, right_shoulder, right_hand, color, 7)
    facing = (math.sin(yaw), math.cos(yaw), 0.0)
    if debug:
        _line3d(draw, camera, head, add(head, mul(facing, 0.55)), (255, 238, 80), 5)
    projected_head = project(head, camera)
    projected_neck = project(add(feet, (0.0, 0.0, 1.52)), camera)
    if projected_head and projected_neck:
        radius = max(7, int(abs(projected_head[1] - projected_neck[1]) * 1.5))
        draw.ellipse(
            (
                projected_head[0] - radius,
                projected_head[1] - radius,
                projected_head[0] + radius,
                projected_head[1] + radius,
            ),
            fill=color,
            outline=(255, 255, 255),
            width=3,
        )
        if debug:
            draw.text(
                (projected_head[0] + radius + 5, projected_head[1] - radius),
                character_id,
                fill=(255, 255, 255),
            )


def _interpolated_camera(scene: dict, amount: float) -> dict:
    camera = scene["camera"]
    result = dict(camera)
    if camera.get("endPosition"):
        result["position"] = list(
            lerp(vec(camera["position"]), vec(camera["endPosition"]), amount)
        )
    if camera.get("endLookAt"):
        result["lookAt"] = list(
            lerp(vec(camera["lookAt"]), vec(camera["endLookAt"]), amount)
        )
    return result


def render_scene_proxy(
    show: dict,
    episode: dict,
    scene: dict,
    time_seconds: float,
    destination: Path,
    debug: bool = True,
) -> None:
    start, finish = (float(value) for value in scene["timeRangeSeconds"])
    amount = min(1.0, max(0.0, (time_seconds - start) / max(finish - start, 1e-6)))
    camera = _interpolated_camera(scene, amount)
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
        for edge_start, edge_finish in _box_edges(
            vec(landmark["position"]), vec(landmark["size"])
        ):
            _line3d(draw, camera, edge_start, edge_finish, (70, 115, 145), 3)
        label_point = project(
            add(vec(landmark["position"]), (0.0, 0.0, float(landmark["size"][2]))),
            camera,
        )
        if debug and label_point:
            draw.text((label_point[0] + 4, label_point[1]), landmark_id, fill=(110, 180, 210))

    states: list[tuple[float, str, dict]] = []
    for character_id in scene["characterIds"]:
        state = episode_character_state(episode, character_id, time_seconds)
        if state["locationId"] != scene["locationId"]:
            raise ValueError(
                f"Scene {scene['sceneNumber']} shows {character_id} at "
                f"{scene['locationId']}, but timeline places them at {state['locationId']}"
            )
        depth_from_camera = length(sub(vec(state["position"]), vec(camera["position"])))
        states.append((depth_from_camera, character_id, state))
    for _, character_id, state in sorted(states, reverse=True):
        _draw_character(draw, camera, character_id, state, debug)

    title = (
        f"scene {scene['sceneNumber']:02d}  t={time_seconds:.2f}s  "
        f"{scene['locationId']}  fov={float(camera['verticalFovDegrees']):.1f}"
    )
    if debug:
        draw.rectangle((0, 0, PROXY_WIDTH, 54), fill=(0, 0, 0))
        draw.text((18, 17), title, fill=(255, 255, 255), font=ImageFont.load_default())
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def previs_dir(show_id: str, episode_number: int) -> Path:
    return OUTPUT_DIR / show_id / str(episode_number) / "previs"


def proxy_frame_path(
    show_id: str, episode_number: int, scene_number: int, label: str
) -> Path:
    return previs_dir(show_id, episode_number) / f"scene_{scene_number:02d}_{label}.png"


def validate_spatial_episode(show: dict, episode: dict) -> list[str]:
    errors: list[str] = []
    timeline = episode.get("spatialTimeline")
    if not timeline:
        return errors
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
                elif length(sub(position, vec(previous["position"]))) / elapsed > 4.0:
                    errors.append(f"{character_id}: impossible movement exceeds 4 m/s")
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
        if abs((finish - start) - float(scene["durationSeconds"])) > 0.05:
            errors.append(
                f"scene {scene['sceneNumber']}: timeline range does not match durationSeconds"
            )
        location = show["locations"].get(scene["locationId"]) or {}
        if not location.get("spatial"):
            errors.append(
                f"scene {scene['sceneNumber']}: location {scene['locationId']!r} "
                "has no spatial stage"
            )
            continue
        camera = scene["camera"]
        try:
            _camera_basis(camera)
        except ValueError as exc:
            errors.append(f"scene {scene['sceneNumber']}: {exc}")
            continue
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
            _, _, head_height = _stance_heights(state["stance"])
            head = add(vec(state["position"]), (0.0, 0.0, head_height))
            projected = project(head, camera)
            if not projected or not (
                -0.1 * PROXY_WIDTH <= projected[0] <= 1.1 * PROXY_WIDTH
                and -0.1 * PROXY_HEIGHT <= projected[1] <= 1.1 * PROXY_HEIGHT
            ):
                errors.append(
                    f"scene {scene['sceneNumber']}: {character_id}'s head is outside camera"
                )
            if length(sub(vec(state["position"]), vec(camera["position"]))) < 0.3:
                errors.append(f"scene {scene['sceneNumber']}: camera intersects {character_id}")
    return errors


def compile_spatial_video_prompt(episode: dict, scene: dict) -> str:
    if not scene.get("timeRangeSeconds") or not scene.get("camera"):
        return scene["videoPrompt"]
    start, finish = (float(value) for value in scene["timeRangeSeconds"])
    statements: list[str] = []
    for character_id in scene["characterIds"]:
        first = episode_character_state(episode, character_id, start)
        last = episode_character_state(episode, character_id, finish)
        displacement = sub(vec(last["position"]), vec(first["position"]))
        if length(displacement) < 0.08 and first["stance"] == last["stance"]:
            statements.append(
                f"{character_id} remains at the established mark with the established eyeline"
            )
        else:
            direction = "forward" if displacement[1] >= 0 else "backward"
            if abs(displacement[0]) > abs(displacement[1]):
                direction = "right" if displacement[0] > 0 else "left"
            statements.append(f"{character_id} moves one controlled step {direction}")
    camera = scene["camera"]
    if not camera.get("endPosition") and not camera.get("endLookAt"):
        camera_text = "CAMERA: locked."
    else:
        camera_delta = sub(
            vec(camera.get("endPosition") or camera["position"]),
            vec(camera["position"]),
        )
        if abs(camera_delta[0]) > max(abs(camera_delta[1]), abs(camera_delta[2])):
            camera_text = "CAMERA: makes one smooth lateral flyby along the established path."
        elif camera_delta[2] > 0.2:
            camera_text = "CAMERA: cranes upward smoothly along the established path."
        else:
            camera_text = "CAMERA: dollies smoothly along the established path."
    spatial_text = "VISUAL: " + "; ".join(statements) + "." if statements else ""
    return " ".join(part for part in (spatial_text, scene["videoPrompt"], camera_text) if part)


def scene_has_spatial_change(episode: dict, scene: dict) -> bool:
    if not scene.get("timeRangeSeconds") or not scene.get("camera"):
        return False
    camera = scene["camera"]
    if camera.get("endPosition") or camera.get("endLookAt"):
        return True
    start, finish = (float(value) for value in scene["timeRangeSeconds"])
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


def compile_spatial_image_summary(episode: dict, scene: dict) -> str:
    if not scene.get("timeRangeSeconds") or not scene.get("camera"):
        return ""
    start = float(scene["timeRangeSeconds"][0])
    camera = scene["camera"]
    clauses: list[str] = []
    timeline_tracks = (episode.get("spatialTimeline") or {}).get("characterTracks") or {}
    for character_id in scene["characterIds"]:
        state = episode_character_state(episode, character_id, start)
        _, _, head_height = _stance_heights(state["stance"])
        head = add(vec(state["position"]), (0.0, 0.0, head_height))
        projected = project(head, camera)
        if not projected:
            continue
        side = (
            "frame left"
            if projected[0] < PROXY_WIDTH * 0.42
            else "frame right"
            if projected[0] > PROXY_WIDTH * 0.58
            else "frame center"
        )
        target_id = state.get("lookAtId")
        target_direction = None
        if target_id in timeline_tracks:
            target = episode_character_state(episode, target_id, start)
            _, _, target_head_height = _stance_heights(target["stance"])
            target_projected = project(
                add(vec(target["position"]), (0.0, 0.0, target_head_height)),
                camera,
            )
            if target_projected:
                target_direction = (
                    "frame left"
                    if target_projected[0] < projected[0]
                    else "frame right"
                )
        if target_direction is None:
            yaw = math.radians(float(state["bodyYawDegrees"]))
            facing_point = add(head, (math.sin(yaw), math.cos(yaw), 0.0))
            facing_projected = project(facing_point, camera)
            if facing_projected:
                target_direction = (
                    "frame left"
                    if facing_projected[0] < projected[0]
                    else "frame right"
                )
        clause = f"{character_id} appears {side}"
        if target_direction:
            clause += (
                f" in three-quarter profile with body, head, and eyes turned "
                f"toward {target_direction}"
            )
        projected_feet = project(vec(state["position"]), camera)
        if projected_feet:
            if projected_feet[1] > PROXY_HEIGHT * 1.75:
                framing = "extreme facial close-up"
            elif projected_feet[1] > PROXY_HEIGHT * 1.15:
                framing = "tight head-and-shoulders close-up"
            elif projected_feet[1] > PROXY_HEIGHT * 0.9:
                framing = "medium close-up"
            else:
                framing = "medium or wider shot"
            clause += f"; the projected crop is a {framing}"
        clauses.append(clause)
    if not clauses:
        return ""
    return (
        "Ground-truth blocking: "
        + "; ".join(clauses)
        + ". Follow the mannequin orientation. Keep both pupils toward the stated frame edge; "
        "no visible character faces or makes eye contact with the viewer."
    )


def character_screen_position(
    episode: dict, scene: dict, character_id: str
) -> tuple[float, float]:
    start = float(scene["timeRangeSeconds"][0])
    state = episode_character_state(episode, character_id, start)
    _, _, head_height = _stance_heights(state["stance"])
    projected = project(
        add(vec(state["position"]), (0.0, 0.0, head_height)), scene["camera"]
    )
    if not projected:
        raise ValueError(
            f"Character {character_id!r} is behind scene {scene['sceneNumber']} camera"
        )
    return projected[0], projected[1]


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
        start, finish = (float(value) for value in scene["timeRangeSeconds"])
        samples = [("start", start), ("end", finish)]
        samples.extend(
            (f"guide_{index:02d}", float(time_seconds))
            for index, time_seconds in enumerate(scene.get("guideKeyframesSeconds") or [], start=1)
        )
        for label, time_seconds in samples:
            destination = proxy_frame_path(
                show["id"], episode["episodeNumber"], scene["sceneNumber"], label
            )
            render_scene_proxy(show, episode, scene, time_seconds, destination)
            generated.append(destination)
            condition_destination = proxy_frame_path(
                show["id"],
                episode["episodeNumber"],
                scene["sceneNumber"],
                f"{label}_condition",
            )
            render_scene_proxy(
                show,
                episode,
                scene,
                time_seconds,
                condition_destination,
                debug=False,
            )
            generated.append(condition_destination)
    return generated


def write_contact_sheet(paths: list[Path], destination: Path) -> None:
    starts = [path for path in paths if path.name.endswith("_start.png")]
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

    scripts = sorted(SCRIPTS_DIR.glob("*.json"))
    if args.show:
        scripts = [path for path in scripts if path.stem == args.show]
    if not scripts:
        raise SystemExit("No matching show scripts")
    for script in scripts:
        show = load_show(script)
        for episode in show["episodes"]:
            if args.episode is not None and episode["episodeNumber"] != args.episode:
                continue
            paths = generate_episode_previs(show, episode, args.scene)
            contact_sheet = previs_dir(show["id"], episode["episodeNumber"]) / "contact_sheet.png"
            write_contact_sheet(paths, contact_sheet)
            print(
                f"Wrote {len(paths)} proxy frames for {show['id']}/{episode['episodeNumber']} "
                f"to {contact_sheet.parent}",
                flush=True,
            )


if __name__ == "__main__":
    main()
