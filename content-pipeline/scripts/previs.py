#!/usr/bin/env python3
"""content:previs. Render every shot from the world, write its guides, and check the script.

Each sampled moment is observed once (``observe.observe``). The start and end
frames become the guides a still is drawn from. All samples feed the checks.
Every scene is rendered even when one fails, and every failure is listed.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from checks import check_scene
from clip_spec import FPS as CLIP_FPS
from clip_spec import CLIP_HEIGHT, CLIP_WIDTH, clip_frame_count, fit_to_clip
from coords import schema_to_gltf
from observe import in_shot, observe
from pipeline_paths import (
    OUTPUT_DIR,
    blockout_video_path,
    clay_24fps_path,
    clay_frame_path,
    contact_sheet_path,
    discover_show_scripts,
    episode_blockout_path,
    guide_path,
    scene_description_path,
)
from render import HEIGHT, WIDTH, render
from world import camera_at, effect_entity, entities_at, load_show, required_meshes, scene_has_spatial_change

BLOCKOUT_FPS = 8
# Plate color is banded across nearby shades; one band is one flat area.
CLOTHES_COLOR_BIN = 32
# A window this wide replaces speckle with the color around it.
CLOTHES_FLATTEN_RADIUS = 3


def sample_times(start: float, finish: float) -> list[float]:
    count = max(2, int(round((finish - start) * BLOCKOUT_FPS)) + 1)
    return [start + (finish - start) * index / (count - 1) for index in range(count)]


def observation_path(show_id: str, episode_number: int, scene_number: int, label: str) -> Path:
    return guide_path(show_id, episode_number, scene_number, label, "observation").with_suffix(".json")


def load_observation(show_id: str, episode_number: int, scene_number: int, label: str = "start") -> dict:
    path = observation_path(show_id, episode_number, scene_number, label)
    if not path.is_file():
        raise SystemExit(f"Missing {path}. Run `pnpm run content:previs` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def depth_image(depth: np.ndarray) -> Image.Image:
    """Near surfaces are bright. Empty space is black."""
    valid = np.isfinite(depth)
    gray = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        near, far = np.percentile(depth[valid], 1), np.percentile(depth[valid], 99)
        normalized = np.clip((far - depth) / max(far - near, 1e-3), 0.0, 1.0)
        gray[valid] = np.rint(48.0 + normalized[valid] * 207.0).astype(np.uint8)
    return Image.fromarray(np.stack((gray,) * 3, axis=-1))


def edge_image(depth: np.ndarray) -> Image.Image:
    """White where depth jumps or a surface meets empty space."""
    valid = np.isfinite(depth)
    values = np.where(valid, depth, 0.0).astype(np.float32)
    scale = np.maximum(values, 1e-3)
    jump = np.zeros(valid.shape, dtype=bool)
    jump[:, 1:] |= np.abs(values[:, 1:] - values[:, :-1]) / scale[:, 1:] > 0.04
    jump[1:, :] |= np.abs(values[1:, :] - values[:-1, :]) / scale[1:, :] > 0.04
    rim = np.zeros(valid.shape, dtype=bool)
    rim[:, 1:] |= ~valid[:, :-1]
    rim[:, :-1] |= ~valid[:, 1:]
    rim[1:, :] |= ~valid[:-1, :]
    rim[:-1, :] |= ~valid[1:, :]
    gray = np.where(valid & (jump | rim), 255, 0).astype(np.uint8)
    return Image.fromarray(np.stack((gray,) * 3, axis=-1))


def _box_count(mask: np.ndarray, radius: int) -> np.ndarray:
    values = np.pad(mask.astype(np.int32), radius)
    integral = np.zeros((values.shape[0] + 1, values.shape[1] + 1), dtype=np.int32)
    integral[1:, 1:] = values.cumsum(0).cumsum(1)
    height, width = mask.shape
    y, x = np.arange(height), np.arange(width)
    y2, x2 = y + 2 * radius + 1, x + 2 * radius + 1
    return integral[y2[:, None], x2[None, :]] - integral[y[:, None], x2[None, :]] - integral[y2[:, None], x[None, :]] + integral[y[:, None], x[None, :]]


def flatten_colors(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Flat color inside each area of one figure. A small stain takes the color around it."""
    if not mask.any():
        return image
    rows, columns = np.nonzero(mask.any(axis=1))[0], np.nonzero(mask.any(axis=0))[0]
    pad = CLOTHES_FLATTEN_RADIUS + 1
    top, bottom = max(rows[0] - pad, 0), min(rows[-1] + pad + 1, mask.shape[0])
    left, right = max(columns[0] - pad, 0), min(columns[-1] + pad + 1, mask.shape[1])
    out = image.copy()
    out[top:bottom, left:right] = _flatten(image[top:bottom, left:right], mask[top:bottom, left:right])
    return out


def _flatten(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    covered = int(mask.sum())
    quantized = image.astype(np.int32) // CLOTHES_COLOR_BIN
    keys = np.zeros(mask.shape, dtype=np.int32)
    keys[mask] = (quantized[..., 0] * 256 + quantized[..., 1] * 16 + quantized[..., 2])[mask]
    present = [int(key) for key in np.unique(keys[mask])]
    medians = {key: np.median(image[mask & (keys == key)], axis=0).astype(np.uint8) for key in present}
    winner = np.argmax(np.stack([_box_count((keys == key) & mask, CLOTHES_FLATTEN_RADIUS) for key in present]), axis=0)
    flat = np.zeros(mask.shape, dtype=np.int32)
    for index, key in enumerate(present):
        flat[mask & (winner == index)] = key
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    minimum = max(20, int(covered * 0.01))
    for key in np.unique(flat[mask]):
        labeled, count = ndimage.label(flat == key, structure=cross)
        sizes = np.bincount(labeled.ravel())
        for component, box in enumerate(ndimage.find_objects(labeled), start=1):
            if box is None or sizes[component] >= minimum:
                continue
            # Work in the patch's own box, one pixel larger, not the whole figure.
            rows = slice(max(box[0].start - 1, 0), box[0].stop + 1)
            columns = slice(max(box[1].start - 1, 0), box[1].stop + 1)
            area = labeled[rows, columns] == component
            border = ndimage.binary_dilation(area, structure=cross) & mask[rows, columns] & ~area
            if border.any():
                neighbors, frequency = np.unique(flat[rows, columns][border], return_counts=True)
                flat[rows, columns][area] = neighbors[int(np.argmax(frequency))]
    out = image.copy()
    for key in np.unique(flat[mask]):
        out[mask & (flat == key)] = medians[int(key)]
    return out


def color_image(frame, entities, labels: np.ndarray | None, backdrop: dict | None) -> Image.Image:
    """Every entity in flat plate colors, empty space in the location's flat colors.

    One picture for everything the camera sees. Occlusion is the render's.
    """
    canvas = np.zeros((*frame.entity.shape, 3), dtype=np.uint8)
    if labels is not None:
        for number, name in enumerate(("sky", "ground", "surround"), start=1):
            canvas[labels == number] = backdrop[f"{name}Color"]
    for index, entity in enumerate(entities):
        if entity.kind == "floor":
            continue
        mask = frame.entity == index + 1
        canvas[mask] = flatten_colors(frame.albedo, mask)[mask]
    return Image.fromarray(canvas)


# Every person, prop, and landmark is drawn alone and pasted in; only empty space is drawn in the shot.
DRAWN_ALONE = ("character", "prop", "landmark")


def crop_window(entity, camera: dict) -> list[float] | None:
    """A frame-shaped window around the whole entity and its effects, so it is always drawn whole.

    A person cut by the frame edge is still a whole person in its window (the
    window may run past the frame); the shot keeps only what it shows. Only an
    entity bigger than the frame gets a frame-sized window over the part in view,
    so a drawing is only ever scaled down into the shot, never up.
    """
    from render import screen_box

    parts = [entity, *(effect_entity(entity, number) for number in range(len(entity.effects)))]
    boxes = [box for box in (screen_box(part, camera) for part in parts) if box is not None]
    if not boxes:
        return None
    x0, y0 = min(box[0] for box in boxes), min(box[1] for box in boxes)
    x1, y1 = max(box[2] for box in boxes), max(box[3] for box in boxes)
    if min(x1, WIDTH) <= max(x0, 0.0) or min(y1, HEIGHT) <= max(y0, 0.0):
        return None
    height = max((y1 - y0) * 1.06, (x1 - x0) * 1.06 * HEIGHT / WIDTH, 16.0)
    if height <= HEIGHT:
        width = height * WIDTH / HEIGHT
        left, top = (x0 + x1) / 2 - width / 2, (y0 + y1) / 2 - height / 2
    else:
        x0, y0, x1, y1 = max(x0, 0.0), max(y0, 0.0), min(x1, float(WIDTH)), min(y1, float(HEIGHT))
        height = min(max((y1 - y0) * 1.06, (x1 - x0) * 1.06 * HEIGHT / WIDTH, 16.0), float(HEIGHT))
        width = height * WIDTH / HEIGHT
        left = min(max((x0 + x1) / 2 - width / 2, 0.0), WIDTH - width)
        top = min(max((y0 + y1) / 2 - height / 2, 0.0), HEIGHT - height)
    return [round(left, 2), round(top, 2), round(left + width, 2), round(top + height, 2)]


# A drawn outline may differ from the mesh by this much (hair, cloth), and is softened over it.
SHOWN_GROW_PIXELS = 4
# Flames and smoke spill past their box, and fade out over a wider edge.
EFFECT_GROW_PIXELS = 8


def shown_mask(frame, entities, index: int, alone, effects=()) -> Image.Image:
    """Soft mask of where the shot shows this entity: its rendered silhouette, grown a
    little so the drawing's own hair and cloth edges survive, plus the boxes of its
    effects (``effects``: each box rendered alone), minus what is in front.

    A drawing can never land outside it, whatever was drawn. The entity does not
    hide its own effects: its drawing already has them in front of or behind it.
    """
    from scipy import ndimage

    def part(render_alone, grow: int, sigma: float) -> np.ndarray:
        inside = ndimage.binary_dilation(np.isfinite(render_alone.depth), iterations=grow)
        shown = inside & ~hidden_mask(frame, entities, index, render_alone)
        return ndimage.gaussian_filter(shown.astype(np.float32), sigma=sigma)

    soft = part(alone, SHOWN_GROW_PIXELS, 1.0)
    for effect in effects:
        soft = np.maximum(soft, part(effect, EFFECT_GROW_PIXELS, 3.0))
    return Image.fromarray(np.rint(np.clip(soft, 0, 1) * 255).astype(np.uint8))


def hidden_mask(frame, entities, index: int, alone) -> np.ndarray:
    """Where something closer than this entity covers it in the shot. The floor hides nothing.

    Inside its silhouette that is anything in front of its own surface; just
    outside (hair or cloth the drawing adds) anything nearer than its nearest point.
    """
    own = alone.depth
    inside = np.isfinite(own)
    if not inside.any():
        return np.zeros(own.shape, dtype=bool)
    floors = [number + 1 for number, entity in enumerate(entities) if entity.kind == "floor"]
    other = (frame.entity > 0) & (frame.entity != index + 1) & ~np.isin(frame.entity, floors)
    slack = np.maximum(np.float32(0.05), np.where(inside, own, 0) * np.float32(0.02))
    in_front = np.where(inside, frame.depth < own - slack, frame.depth < np.float32(own[inside].min()))
    return other & in_front


def write_guides(show, episode, scene, label, observation, frame, labels, entities) -> list[Path]:
    """Scene guides, and for each person, prop, and landmark in the shot its own guides, drawn alone.

    The crop is the same camera magnified onto its part of the frame, so pose and
    turn are exactly the shot's. ``drawn_<id>_shown`` is where the shot shows it
    and its effects, the only place its drawing may land. The guides show only
    its solid mesh: an effect is drawn from its look picture and words.
    """
    ids = (show["id"], episode["episodeNumber"], scene["sceneNumber"])
    location = show["locations"][scene["locationId"]]
    camera = observation["camera"]
    images = {
        clay_frame_path(*ids, label): Image.fromarray(frame.clay),
        guide_path(*ids, label, "depth"): depth_image(frame.depth),
        guide_path(*ids, label, "edges"): edge_image(frame.depth),
        guide_path(*ids, label, "color"): color_image(frame, entities, labels, location["backdrop"]),
    }
    observation["effects"] = []
    for index, entity in enumerate(entities):
        if entity.kind not in DRAWN_ALONE:
            continue
        effects = [render([effect_entity(entity, number)], camera) for number in range(len(entity.effects))]
        seen = [bool(np.isfinite(effect.depth).any()) for effect in effects]
        if not in_shot(observation, entity.id) and not any(seen):
            continue
        crop = crop_window(entity, camera)
        if crop is None:
            continue
        alone = render([entity], camera, crop=crop)
        prefix = f"drawn_{entity.id}"
        images[guide_path(*ids, label, f"{prefix}_depth")] = depth_image(alone.depth)
        images[guide_path(*ids, label, f"{prefix}_edges")] = edge_image(alone.depth)
        images[guide_path(*ids, label, f"{prefix}_color")] = color_image(alone, [entity], None, None)
        shown = shown_mask(frame, entities, index, render([entity], camera), effects)
        images[guide_path(*ids, label, f"{prefix}_shown")] = shown
        observation["entities"][entity.id]["crop"] = crop
        observation["effects"] += [
            {"id": entity.id, "appearance": effect["appearance"]} for effect, visible in zip(entity.effects, seen) if visible
        ]
    for path, image in images.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
    record = observation_path(*ids, label)
    record.write_text(json.dumps(observation, indent=2) + "\n", encoding="utf-8")
    return [*images, record]


def write_world(show, episode, scene, samples) -> Path:
    """The world the viewer replays: each mesh once, then every entity's pose per sample."""
    models = {}
    frames = []
    for time_seconds, observation, entities in samples:
        placed = []
        for entity in entities:
            if entity.mesh.path is None:
                continue
            models[entity.id] = {"kind": entity.kind, "model": entity.mesh.path.relative_to(OUTPUT_DIR).as_posix()}
            placed.append(
                {
                    "id": entity.id,
                    "position": list(schema_to_gltf(entity.offset)),
                    "yawDegrees": round(entity.yaw_degrees, 3),
                    "visible": in_shot(observation, entity.id),
                }
            )
        camera = observation["camera"]
        frames.append(
            {
                "timeSeconds": time_seconds,
                "camera": {
                    "position": list(schema_to_gltf(tuple(camera["position"]))),
                    "lookAt": list(schema_to_gltf(tuple(camera["lookAt"]))),
                    "verticalFovDegrees": camera["verticalFovDegrees"],
                    "rollDegrees": camera["rollDegrees"],
                },
                "entities": placed,
            }
        )
    payload = {
        "space": "gltf-y-up",
        "showId": show["id"],
        "episodeNumber": episode["episodeNumber"],
        "sceneNumber": scene["sceneNumber"],
        "locationId": scene["locationId"],
        "timeRangeSeconds": scene["timeRangeSeconds"],
        "models": models,
        "frames": frames,
    }
    destination = scene_description_path(show["id"], episode["episodeNumber"], scene["sceneNumber"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def write_clay_24fps(show, episode, scene) -> Path | None:
    """Clay at clip size and rate, the input Depth Anything turns into the control video."""
    from ffmpeg_tools import encode_rgb_frames

    path = clay_24fps_path(show["id"], episode["episodeNumber"], scene["sceneNumber"])
    if not scene_has_spatial_change(episode, scene):
        path.unlink(missing_ok=True)
        return None
    start = float(scene["timeRangeSeconds"][0])
    raw = b"".join(
        fit_to_clip(
            Image.fromarray(render(entities_at(show, episode, scene, t), camera_at(scene, t), full=False).clay)
        ).tobytes()
        for t in (start + index / CLIP_FPS for index in range(clip_frame_count(scene["timeRangeSeconds"])))
    )
    encode_rgb_frames(raw, CLIP_WIDTH, CLIP_HEIGHT, CLIP_FPS, path, crf=14)
    return path


def _clear_guides(show, episode, scene) -> None:
    """Previs owns the guides folder. The control video belongs to content:generate."""
    folder = guide_path(show["id"], episode["episodeNumber"], scene["sceneNumber"], "start", "depth").parent
    if not folder.is_dir():
        return
    for item in folder.iterdir():
        if item.name == "control_depth.mp4":
            continue
        shutil.rmtree(item) if item.is_dir() else item.unlink()


def render_scene(show: dict, episode: dict, scene: dict) -> tuple[list[str], list[Path]]:
    """Render, write guides and playblast, and return the scene's check errors."""
    from ffmpeg_tools import encode_rgb_frames

    _clear_guides(show, episode, scene)
    start, finish = (float(value) for value in scene["timeRangeSeconds"])
    times = sample_times(start, finish)
    samples = []
    clay = []
    written: list[Path] = []
    for time_seconds in times:
        label = "start" if time_seconds == times[0] else "end" if time_seconds == times[-1] else None
        observation, frame, labels, entities = observe(show, episode, scene, time_seconds, full=label is not None)
        samples.append((time_seconds, observation, entities))
        clay.append(frame.clay)
        if label:
            written += write_guides(show, episode, scene, label, observation, frame, labels, entities)
    seen = [
        entity_id
        for entity_id, entry in samples[0][1]["entities"].items()
        if entry["kind"] != "backdrop" and in_shot(samples[0][1], entity_id)
    ]
    print(f"  start frame shows: {', '.join(seen) or 'nothing'}", flush=True)
    video = blockout_video_path(show["id"], episode["episodeNumber"], scene["sceneNumber"])
    encode_rgb_frames(b"".join(image.tobytes() for image in clay), WIDTH, HEIGHT, BLOCKOUT_FPS, video)
    written += [video, write_world(show, episode, scene, samples)]
    clay_24 = write_clay_24fps(show, episode, scene)
    if clay_24:
        written.append(clay_24)
    errors = check_scene(show, episode, scene, samples)
    record = record_path(show, episode, scene)
    record.write_text(json.dumps({"inputs": inputs_digest(show, episode, scene), "errors": errors}, indent=2), encoding="utf-8")
    return errors, written + [record]


def record_path(show: dict, episode: dict, scene: dict) -> Path:
    return guide_path(show["id"], episode["episodeNumber"], scene["sceneNumber"], "previs", "record").with_suffix(".json")


def inputs_digest(show: dict, episode: dict, scene: dict) -> str:
    """Everything a scene's previs depends on: the script, the meshes, the settings, and this code."""
    import hashlib

    digest = hashlib.sha256()
    digest.update(json.dumps([show, episode["spatialTimeline"], scene], sort_keys=True).encode("utf-8"))
    for path in required_meshes(show):
        digest.update(f"{path}:{path.stat().st_mtime_ns if path.is_file() else 0}".encode("utf-8"))
    here = Path(__file__).resolve().parent
    for name in ("world.py", "render.py", "observe.py", "checks.py", "previs.py", "body_parts.py", "../prompts.json"):
        digest.update((here / name).read_bytes())
    return digest.hexdigest()


def current_previs(show: dict, episode: dict, scene: dict) -> list[str] | None:
    """The scene's check errors from the last previs, when nothing it depends on has changed since."""
    record = record_path(show, episode, scene)
    if not record.is_file():
        return None
    stored = json.loads(record.read_text(encoding="utf-8"))
    return stored["errors"] if stored.get("inputs") == inputs_digest(show, episode, scene) else None


def require_meshes(show: dict) -> None:
    missing = [path for path in required_meshes(show) if not path.is_file()]
    if missing:
        listed = "\n- ".join(str(path) for path in missing)
        raise SystemExit(
            f"Previs draws the real meshes, and these are missing:\n- {listed}\n"
            "Run `pnpm run content:plates`, review the plates, then `pnpm run content:assets`."
        )


def previs_episode(show: dict, episode: dict, scene_number: int | None = None) -> tuple[list[str], list[Path]]:
    require_meshes(show)
    errors: list[str] = []
    written: list[Path] = []
    for scene in episode["scenes"]:
        if scene_number is not None and scene["sceneNumber"] != scene_number:
            continue
        print(f"Previs scene {scene['sceneNumber']:02d} at {scene['locationId']}...", flush=True)
        scene_errors, scene_written = render_scene(show, episode, scene)
        errors += scene_errors
        written += scene_written
    return errors, written


def write_contact_sheet(paths: list[Path], destination: Path) -> None:
    starts = [path for path in paths if path.name == "start.png"]
    if not starts:
        return
    width = 240
    height = round(width * HEIGHT / WIDTH)
    sheet = Image.new("RGB", (4 * width, math.ceil(len(starts) / 4) * height))
    for index, path in enumerate(starts):
        image = Image.open(path).convert("RGB")
        image.thumbnail((width, height))
        sheet.paste(image, ((index % 4) * width, (index // 4) * height))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def write_episode_blockout(show: dict, episode: dict) -> Path | None:
    """Join scene playblasts into one episode file when every scene has one."""
    from ffmpeg_tools import concat_videos

    paths = [blockout_video_path(show["id"], episode["episodeNumber"], s["sceneNumber"]) for s in episode["scenes"]]
    if not all(path.is_file() and path.stat().st_size >= 1024 for path in paths):
        return None
    destination = episode_blockout_path(show["id"], episode["episodeNumber"])
    concat_videos(paths, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Render clay previs, guides, and script checks.")
    parser.add_argument("--show", help="Only this show id")
    parser.add_argument("--episode", type=int, help="Only this episode")
    parser.add_argument("--scene", type=int, help="Only this scene (requires --episode)")
    parser.add_argument("--force", action="store_true", help="Accepted for content:render; previs always rewrites")
    parser.add_argument("--seed", type=int, help="Accepted for content:render; previs has no seed")
    args = parser.parse_args()
    if args.scene is not None and args.episode is None:
        parser.error("--scene requires --episode")
    scripts = discover_show_scripts(args.show)
    if not scripts:
        raise SystemExit("No matching show scripts")
    failures: list[str] = []
    for script in scripts:
        show = load_show(script)
        for episode in show["episodes"]:
            if args.episode is not None and episode["episodeNumber"] != args.episode:
                continue
            errors, written = previs_episode(show, episode, args.scene)
            failures += [f"{show['id']}/{episode['episodeNumber']} {error}" for error in errors]
            write_contact_sheet(written, contact_sheet_path(show["id"], episode["episodeNumber"]))
            blockout = write_episode_blockout(show, episode)
            print(f"Wrote {len(written)} previs files for {show['id']}/{episode['episodeNumber']}", flush=True)
            if blockout:
                print(f"Wrote {blockout}", flush=True)
    if failures:
        raise SystemExit("The script does not match its 3D scene:\n- " + "\n- ".join(failures))
    print("Every shot matches its script.", flush=True)


if __name__ == "__main__":
    main()
