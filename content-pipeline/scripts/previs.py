#!/usr/bin/env python3
"""content:previs. Render every shot from the world, write what it sees, and check the script.

Each sampled moment is observed once (``observe.observe``). The start and end
frames are kept as clay pictures and observations (who is in the frame, where);
the words for the video model are built from those. All samples feed the checks.
Every scene is rendered even when one fails, and every failure is listed.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

from PIL import Image

from checks import check_scene
from coords import schema_to_gltf
from observe import in_shot, observe
from pipeline_paths import (
    OUTPUT_DIR,
    blockout_video_path,
    clay_frame_path,
    contact_sheet_path,
    discover_show_scripts,
    episode_blockout_path,
    guide_path,
    scene_description_path,
)
from render import HEIGHT, WIDTH
from wan import FPS as CLIP_FPS
from world import load_show, required_meshes

BLOCKOUT_FPS = 8


def sample_times(start: float, finish: float) -> list[float]:
    """Moments of the shot, from its first frame to its last, one video frame before the cut:
    at the cut itself the next shot's blocking already holds (someone has moved on)."""
    finish = max(start, finish - 1.0 / CLIP_FPS)
    count = max(2, int(round((finish - start) * BLOCKOUT_FPS)) + 1)
    return [start + (finish - start) * index / (count - 1) for index in range(count)]


def observation_path(show_id: str, episode_number: int, scene_number: int, label: str) -> Path:
    return guide_path(show_id, episode_number, scene_number, label, "observation").with_suffix(".json")


def load_observation(show_id: str, episode_number: int, scene_number: int, label: str = "start") -> dict:
    path = observation_path(show_id, episode_number, scene_number, label)
    if not path.is_file():
        raise SystemExit(f"Missing {path}. Run `pnpm run content:previs` first.")
    return json.loads(path.read_text(encoding="utf-8"))


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


def render_scene(show: dict, episode: dict, scene: dict) -> tuple[list[str], list[Path]]:
    """Render the shot, write its clay frames, observations, and playblast, and return its check errors."""
    from ffmpeg_tools import encode_rgb_frames

    ids = (show["id"], episode["episodeNumber"], scene["sceneNumber"])
    guides = guide_path(*ids, "start", "observation").parent
    if guides.is_dir():
        shutil.rmtree(guides)
    guides.mkdir(parents=True)
    start, finish = (float(value) for value in scene["timeRangeSeconds"])
    times = sample_times(start, finish)
    samples = []
    clay = []
    written: list[Path] = []
    for time_seconds in times:
        label = "start" if time_seconds == times[0] else "end" if time_seconds == times[-1] else None
        observation, frame, entities = observe(show, episode, scene, time_seconds, full=label is not None)
        samples.append((time_seconds, observation, entities))
        clay.append(frame.clay)
        if label:
            picture, record = clay_frame_path(*ids, label), observation_path(*ids, label)
            Image.fromarray(frame.clay).save(picture)
            record.write_text(json.dumps(observation, indent=2) + "\n", encoding="utf-8")
            written += [picture, record]
    seen = [entity_id for entity_id, entry in samples[0][1]["entities"].items() if entry["kind"] != "backdrop" and in_shot(samples[0][1], entity_id)]
    print(f"  start frame shows: {', '.join(seen) or 'nothing'}", flush=True)
    video = blockout_video_path(*ids)
    encode_rgb_frames(b"".join(image.tobytes() for image in clay), WIDTH, HEIGHT, BLOCKOUT_FPS, video)
    written += [video, write_world(show, episode, scene, samples)]
    errors = check_scene(show, episode, scene, samples)
    record = record_path(show, episode, scene)
    record.write_text(json.dumps({"inputs": inputs_digest(show, episode, scene), "errors": errors}, indent=2), encoding="utf-8")
    return errors, written + [record]


def record_path(show: dict, episode: dict, scene: dict) -> Path:
    return guide_path(show["id"], episode["episodeNumber"], scene["sceneNumber"], "previs", "record").with_suffix(".json")


def inputs_digest(show: dict, episode: dict, scene: dict) -> str:
    """Everything a scene's previs depends on: the script, the meshes, and this code."""
    import hashlib

    digest = hashlib.sha256()
    digest.update(json.dumps([show, episode["spatialTimeline"], scene], sort_keys=True).encode("utf-8"))
    for path in required_meshes(show):
        digest.update(f"{path}:{path.stat().st_mtime_ns if path.is_file() else 0}".encode("utf-8"))
    here = Path(__file__).resolve().parent
    for name in ("world.py", "render.py", "observe.py", "checks.py", "previs.py", "shots.py"):
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
    parser.add_argument("--size", help="Accepted for content:render; previs renders at its own size")
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
