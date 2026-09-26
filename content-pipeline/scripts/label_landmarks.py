#!/usr/bin/env python3
"""Write landmark positions into the show script from a whole-location mesh.

Only locations with no people use one mesh. This step renders that mesh from
known cameras, asks Florence-2 where each landmark is, and unprojects the
depth buffer. A location with people already has those positions in the script.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from asset_resolver import location_has_people  # noqa: E402
from mesh_io import read_schema_mesh  # noqa: E402
from pipeline_paths import (  # noqa: E402
    OUTPUT_DIR,
    discover_show_scripts,
    load_content_env,
    location_dir,
    show_id_for_script,
)
from spatial_previs import (  # noqa: E402
    NEAR_CLIP,
    VIEWPORT_GRAY,
    Vec3,
    _camera_basis,
    unproject,
)

RENDER = 768
# The microsoft repo still ships remote code that crashes on Transformers 5.
# ComfyUI requires that Transformers line, which includes Florence-2 natively.
FLORENCE_ID = "florence-community/Florence-2-large"
FLORENCE_TASK = "<CAPTION_TO_PHRASE_GROUNDING>"


def landmark_phrase(landmark_id: str) -> str:
    """Spoken name Florence can ground. ``gate_column_l`` becomes ``left gate column``."""
    name = str(landmark_id)
    side = ""
    if name.endswith("_l"):
        side = "left "
        name = name[:-2]
    elif name.endswith("_r"):
        side = "right "
        name = name[:-2]
    return (side + name.replace("_", " ")).strip()


def pending_landmark_ids(landmarks: dict, force: bool) -> list[str]:
    pending: list[str] = []
    for landmark_id, landmark in landmarks.items():
        if not isinstance(landmark, dict):
            continue
        if landmark.get("position") is not None and not force:
            continue
        pending.append(str(landmark_id))
    return pending


def location_views(size: tuple[float, float, float]) -> list[dict]:
    """Five cameras around the location so a hidden side can still be named."""
    return [
        orbit_camera(size, azimuth, elevation)
        for azimuth, elevation in ((30, 25), (120, 25), (210, 25), (300, 25), (15, 68))
    ]


def orbit_camera(
    size: tuple[float, float, float],
    azimuth_degrees: float,
    elevation_degrees: float,
    fov: float = 36.0,
) -> dict:
    width, depth, height = size
    target = (0.0, 0.0, float(height) * 0.4)
    radius = 0.5 * math.hypot(width, depth, height)
    distance = radius / math.tan(math.radians(fov) / 2.0) * 1.25
    elevation = math.radians(elevation_degrees)
    azimuth = math.radians(azimuth_degrees)
    offset = (
        distance * math.cos(elevation) * math.sin(azimuth),
        -distance * math.cos(elevation) * math.cos(azimuth),
        distance * math.sin(elevation),
    )
    return {
        "position": [target[0] + offset[0], target[1] + offset[1], target[2] + offset[2]],
        "lookAt": list(target),
        "verticalFovDegrees": fov,
        "rollDegrees": 0.0,
    }


def accepted_boxes(
    boxes: list[tuple[float, float, float, float]],
    width: int,
    height: int,
) -> list[tuple[float, float, float, float]]:
    """Object-sized boxes, smallest first. Specks and near-full-frame boxes are dropped."""
    frame = float(width * height)
    accepted: list[tuple[float, tuple[float, float, float, float]]] = []
    for box in boxes:
        area = abs(box[2] - box[0]) * abs(box[3] - box[1])
        if area < frame * 0.002 or area > frame * 0.35:
            continue
        accepted.append((area, box))
    accepted.sort(key=lambda item: item[0])
    return [box for _area, box in accepted]


def choose_box(
    boxes: list[tuple[float, float, float, float]],
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    """Smallest box that is an object, not a speck and not the whole frame."""
    accepted = accepted_boxes(boxes, width, height)
    return accepted[0] if accepted else None


def surface_point(
    depth: np.ndarray,
    box: tuple[float, float, float, float],
    camera: dict,
    *,
    width: int,
    height: int,
) -> Vec3 | None:
    """Median schema point of the finite depth samples inside one box."""
    x0, x1 = sorted((int(round(box[0])), int(round(box[2]))))
    y0, y1 = sorted((int(round(box[1])), int(round(box[3]))))
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(width - 1, x1)
    y1 = min(height - 1, y1)
    if x1 < x0 or y1 < y0:
        return None
    region = depth[y0 : y1 + 1, x0 : x1 + 1]
    rows, cols = np.where(np.isfinite(region))
    if len(cols) < 8:
        return None
    if len(cols) > 4000:
        choice = np.linspace(0, len(cols) - 1, 4000).astype(int)
        rows = rows[choice]
        cols = cols[choice]
    points: list[Vec3] = []
    for row, col in zip(rows, cols):
        points.append(
            unproject(
                float(x0 + int(col)) + 0.5,
                float(y0 + int(row)) + 0.5,
                float(region[int(row), int(col)]),
                camera,
                width=width,
                height=height,
            )
        )
    return _median_point(points)


def fuse_points(points: list[Vec3], tolerance: float) -> Vec3 | None:
    """Keep the largest cluster of view hits and return its median."""
    if not points:
        return None
    clusters: list[list[Vec3]] = []
    for point in points:
        placed = False
        for cluster in clusters:
            if math.dist(point, _median_point(cluster)) <= tolerance:
                cluster.append(point)
                placed = True
                break
        if not placed:
            clusters.append([point])
    return _median_point(max(clusters, key=len))


def render_location(vertices: np.ndarray, faces: np.ndarray, camera: dict):
    from clay_gpu import ClayBatch, raster_clay

    basis = _camera_basis(camera, viewport_height=RENDER)
    batch = ClayBatch(vertices, faces, 176)
    return raster_clay(
        [batch],
        basis,
        width=RENDER,
        height=RENDER,
        near=NEAR_CLIP,
        background=VIEWPORT_GRAY,
    )


def locate_landmarks(
    vertices: np.ndarray,
    faces: np.ndarray,
    size: tuple[float, float, float],
    landmark_ids: list[str],
    grounder,
    destination: Path | None = None,
    views: list[dict] | None = None,
) -> dict[str, Vec3 | None]:
    """Name each landmark on the shaded mesh and return schema-space meters."""
    hits: dict[str, list[Vec3]] = {landmark_id: [] for landmark_id in landmark_ids}
    cameras = views if views is not None else location_views(size)
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
    for index, camera in enumerate(cameras):
        image, depth = render_location(vertices, faces, camera)
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        for landmark_id in landmark_ids:
            phrase = landmark_phrase(landmark_id)
            box = None
            point = None
            for candidate in accepted_boxes(grounder.boxes(image, phrase), RENDER, RENDER):
                point = surface_point(depth, candidate, camera, width=RENDER, height=RENDER)
                if point is not None:
                    box = candidate
                    break
            if box is None or point is None:
                continue
            hits[landmark_id].append(point)
            draw.rectangle(box, outline=(220, 80, 40), width=3)
            draw.text((box[0], max(0.0, box[1] - 16)), phrase, fill=(255, 220, 180))
        if destination is not None:
            annotated.save(destination / f"view_{index:02d}.png")
    tolerance = 0.25 * max(size[0], size[1], 1.0)
    return {landmark_id: fuse_points(points, tolerance) for landmark_id, points in hits.items()}


def set_landmark_position(
    text: str,
    location_id: str,
    landmark_id: str,
    position: tuple[float, float, float],
) -> str:
    """Replace one landmark object and leave the rest of the script text alone."""
    data = json.loads(text)
    landmarks = data["locations"][location_id]["spatial"]["landmarks"]
    if landmark_id not in landmarks or not isinstance(landmarks[landmark_id], dict):
        raise KeyError(f"{location_id}.{landmark_id}")
    literal = json.dumps(
        {"position": [round(float(value), 2) for value in position]},
        separators=(", ", ": "),
    )
    start = _landmark_value_start(text, location_id, landmark_id)
    _value, end = json.JSONDecoder().raw_decode(text, start)
    return text[:start] + literal + text[end:]


def _landmark_value_start(text: str, location_id: str, landmark_id: str) -> int:
    """Index of the landmark object's ``{``, after that location's landmarks map.

    A later ``lookAtId`` that repeats the same name is a string, so it is skipped.
    """
    location_at = text.find(f'"{location_id}"')
    if location_at < 0:
        raise KeyError(location_id)
    landmarks_at = text.find('"landmarks"', location_at)
    if landmarks_at < 0:
        raise KeyError(f"{location_id}.landmarks")
    needle = f'"{landmark_id}"'
    index = landmarks_at
    while True:
        index = text.find(needle, index)
        if index < 0:
            raise KeyError(f"{location_id}.{landmark_id}")
        after_key = index + len(needle)
        colon = text.find(":", after_key)
        if colon < 0 or text[after_key:colon].strip() != "":
            index = after_key
            continue
        value_at = colon + 1
        while value_at < len(text) and text[value_at].isspace():
            value_at += 1
        if text.startswith("{", value_at):
            return value_at
        index = after_key


class FlorenceGrounder:
    """Open-vocabulary boxes from Florence-2. One model, no mask model."""

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._device = "cpu"

    def boxes(self, image: Image.Image, phrase: str) -> list[tuple[float, float, float, float]]:
        self._load()
        import torch

        assert self._model is not None and self._processor is not None
        inputs = self._processor(text=FLORENCE_TASK + phrase, images=image, return_tensors="pt")
        moved = {
            key: value.to(self._device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with torch.no_grad():
            generated = self._model.generate(
                **moved,
                max_new_tokens=256,
                num_beams=3,
            )
        text = self._processor.batch_decode(generated, skip_special_tokens=False)[0]
        parsed = self._processor.post_process_generation(
            text,
            task=FLORENCE_TASK,
            image_size=(image.width, image.height),
        )
        payload = parsed.get(FLORENCE_TASK, {})
        boxes = payload.get("bboxes") if isinstance(payload, dict) else None
        if not boxes:
            return []
        return [tuple(float(value) for value in box) for box in boxes]

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, Florence2ForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Florence-2 needs transformers, timm, and einops. "
                "Run `pnpm run content:asset-deps`."
            ) from exc
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {FLORENCE_ID} on {self._device}", flush=True)
        self._model = Florence2ForConditionalGeneration.from_pretrained(
            FLORENCE_ID,
            attn_implementation="eager",
            dtype=torch.float32,
        ).to(self._device)
        self._processor = AutoProcessor.from_pretrained(FLORENCE_ID)
        self._model.eval()


def label_script(
    script_path: Path,
    grounder,
    *,
    force: bool = False,
    only_location: str | None = None,
) -> int:
    text = script_path.read_text(encoding="utf-8")
    show = json.loads(text)
    show_id = show_id_for_script(script_path)
    written = 0
    saw_location = False
    for location_id, location in (show.get("locations") or {}).items():
        if only_location and str(location_id) != only_location:
            continue
        saw_location = True
        if not isinstance(location, dict):
            continue
        if location_has_people(show, str(location_id)):
            print(f"{location_id}: positions are authored with each landmark", flush=True)
            continue
        spatial = location.get("spatial") or {}
        landmarks = spatial.get("landmarks") or {}
        pending = pending_landmark_ids(landmarks, force)
        if not landmarks:
            continue
        if not pending:
            print(f"{location_id}: positions already set", flush=True)
            continue
        mesh_path = location_dir(show_id, str(location_id)) / "model.glb"
        if not mesh_path.is_file():
            raise SystemExit(
                f"No mesh at {mesh_path}. Run `pnpm run content:assets` before landmarks."
            )
        size = tuple(float(value) for value in spatial["sizeMeters"])
        destination = OUTPUT_DIR / "landmarks" / show_id / str(location_id)
        print(f"{location_id}: placing {', '.join(pending)}", flush=True)
        vertices, faces = read_schema_mesh(mesh_path)
        found = locate_landmarks(vertices, faces, size, pending, grounder, destination)  # type: ignore[arg-type]
        record: dict[str, dict] = {}
        for landmark_id, point in found.items():
            if point is None:
                print(f"  {landmark_id}: not found", flush=True)
                continue
            text = set_landmark_position(text, str(location_id), landmark_id, point)
            rounded = [round(float(value), 2) for value in point]
            record[landmark_id] = {"position": rounded}
            written += 1
            print(f"  {landmark_id}: {rounded}", flush=True)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "landmarks.json").write_text(
            json.dumps({"locationId": location_id, "landmarks": record}, indent=2) + "\n",
            encoding="utf-8",
        )
    if only_location and not saw_location:
        raise SystemExit(f"No location {only_location!r} in {script_path}")
    if written:
        script_path.write_text(text, encoding="utf-8")
        print(f"Updated {script_path}", flush=True)
    return written


def _median_point(points: list[Vec3]) -> Vec3:
    stacked = np.asarray(points, dtype=np.float64)
    median = np.median(stacked, axis=0)
    return (float(median[0]), float(median[1]), float(median[2]))


def main() -> None:
    load_content_env()
    parser = argparse.ArgumentParser(
        description="Fill landmark positions in the show script from the location mesh."
    )
    parser.add_argument("--show", help="Only this show id")
    parser.add_argument("--location", help="Only this location id")
    parser.add_argument("--force", action="store_true", help="Replace positions that are already set")
    args = parser.parse_args()
    scripts = discover_show_scripts(args.show)
    if not scripts:
        target = f" for show {args.show!r}" if args.show else ""
        raise SystemExit(f"No show JSON files found{target}")
    grounder = FlorenceGrounder()
    for script in scripts:
        label_script(script, grounder, force=args.force, only_location=args.location)


if __name__ == "__main__":
    main()
