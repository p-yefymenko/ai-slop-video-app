"""Turn each location description into one plate, then one mesh.

Qwen draws the whole place. TRELLIS.2 turns that reviewed picture into one mesh.
The files stay under ``output/plates/<show>`` and ``output/assets/<show>``.
Characters stay out of the mesh. Their clay mannequins are built at previs time.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from asset_generate import generate_asset_mesh, plate_is_ready
from asset_sources import materialize_mesh
from mesh_io import (
    DECIMATOR,
    TRIANGLE_BUDGET,
    fit_to_size,
    read_schema_mesh,
    require_triangle_budget,
    write_schema_glb,
)
from pipeline_paths import OUTPUT_DIR

GENERATED_SOURCE = "trellis2"


@dataclass
class AssetRequest:
    consumer_id: str
    label: str
    size: tuple[float, float, float]
    location_id: str
    appearance: str


@dataclass
class ResolvedLocation:
    location_id: str
    glb: Path
    source: str
    source_id: str
    title: str
    size: tuple[float, float, float]


class AssetResolver:
    def __init__(
        self,
        show_id: str,
        *,
        output_dir: Path = OUTPUT_DIR,
        generator=None,
        offline: bool = False,
        refresh: bool = False,
        write_thumbs: bool = False,
        triangle_budget: int = TRIANGLE_BUDGET,
    ) -> None:
        self.show_id = show_id
        self.output_dir = output_dir
        self.generator = generator if generator is not None else generate_asset_mesh
        self.offline = offline
        self.refresh = refresh
        self.write_thumbs = write_thumbs
        self.triangle_budget = require_triangle_budget(triangle_budget)
        self.warnings: list[str] = []

    def resolve_show(self, show: dict) -> dict[str, ResolvedLocation]:
        resolved: dict[str, ResolvedLocation] = {}
        for request in collect_requests(show):
            resolved[request.consumer_id] = self.resolve_one(request)
        return resolved

    def resolve_one(self, request: AssetRequest) -> ResolvedLocation:
        plate = self._plate_path(request.location_id)
        if not plate_is_ready(plate):
            raise RuntimeError(
                f"{request.consumer_id}: no reviewed plate at {plate}. "
                "Run `pnpm run content:plates`."
            )
        plate_record = _read_json(plate.with_name("plate.json"), {})
        digest = asset_hash(GENERATED_SOURCE, appearance_source_id(request.appearance), request.size)
        if plate_record.get("descriptionHash") not in (None, digest):
            raise RuntimeError(
                f"{request.consumer_id}: the plate does not match the location text. "
                "Run `pnpm run content:plates`."
            )
        cached = self._cached(request, digest)
        if cached is not None and not self.refresh:
            return cached
        if self.offline:
            raise RuntimeError(
                f"{request.consumer_id}: no mesh yet; offline mode does not generate"
            )
        return self._import_generated(request, digest)

    def _cached(self, request: AssetRequest, digest: str) -> ResolvedLocation | None:
        directory = self._location_dir(request.location_id)
        meta_path = directory / "location.json"
        glb = directory / "model.glb"
        if not meta_path.is_file() or not glb.is_file():
            return None
        record = json.loads(meta_path.read_text(encoding="utf-8"))
        if record.get("descriptionHash") != digest:
            return None
        if record.get("triangleBudget") != self.triangle_budget:
            return None
        if record.get("decimator") != DECIMATOR or record.get("fit") != "uniform":
            return None
        size = record.get("sizeMeters") or list(request.size)
        return ResolvedLocation(
            location_id=request.location_id,
            glb=glb,
            source=str(record.get("source") or GENERATED_SOURCE),
            source_id=str(record.get("sourceId") or ""),
            title=str(record.get("title") or request.appearance),
            size=(float(size[0]), float(size[1]), float(size[2])),
        )

    def _import_generated(self, request: AssetRequest, digest: str) -> ResolvedLocation:
        directory = self._location_dir(request.location_id)
        directory.mkdir(parents=True, exist_ok=True)
        plate_copy = directory / "plate.png"
        shutil.copyfile(self._plate_path(request.location_id), plate_copy)
        try:
            fetched = materialize_mesh(
                self.generator(
                    request.appearance,
                    directory,
                    triangle_budget=self.triangle_budget,
                )
            )
        except Exception as exc:
            raise RuntimeError(f"{request.consumer_id}: {exc}") from exc
        finally:
            if plate_copy.is_file():
                plate_copy.unlink()
        vertices, faces = read_schema_mesh(fetched)
        fitted = fit_to_size(vertices, request.size)
        source_id = appearance_source_id(request.appearance)
        write_schema_glb(directory / "model.glb", fitted, faces)
        if fetched != directory / "model.glb" and fetched.is_file():
            fetched.unlink()
        if self.write_thumbs:
            from spatial_previs import render_mesh_thumbnail

            render_mesh_thumbnail(fitted, faces, directory / "thumb.png")
        record = {
            "showId": self.show_id,
            "locationId": request.location_id,
            "space": "gltf-y-up",
            "sizeMeters": list(request.size),
            "descriptionHash": digest,
            "title": request.appearance,
            "source": GENERATED_SOURCE,
            "sourceId": source_id,
            "author": "Qwen-Image-Edit-2511, TRELLIS.2",
            "license": "MIT",
            "pageUrl": "https://github.com/microsoft/TRELLIS.2",
            "retrieved": date.today().isoformat(),
            "triangleCount": int(len(faces)),
            "triangleBudget": self.triangle_budget,
            "decimator": DECIMATOR,
            "fit": "uniform",
            "model": "model.glb",
        }
        (directory / "location.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        return ResolvedLocation(
            location_id=request.location_id,
            glb=directory / "model.glb",
            source=GENERATED_SOURCE,
            source_id=source_id,
            title=request.appearance,
            size=request.size,
        )

    def _plate_path(self, location_id: str) -> Path:
        return self.output_dir / "plates" / self.show_id / location_id / "plate.png"

    def _location_dir(self, location_id: str) -> Path:
        return self.output_dir / "assets" / self.show_id / location_id


def location_scene_description(location_id: str, location: dict) -> str:
    """One Trellis subject: the location text, as an open place with no people."""
    spatial = location.get("spatial") or {}
    size = _vec3(spatial.get("sizeMeters"), (8.0, 10.0, 4.0))
    place = (_appearance(location.get("promptBlock")) or location_id.replace("_", " ")).rstrip(".")
    return " ".join(
        (
            f"A wide open set of {place}.",
            "Most of the ground is bare floor, with wide empty space between the features.",
            "The features are few, small beside the place, and spaced far apart.",
            "The floor reaches the edges of the model. Not a crowded diorama and not a boxed platform.",
            "No human figures, figurines, statues, animals, or readable text.",
            f"The whole place is about {size[0]:g} meters wide, {size[1]:g} meters deep, and {size[2]:g} meters tall.",
        )
    )


def write_location_plates(
    show: dict,
    writer,
    *,
    output_dir: Path = OUTPUT_DIR,
    refresh: bool = False,
    offline: bool = False,
) -> list[Path]:
    """Draw each location plate and leave the mesh for a later step.

    A redrawn plate drops the mesh that was built from the previous picture.
    """
    show_id = str(show.get("id") or "")
    plates: list[Path] = []
    for request in collect_requests(show):
        appearance = request.appearance
        digest = asset_hash(GENERATED_SOURCE, appearance_source_id(appearance), request.size)
        plate = output_dir / "plates" / show_id / request.location_id / "plate.png"
        meta_path = plate.with_name("plate.json")
        stored = _read_json(meta_path, {})
        if plate_is_ready(plate) and stored.get("descriptionHash") == digest and not refresh:
            print(f"  {request.label}: {plate}", flush=True)
            plates.append(plate)
            continue
        if offline:
            raise RuntimeError(
                f"{request.consumer_id}: no plate at {plate}; offline mode does not draw one"
            )
        writer(appearance, plate.parent)
        _discard_location_mesh(output_dir, show_id, request.location_id)
        meta_path.write_text(
            json.dumps({"locationId": request.location_id, "descriptionHash": digest}, indent=2),
            encoding="utf-8",
        )
        print(f"  {request.label}: {plate}", flush=True)
        plates.append(plate)
    return plates


def _discard_location_mesh(output_dir: Path, show_id: str, location_id: str) -> None:
    directory = output_dir / "assets" / show_id / location_id
    for name in ("model.glb", "location.json", "thumb.png"):
        path = directory / name
        if path.is_file():
            path.unlink()


def collect_requests(show: dict) -> list[AssetRequest]:
    requests: list[AssetRequest] = []
    for location_id, location in (show.get("locations") or {}).items():
        if not isinstance(location, dict):
            continue
        spatial = location.get("spatial") or {}
        requests.append(
            AssetRequest(
                consumer_id=f"location:{location_id}",
                label=str(location_id),
                size=_vec3(spatial.get("sizeMeters"), (8.0, 10.0, 4.0)),
                location_id=str(location_id),
                appearance=location_scene_description(str(location_id), location),
            )
        )
    return requests


def appearance_source_id(appearance: str) -> str:
    text = " ".join(appearance.split())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def asset_hash(source: str, source_id: str, size: tuple[float, float, float]) -> str:
    payload = {
        "source": source.strip().lower(),
        "sourceId": source_id.strip(),
        "sizeMeters": [round(float(value), 4) for value in size],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def export_credits(output_dir: Path, destination: Path) -> None:
    lines = [
        "# Credits",
        "",
        "Location meshes are generated for one show. Qwen-Image-Edit-2511 draws the place",
        "and TRELLIS.2 turns that picture into one mesh. Both models are used under",
        "their published licenses (Qwen Apache-2.0, TRELLIS.2 MIT).",
        "",
    ]
    records = sorted((output_dir / "assets").glob("*/*/location.json")) if (output_dir / "assets").is_dir() else []
    if not records:
        lines.append("No generated meshes have been recorded.")
    else:
        for path in records:
            record = json.loads(path.read_text(encoding="utf-8"))
            show_id = record.get("showId") or path.parents[1].name
            location_id = record.get("locationId") or path.parent.name
            lines.append(
                f"- {show_id}/{location_id} by Qwen-Image-Edit-2511, TRELLIS.2 (MIT). "
                "Source: trellis2. https://github.com/microsoft/TRELLIS.2"
            )
    lines.append("")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")


def _appearance(raw) -> str | None:
    if not isinstance(raw, str):
        return None
    text = " ".join(raw.split())
    return text or None


def _vec3(raw, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if not isinstance(raw, list) or len(raw) != 3:
        return default
    return (float(raw[0]), float(raw[1]), float(raw[2]))


def _read_json(path: Path, default: dict) -> dict:
    if not path.is_file():
        return json.loads(json.dumps(default))
    return json.loads(path.read_text(encoding="utf-8"))
