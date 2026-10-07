"""Turn every entity of a show into a plate and a mesh.

A landmark or a prop is one isolated object, drawn from its appearance and
fitted into its size. Two with the same appearance and size share one picture
and mesh. A character is drawn wearing ``body`` plus every attribute and fitted
to their standing height. Every mesh keeps the plate's color on its vertices.
A character is Pixal3D, pixel-aligned to its frontal plate, front turned to
schema +Y; an object is TRELLIS.2 in its own upright frame, textured.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from asset_generate import generate_asset_mesh, plate_is_ready, plate_prompt
from asset_sources import materialize_mesh
from coords import schema_to_gltf
from mesh_io import (
    DECIMATOR,
    TRIANGLE_BUDGET,
    cap_holes,
    drop_thin_side_protrusions,
    face_schema_forward,
    fit_to_size,
    read_schema_mesh,
    read_vertex_colors,
    require_triangle_budget,
    standing_proportion_warning,
    write_schema_glb,
)
from pipeline_paths import OUTPUT_DIR
from world import character_appearance_text

GENERATED_SOURCE = "trellis2"
CHARACTER_MODEL = "pixal3d"
OBJECT_MODEL = "trellis2-textured"
FACING = "schema-plus-y"
PIXAL3D_PAGE = "https://github.com/TencentARC/Pixal3D"
# Plate color sampled by Pixal3D and stored on each vertex.
SURFACE = "vertex-color"
# Character meshes cap small openings with new triangles. Objects keep theirs.
CHARACTER_MESH_REPAIR = "cap-holes"


@dataclass
class AssetRequest:
    consumer_id: str
    label: str
    size: tuple[float, float, float]
    location_id: str
    appearance: str
    landmark_id: str | None = None
    character_id: str | None = None
    prop_id: str | None = None
    position: tuple[float, float, float] | None = None

    @property
    def shared(self) -> bool:
        """Landmarks and props with the same appearance and size share one mesh."""
        return not self.character_id

    @property
    def record_name(self) -> str:
        if self.character_id:
            return "character.json"
        return "prop.json" if self.prop_id else "landmark.json"

    def directory(self, output_dir: Path, stage: str, show_id: str) -> Path:
        root = output_dir / stage / show_id
        if self.character_id:
            return root / "characters" / self.character_id
        if self.prop_id:
            return root / "props" / self.prop_id
        return root / self.location_id / str(self.landmark_id)

    def identity(self) -> dict:
        if self.character_id:
            return {"characterId": self.character_id}
        if self.prop_id:
            return {"propId": self.prop_id}
        return {
            "landmarkId": self.landmark_id,
            "schemaPosition": list(self.position),
            "position": list(schema_to_gltf(self.position)),
        }


@dataclass
class ResolvedAsset:
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

    def resolve_show(self, show: dict) -> dict[str, ResolvedAsset]:
        resolved: dict[str, ResolvedAsset] = {}
        shared: dict[str, tuple[ResolvedAsset, str]] = {}
        for request in collect_requests(show):
            digest = description_hash(request) if request.shared else None
            if digest and digest in shared:
                source, label = shared[digest]
                print(f"  {request.label}: reuses {label}", flush=True)
                resolved[request.consumer_id] = self._place_shared_mesh(request, source)
                continue
            item = self.resolve_one(request)
            resolved[request.consumer_id] = item
            if digest:
                shared[digest] = (item, request.label)
        return resolved

    def resolve_one(self, request: AssetRequest) -> ResolvedAsset:
        plate = self._dir(request, "plates") / "plate.png"
        if not plate_is_ready(plate):
            raise RuntimeError(
                f"{request.consumer_id}: no reviewed plate at {plate}. Run `pnpm run content:plates`."
            )
        digest = description_hash(request)
        if _read_json(plate.with_name("plate.json"), {}).get("descriptionHash") not in (None, digest):
            raise RuntimeError(
                f"{request.consumer_id}: the plate does not match the current text. Run `pnpm run content:plates`."
            )
        cached = self._cached(request, digest)
        if cached is not None and not self.refresh:
            return cached
        if self.offline:
            raise RuntimeError(f"{request.consumer_id}: no mesh yet; offline mode does not generate")
        return self._import_generated(request, digest)

    def _dir(self, request: AssetRequest, stage: str = "assets") -> Path:
        return request.directory(self.output_dir, stage, self.show_id)

    def _cached(self, request: AssetRequest, digest: str) -> ResolvedAsset | None:
        directory = self._dir(request)
        meta_path = directory / request.record_name
        glb = directory / "model.glb"
        if not meta_path.is_file() or not glb.is_file():
            return None
        record = json.loads(meta_path.read_text(encoding="utf-8"))
        if record.get("descriptionHash") != digest or record.get("triangleBudget") != self.triangle_budget:
            return None
        if record.get("decimator") != DECIMATOR or record.get("fit") != "uniform":
            return None
        if (record.get("meshModel"), record.get("surface")) != (_model(request), SURFACE):
            return None
        if request.character_id and record.get("meshRepair") != CHARACTER_MESH_REPAIR:
            return None
        size = record.get("sizeMeters") or list(request.size)
        return ResolvedAsset(
            location_id=request.location_id,
            glb=glb,
            source=str(record.get("source") or GENERATED_SOURCE),
            source_id=str(record.get("sourceId") or ""),
            title=str(record.get("title") or request.appearance),
            size=(float(size[0]), float(size[1]), float(size[2])),
        )

    def _import_generated(self, request: AssetRequest, digest: str) -> ResolvedAsset:
        directory = self._dir(request)
        directory.mkdir(parents=True, exist_ok=True)
        plate_copy = directory / "plate.png"
        shutil.copyfile(self._dir(request, "plates") / "plate.png", plate_copy)
        try:
            fetched = materialize_mesh(
                self.generator(
                    request.appearance,
                    directory,
                    triangle_budget=self.triangle_budget,
                    character=bool(request.character_id),
                )
            )
        except Exception as exc:
            raise RuntimeError(f"{request.consumer_id}: {exc}") from exc
        finally:
            if plate_copy.is_file():
                plate_copy.unlink()
        vertices, faces = read_schema_mesh(fetched)
        colors = read_vertex_colors(fetched)
        if colors is not None and len(colors) != len(vertices):
            colors = None
        if request.character_id:
            vertices = face_schema_forward(vertices)
            write_schema_glb(directory / "model.precleanup.glb", vertices, faces, colors)
            vertices, faces, colors, lost_vertices, lost_triangles = drop_thin_side_protrusions(
                vertices, faces, colors
            )
            print(
                f"  {request.character_id}: removed {lost_vertices} vertices and {lost_triangles} triangles",
                flush=True,
            )
        fitted = fit_to_size(vertices, request.size)
        if request.character_id:
            fitted, faces = cap_holes(fitted, faces)
            warning = standing_proportion_warning(request.character_id, fitted)
            if warning:
                self.warnings.append(warning)
        source_id = appearance_source_id(request.appearance)
        write_schema_glb(directory / "model.glb", fitted, faces, colors)
        if fetched != directory / "model.glb" and fetched.is_file():
            fetched.unlink()
        if self.write_thumbs:
            from render import thumbnail

            thumbnail(fitted, faces, directory / "thumb.png")
        character = bool(request.character_id)
        source = _model(request)
        record = {
            "showId": self.show_id,
            "locationId": request.location_id,
            "space": "gltf-y-up",
            "sizeMeters": list(request.size),
            "descriptionHash": digest,
            "title": request.appearance,
            "source": source,
            "sourceId": source_id,
            "author": "Qwen-Image-Edit-2511, Pixal3D" if character else "Qwen-Image-Edit-2511, TRELLIS.2",
            "license": "MIT",
            "pageUrl": PIXAL3D_PAGE if character else "https://github.com/microsoft/TRELLIS.2",
            "meshModel": _model(request),
            "surface": SURFACE,
            "retrieved": date.today().isoformat(),
            "triangleCount": int(len(faces)),
            "triangleBudget": self.triangle_budget,
            "decimator": DECIMATOR,
            "fit": "uniform",
            "model": "model.glb",
            **request.identity(),
        }
        if character:
            record["meshRepair"] = CHARACTER_MESH_REPAIR
            record["facing"] = FACING
        (directory / request.record_name).write_text(json.dumps(record, indent=2), encoding="utf-8")
        return ResolvedAsset(request.location_id, directory / "model.glb", source, source_id, request.appearance, request.size)

    def _place_shared_mesh(self, request: AssetRequest, source: ResolvedAsset) -> ResolvedAsset:
        """Copy a mesh already built for the same appearance and size."""
        directory = self._dir(request)
        directory.mkdir(parents=True, exist_ok=True)
        dest = directory / "model.glb"
        if dest.resolve() != source.glb.resolve():
            shutil.copyfile(source.glb, dest)
        records = [path for path in source.glb.parent.glob("*.json") if path.name in ("landmark.json", "prop.json")]
        record = json.loads(records[0].read_text(encoding="utf-8")) if records else {}
        for key in ("landmarkId", "propId", "schemaPosition", "position"):
            record.pop(key, None)
        record.update(
            {
                "showId": self.show_id,
                "locationId": request.location_id,
                "space": "gltf-y-up",
                "sizeMeters": list(request.size),
                "descriptionHash": description_hash(request),
                "title": request.appearance,
                "source": source.source,
                "sourceId": source.source_id,
                "model": "model.glb",
                **request.identity(),
            }
        )
        (directory / request.record_name).write_text(json.dumps(record, indent=2), encoding="utf-8")
        return ResolvedAsset(request.location_id, dest, source.source, source.source_id, request.appearance, request.size)


def _model(request: AssetRequest) -> str:
    return CHARACTER_MODEL if request.character_id else OBJECT_MODEL


def write_plates(
    show: dict,
    writer,
    *,
    output_dir: Path = OUTPUT_DIR,
    refresh: bool = False,
    offline: bool = False,
) -> list[Path]:
    """Draw each entity's plate and leave the mesh for a later step.

    A redrawn plate drops the mesh that was built from the previous picture.
    Landmarks and props with the same appearance and size share one picture.
    """
    show_id = str(show["id"])
    plates: list[Path] = []
    shared: dict[str, tuple[Path, str]] = {}
    for request in collect_requests(show):
        digest = description_hash(request)
        plate = request.directory(output_dir, "plates", show_id) / "plate.png"
        meta_path = plate.with_name("plate.json")
        current = plate_is_ready(plate) and _read_json(meta_path, {}).get("descriptionHash") == digest and not refresh
        source = shared.get(digest) if request.shared else None
        if source is not None:
            if not current:
                plate.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source[0], plate)
                meta_path.write_text(
                    json.dumps({"locationId": request.location_id, "descriptionHash": digest, **_ids(request)}, indent=2),
                    encoding="utf-8",
                )
                _discard_mesh(output_dir, show_id, request)
            print(f"  {request.label}: reuses {source[1]}", flush=True)
            plates.append(plate)
            continue
        if not current:
            if offline:
                raise RuntimeError(f"{request.consumer_id}: no plate at {plate}; offline mode does not draw one")
            writer(request.appearance, plate.parent, character=bool(request.character_id))
            _discard_mesh(output_dir, show_id, request)
            meta_path.write_text(
                json.dumps({"locationId": request.location_id, "descriptionHash": digest, **_ids(request)}, indent=2),
                encoding="utf-8",
            )
        print(f"  {request.label}: {plate}", flush=True)
        plates.append(plate)
        if request.shared:
            shared[digest] = (plate, request.label)
    return plates


def _ids(request: AssetRequest) -> dict:
    return {key: value for key, value in request.identity().items() if key.endswith("Id")}


def _discard_mesh(output_dir: Path, show_id: str, request: AssetRequest) -> None:
    directory = request.directory(output_dir, "assets", show_id)
    for name in ("model.glb", "thumb.png", request.record_name):
        path = directory / name
        if path.is_file():
            path.unlink()


def collect_requests(show: dict) -> list[AssetRequest]:
    """Every landmark of every location, every prop, then every character."""
    requests: list[AssetRequest] = []
    for location_id, location in show["locations"].items():
        for landmark_id, landmark in location["spatial"]["landmarks"].items():
            requests.append(
                AssetRequest(
                    consumer_id=f"landmark:{location_id}:{landmark_id}",
                    label=f"{location_id}/{landmark_id}",
                    size=_vec3(landmark["size"]),
                    location_id=location_id,
                    appearance=" ".join(landmark["appearance"].split()),
                    landmark_id=landmark_id,
                    position=_vec3(landmark["position"]),
                )
            )
    for prop_id, prop in (show.get("props") or {}).items():
        requests.append(
            AssetRequest(
                consumer_id=f"prop:{prop_id}",
                label=f"props/{prop_id}",
                size=_vec3(prop["size"]),
                location_id="props",
                appearance=" ".join(prop["appearance"].split()),
                prop_id=prop_id,
            )
        )
    for character_id, character in show["characters"].items():
        height = float(character["heightMeters"])
        requests.append(
            AssetRequest(
                consumer_id=f"character:{character_id}",
                label=f"characters/{character_id}",
                size=(height, height, height),
                location_id="characters",
                appearance=character_appearance_text(character),
                character_id=character_id,
            )
        )
    return requests


def description_hash(request: AssetRequest) -> str:
    """Identity of the picture: the full plate prompt and the size."""
    text = plate_prompt(request.appearance, character=bool(request.character_id))
    return asset_hash(GENERATED_SOURCE, appearance_source_id(text), request.size)


def appearance_source_id(appearance: str) -> str:
    return hashlib.sha256(" ".join(appearance.split()).encode("utf-8")).hexdigest()


def asset_hash(source: str, source_id: str, size: tuple[float, float, float]) -> str:
    payload = {
        "source": source.strip().lower(),
        "sourceId": source_id.strip(),
        "sizeMeters": [round(float(value), 4) for value in size],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def export_credits(output_dir: Path, destination: Path) -> None:
    lines = [
        "# Credits",
        "",
        "Meshes are generated for one show: one per landmark, prop, and character, fitted",
        "to the size in the script, with the plate's color. Qwen-Image-Edit-2511 draws the",
        "picture. TRELLIS.2 meshes and textures an object; Pixal3D meshes a character.",
        "Qwen is Apache-2.0. TRELLIS.2 and Pixal3D are MIT.",
        "",
    ]
    assets = output_dir / "assets"
    records = sorted(
        [*assets.glob("*/*/*/landmark.json"), *assets.glob("*/*/*/prop.json"), *assets.glob("*/*/*/character.json")]
    ) if assets.is_dir() else []
    if not records:
        lines.append("No generated meshes have been recorded.")
    for path in records:
        name = f"{path.parents[2].name}/{path.parents[1].name}/{path.parent.name}"
        if path.name == "character.json":
            lines.append(f"- {name} by Qwen-Image-Edit-2511, Pixal3D (MIT). {PIXAL3D_PAGE}")
        else:
            lines.append(f"- {name} by Qwen-Image-Edit-2511, TRELLIS.2 (MIT). https://github.com/microsoft/TRELLIS.2")
    lines.append("")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")


def _vec3(raw) -> tuple[float, float, float]:
    return (float(raw[0]), float(raw[1]), float(raw[2]))


def _read_json(path: Path, default: dict) -> dict:
    if not path.is_file():
        return dict(default)
    return json.loads(path.read_text(encoding="utf-8"))
