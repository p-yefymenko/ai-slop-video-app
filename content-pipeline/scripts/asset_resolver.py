"""Turn a location into plates and meshes.

A location with no people is one plate and one mesh of the whole place.
A location with people is one plate and one mesh per landmark, fitted to the
size written on that landmark. Landmarks with the same appearance and size
share that picture and mesh, then each one is placed at its own position.
Characters stay out of those meshes. Each character is their own plate and
mesh, fitted to their standing height, the same way a landmark is built.
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
from coords import schema_to_gltf
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
    landmark_id: str | None = None
    character_id: str | None = None
    position: tuple[float, float, float] | None = None


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
        shared: dict[str, tuple[ResolvedLocation, str]] = {}
        for request in [*collect_requests(show), *collect_character_requests(show)]:
            digest = description_hash(request) if request.landmark_id else None
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

    def resolve_one(self, request: AssetRequest) -> ResolvedLocation:
        plate = self._plate_path(request)
        if not plate_is_ready(plate):
            raise RuntimeError(
                f"{request.consumer_id}: no reviewed plate at {plate}. "
                "Run `pnpm run content:plates`."
            )
        plate_record = _read_json(plate.with_name("plate.json"), {})
        digest = description_hash(request)
        if plate_record.get("descriptionHash") not in (None, digest):
            raise RuntimeError(
                f"{request.consumer_id}: the plate does not match the current text. "
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
        directory = self._asset_dir(request)
        meta_path = directory / self._record_name(request)
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
        directory = self._asset_dir(request)
        directory.mkdir(parents=True, exist_ok=True)
        plate_copy = directory / "plate.png"
        shutil.copyfile(self._plate_path(request), plate_copy)
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
        if request.character_id:
            record["characterId"] = request.character_id
        elif request.landmark_id:
            record["landmarkId"] = request.landmark_id
            if request.position is not None:
                record["schemaPosition"] = list(request.position)
                record["position"] = list(schema_to_gltf(request.position))
        (directory / self._record_name(request)).write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        return ResolvedLocation(
            location_id=request.location_id,
            glb=directory / "model.glb",
            source=GENERATED_SOURCE,
            source_id=source_id,
            title=request.appearance,
            size=request.size,
        )

    def _place_shared_mesh(self, request: AssetRequest, source: ResolvedLocation) -> ResolvedLocation:
        """Copy a mesh that was already built for the same appearance and size."""
        directory = self._asset_dir(request)
        directory.mkdir(parents=True, exist_ok=True)
        dest = directory / "model.glb"
        if dest.resolve() != source.glb.resolve():
            shutil.copyfile(source.glb, dest)
        record_path = source.glb.parent / "landmark.json"
        record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else {}
        record.update(
            {
                "showId": self.show_id,
                "locationId": request.location_id,
                "landmarkId": request.landmark_id,
                "space": "gltf-y-up",
                "sizeMeters": list(request.size),
                "descriptionHash": description_hash(request),
                "title": request.appearance,
                "source": source.source,
                "sourceId": source.source_id,
                "model": "model.glb",
            }
        )
        if request.position is not None:
            record["schemaPosition"] = list(request.position)
            record["position"] = list(schema_to_gltf(request.position))
        (directory / "landmark.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        return ResolvedLocation(
            location_id=request.location_id,
            glb=dest,
            source=source.source,
            source_id=source.source_id,
            title=request.appearance,
            size=request.size,
        )

    def _plate_path(self, request: AssetRequest) -> Path:
        return self._asset_dir(request, "plates") / "plate.png"

    def _asset_dir(self, request: AssetRequest, stage: str = "assets") -> Path:
        if request.character_id:
            return self.output_dir / stage / self.show_id / "characters" / request.character_id
        directory = self.output_dir / stage / self.show_id / request.location_id
        if request.landmark_id:
            return directory / request.landmark_id
        return directory

    def _record_name(self, request: AssetRequest) -> str:
        if request.character_id:
            return "character.json"
        return "landmark.json" if request.landmark_id else "location.json"


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
    Landmarks with the same appearance and size share one picture.
    """
    show_id = str(show.get("id") or "")
    plates: list[Path] = []
    shared: dict[str, tuple[Path, str]] = {}
    for request in [*collect_requests(show), *collect_character_requests(show)]:
        appearance = request.appearance
        digest = description_hash(request)
        if request.character_id:
            plate = output_dir / "plates" / show_id / "characters" / request.character_id / "plate.png"
        else:
            plate = output_dir / "plates" / show_id / request.location_id
            if request.landmark_id:
                plate = plate / request.landmark_id
            plate = plate / "plate.png"
        meta_path = plate.with_name("plate.json")
        stored = _read_json(meta_path, {})
        source = shared.get(digest) if request.landmark_id else None
        if source is not None:
            source_plate, source_label = source
            reused = plate_is_ready(plate) and stored.get("descriptionHash") == digest and not refresh
            if not reused:
                _copy_plate(source_plate, plate, request, digest)
                _discard_mesh(output_dir, show_id, request)
            print(f"  {request.label}: reuses {source_label}", flush=True)
            plates.append(plate)
            continue
        if plate_is_ready(plate) and stored.get("descriptionHash") == digest and not refresh:
            print(f"  {request.label}: {plate}", flush=True)
            plates.append(plate)
            if request.landmark_id:
                shared[digest] = (plate, request.label)
            continue
        if offline:
            raise RuntimeError(
                f"{request.consumer_id}: no plate at {plate}; offline mode does not draw one"
            )
        if request.character_id:
            writer(appearance, plate.parent, character=True)
        elif request.landmark_id:
            writer(appearance, plate.parent, landmark=True)
        else:
            writer(appearance, plate.parent)
        _discard_mesh(output_dir, show_id, request)
        meta = {"locationId": request.location_id, "descriptionHash": digest}
        if request.character_id:
            meta["characterId"] = request.character_id
        elif request.landmark_id:
            meta["landmarkId"] = request.landmark_id
            shared[digest] = (plate, request.label)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"  {request.label}: {plate}", flush=True)
        plates.append(plate)
    return plates


def _copy_plate(source: Path, destination: Path, request: AssetRequest, digest: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    meta = {
        "locationId": request.location_id,
        "descriptionHash": digest,
        "landmarkId": request.landmark_id,
    }
    destination.with_name("plate.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _discard_mesh(output_dir: Path, show_id: str, request: AssetRequest) -> None:
    if request.character_id:
        directory = output_dir / "assets" / show_id / "characters" / request.character_id
        record_name = "character.json"
    else:
        directory = output_dir / "assets" / show_id / request.location_id
        if request.landmark_id:
            directory = directory / request.landmark_id
        record_name = "landmark.json" if request.landmark_id else "location.json"
    names = ("model.glb", "thumb.png", record_name)
    for name in names:
        path = directory / name
        if path.is_file():
            path.unlink()


def location_has_people(show: dict, location_id: str) -> bool:
    """True when a person stands in this location or a scene shows people there."""
    for episode in show.get("episodes") or []:
        if not isinstance(episode, dict):
            continue
        for scene in episode.get("scenes") or []:
            if scene.get("locationId") == location_id and scene.get("characterIds"):
                return True
        tracks = ((episode.get("spatialTimeline") or {}).get("characterTracks")) or {}
        for track in tracks.values():
            for frame in track or []:
                if isinstance(frame, dict) and frame.get("locationId") == location_id:
                    return True
    return False


def collect_requests(show: dict) -> list[AssetRequest]:
    requests: list[AssetRequest] = []
    for location_id, location in (show.get("locations") or {}).items():
        if not isinstance(location, dict):
            continue
        spatial = location.get("spatial") or {}
        if location_has_people(show, str(location_id)):
            requests.extend(_landmark_requests(str(location_id), spatial))
            continue
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


def _landmark_requests(location_id: str, spatial: dict) -> list[AssetRequest]:
    requests: list[AssetRequest] = []
    for landmark_id, landmark in (spatial.get("landmarks") or {}).items():
        if not isinstance(landmark, dict):
            continue
        appearance = _appearance(landmark.get("appearance"))
        size = landmark.get("size")
        position = landmark.get("position")
        if not appearance or not isinstance(size, list) or len(size) != 3:
            raise RuntimeError(
                f"landmark:{location_id}:{landmark_id} needs position, size, and appearance"
            )
        if not isinstance(position, list) or len(position) != 3:
            raise RuntimeError(
                f"landmark:{location_id}:{landmark_id} needs position, size, and appearance"
            )
        requests.append(
            AssetRequest(
                consumer_id=f"landmark:{location_id}:{landmark_id}",
                label=f"{location_id}/{landmark_id}",
                size=(float(size[0]), float(size[1]), float(size[2])),
                location_id=location_id,
                appearance=appearance,
                landmark_id=str(landmark_id),
                position=(float(position[0]), float(position[1]), float(position[2])),
            )
        )
    return requests


def collect_character_requests(show: dict) -> list[AssetRequest]:
    """One standing mesh per character, fitted to their height the way a landmark is."""
    requests: list[AssetRequest] = []
    for character_id, character in (show.get("characters") or {}).items():
        if not isinstance(character, dict):
            continue
        appearance = _appearance(character.get("promptBlock"))
        if not appearance:
            continue
        height = 1.72
        proxy = character.get("proxy") or {}
        try:
            parsed = float(proxy.get("heightMeters") or height)
        except (TypeError, ValueError):
            parsed = height
        if parsed > 0:
            height = parsed
        requests.append(
            AssetRequest(
                consumer_id=f"character:{character_id}",
                label=f"characters/{character_id}",
                size=(height, height, height),
                location_id="characters",
                appearance=appearance,
                character_id=str(character_id),
            )
        )
    return requests


def description_hash(request: AssetRequest) -> str:
    """Identity of the picture. A landmark or character includes its plate prompt."""
    text = request.appearance
    if request.character_id or request.landmark_id:
        from asset_generate import plate_prompt

        text = plate_prompt(
            request.appearance,
            landmark=bool(request.landmark_id),
            character=bool(request.character_id),
        )
    return asset_hash(GENERATED_SOURCE, appearance_source_id(text), request.size)


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
        "Location meshes are generated for one show. An empty location is one mesh of the",
        "place. A location with people is one mesh per landmark, fitted to the size in",
        "the script. Qwen-Image-Edit-2511 draws the picture and TRELLIS.2 meshes it.",
        "Both models are used under their published licenses (Qwen Apache-2.0, TRELLIS.2 MIT).",
        "",
    ]
    roots = []
    assets = output_dir / "assets"
    if assets.is_dir():
        roots.extend(assets.glob("*/*/location.json"))
        roots.extend(assets.glob("*/*/*/landmark.json"))
    records = sorted(roots)
    if not records:
        lines.append("No generated meshes have been recorded.")
    else:
        for path in records:
            record = json.loads(path.read_text(encoding="utf-8"))
            landmark_id = record.get("landmarkId")
            if landmark_id:
                show_id = record.get("showId") or path.parents[2].name
                location_id = record.get("locationId") or path.parents[1].name
                name = f"{show_id}/{location_id}/{landmark_id}"
            else:
                show_id = record.get("showId") or path.parents[1].name
                location_id = record.get("locationId") or path.parent.name
                name = f"{show_id}/{location_id}"
            lines.append(
                f"- {name} by Qwen-Image-Edit-2511, TRELLIS.2 (MIT). "
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
