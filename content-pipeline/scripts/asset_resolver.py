"""Turn each landmark and prop appearance into a prefab, then build one set per location.

Qwen draws the object. TRELLIS.2 turns that picture into a mesh. Characters stay
out of the set. Their clay mannequins are built from ``ShowCharacter.proxy`` at
previs time.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from asset_generate import generate_asset_mesh
from asset_sources import materialize_mesh
from mesh_io import (
    DECIMATOR,
    TRIANGLE_BUDGET,
    fit_to_size,
    read_schema_mesh,
    schema_triangles,
    write_schema_glb,
)
from pipeline_paths import LIBRARY_DIR, OUTPUT_DIR, SHOWS_DIR

GENERATED_SOURCE = "trellis2"

@dataclass
class AssetRequest:
    consumer_id: str
    label: str
    size: tuple[float, float, float]
    position: tuple[float, float, float] | None
    location_id: str | None
    appearance: str | None
    prefab_id: str | None


@dataclass
class ResolvedPrefab:
    prefab_id: str
    glb: Path
    origin: str
    source: str
    source_id: str
    title: str
    author: str
    license: str
    page_url: str
    size: tuple[float, float, float]


class AssetResolver:
    def __init__(
        self,
        show_id: str,
        *,
        library_dir: Path = LIBRARY_DIR,
        shows_dir: Path = SHOWS_DIR,
        output_dir: Path = OUTPUT_DIR,
        generator=None,
        offline: bool = False,
        refresh: bool = False,
        write_thumbs: bool = False,
    ) -> None:
        self.show_id = show_id
        self.library_dir = library_dir
        self.shows_dir = shows_dir
        self.output_dir = output_dir
        self.generator = generator if generator is not None else generate_asset_mesh
        self.offline = offline
        self.refresh = refresh
        self.write_thumbs = write_thumbs
        self.warnings: list[str] = []
        self.lock = _read_json(library_dir / "lock.json", {"version": 1, "needs": {}})
        self.sources_catalog = _read_json(library_dir / "sources.json", {"assets": []})
        self.lock.setdefault("needs", {})
        self.sources_catalog.setdefault("assets", [])

    def resolve_show(self, show: dict) -> dict[str, ResolvedPrefab]:
        resolved: dict[str, ResolvedPrefab] = {}
        for request in collect_requests(show):
            resolved[request.consumer_id] = self.resolve_one(request)
        self._write_json(self.library_dir / "lock.json", self.lock)
        self._write_json(self.library_dir / "sources.json", self.sources_catalog)
        self._write_resolved_index(resolved)
        self._build_sets(show, resolved)
        return resolved

    def resolve_one(self, request: AssetRequest) -> ResolvedPrefab:
        if request.prefab_id:
            pinned = self._find_prefab(request.prefab_id)
            if pinned is not None:
                return pinned
            self.warnings.append(
                f"{request.consumer_id}: pinned prefab {request.prefab_id} is missing"
            )
        appearance = " ".join((request.appearance or "").split())
        if not appearance:
            raise RuntimeError(f"{request.consumer_id}: appearance is required")
        source_id = appearance_source_id(appearance)
        digest = asset_hash(GENERATED_SOURCE, source_id, request.size)
        cached = self._cached(digest)
        if cached is not None and not self.refresh:
            return self._ensure_uniform_fit(request, cached, appearance, source_id, digest)
        if self.offline:
            raise RuntimeError(
                f"{request.consumer_id}: {appearance} is not in the library; offline mode does not generate"
            )
        return self._import_generated(request, appearance, source_id, digest)

    def _cached(self, digest: str) -> ResolvedPrefab | None:
        for root, origin in (
            (self._show_assets(), "show"),
            (self.library_dir / "prefabs", "library"),
        ):
            found = self._find_asset(root, digest, origin)
            if found is not None:
                return found
        entry = self.lock["needs"].get(digest)
        if not entry or self.refresh or entry.get("provisional"):
            return None
        return self._find_prefab(str(entry.get("prefabId") or ""))

    def _show_assets(self) -> Path:
        return self.shows_dir / self.show_id / "assets"

    def _import_generated(
        self,
        request: AssetRequest,
        appearance: str,
        source_id: str,
        digest: str,
    ) -> ResolvedPrefab:
        raw_dir = self.library_dir / "raw" / GENERATED_SOURCE / _raw_folder(source_id)
        try:
            fetched = materialize_mesh(self.generator(appearance, raw_dir))
        except Exception as exc:
            raise RuntimeError(f"{request.consumer_id}: {exc}") from exc
        vertices, faces = read_schema_mesh(fetched)
        # DecimateMesh already capped the triangle count. Scale uniformly into the box.
        fitted = fit_to_size(vertices, request.size)
        leaf = f"{source_id[:16]}-{_size_token(request.size)}"
        prefab_id = f"models/{leaf}"
        directory = self.library_dir / "prefabs" / "models" / leaf
        resolved = self._write_prefab(
            directory,
            prefab_id,
            fitted,
            faces,
            request,
            origin="generated",
            source=GENERATED_SOURCE,
            source_id=source_id,
            title=appearance,
            author="Qwen-Image-Edit-2511, TRELLIS.2",
            license_name="MIT",
            page_url="https://github.com/microsoft/TRELLIS.2",
            digest=digest,
            version="",
        )
        self._remember_source(resolved, "")
        self.lock["needs"][digest] = _lock_entry(request, resolved, "", provisional=False)
        return resolved

    def _write_prefab(
        self,
        directory: Path,
        prefab_id: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        request: AssetRequest,
        *,
        origin: str,
        source: str,
        source_id: str,
        title: str,
        author: str,
        license_name: str,
        page_url: str,
        digest: str | None,
        version: str,
    ) -> ResolvedPrefab:
        directory.mkdir(parents=True, exist_ok=True)
        write_schema_glb(directory / "model.glb", vertices, faces)
        if self.write_thumbs:
            from spatial_previs import render_mesh_thumbnail

            render_mesh_thumbnail(schema_triangles(vertices, faces), directory / "thumb.png")
        record = {
            "id": prefab_id,
            "assetHash": digest,
            "sizeMeters": list(request.size),
            "origin": origin,
            "source": source,
            "sourceId": source_id,
            "title": title,
            "author": author,
            "license": license_name,
            "pageUrl": page_url,
            "version": version,
            "retrieved": date.today().isoformat(),
            "triangleCount": int(len(faces)),
            "triangleBudget": TRIANGLE_BUDGET,
            "decimator": DECIMATOR,
            "fit": "uniform",
        }
        (directory / "prefab.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        return ResolvedPrefab(
            prefab_id=prefab_id,
            glb=directory / "model.glb",
            origin=origin,
            source=source,
            source_id=source_id,
            title=title,
            author=author,
            license=license_name,
            page_url=page_url,
            size=request.size,
        )

    def _find_prefab(self, prefab_id: str) -> ResolvedPrefab | None:
        if not prefab_id:
            return None
        for path, origin in self._prefab_files():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("id") == prefab_id:
                if self._needs_resimplify(path, record):
                    return None
                return self._read_prefab(path, record.get("origin") or origin)
        return None

    def _find_asset(self, root: Path, digest: str, origin: str) -> ResolvedPrefab | None:
        if not root.is_dir():
            return None
        for path in sorted(root.rglob("prefab.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("assetHash") == digest:
                if self._needs_resimplify(path, record):
                    return None
                return self._read_prefab(path, record.get("origin") or origin)
        return None

    def _prefab_files(self) -> list[tuple[Path, str]]:
        roots = (
            (self._show_assets(), "show"),
            (self.library_dir / "prefabs", "library"),
            (self.output_dir / self.show_id / "assets" / "fallback", "fallback"),
        )
        found: list[tuple[Path, str]] = []
        for root, origin in roots:
            if root.is_dir():
                found.extend((path, origin) for path in sorted(root.rglob("prefab.json")))
        return found

    def _read_prefab(self, path: Path, origin: str) -> ResolvedPrefab | None:
        self._cap_stored_mesh(path.parent)
        record = json.loads(path.read_text(encoding="utf-8"))
        glb = path.parent / "model.glb"
        if not glb.is_file():
            return None
        size = record.get("sizeMeters") or [1, 1, 1]
        return ResolvedPrefab(
            prefab_id=str(record.get("id") or path.parent.name),
            glb=glb,
            origin=str(record.get("origin") or origin),
            source=str(record.get("source") or origin),
            source_id=str(record.get("sourceId") or ""),
            title=str(record.get("title") or record.get("id") or path.parent.name),
            author=str(record.get("author") or ""),
            license=str(record.get("license") or ""),
            page_url=str(record.get("pageUrl") or ""),
            size=(float(size[0]), float(size[1]), float(size[2])),
        )

    def _ensure_uniform_fit(
        self,
        request: AssetRequest,
        cached: ResolvedPrefab,
        appearance: str,
        source_id: str,
        digest: str,
    ) -> ResolvedPrefab:
        """Rewrite a cached mesh that was stretched onto ``sizeMeters``."""
        if cached.source != GENERATED_SOURCE:
            return cached
        meta_path = cached.glb.parent / "prefab.json"
        record = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        if record.get("fit") == "uniform":
            return cached
        raw = self.library_dir / "raw" / GENERATED_SOURCE / _raw_folder(source_id) / "model.glb"
        if not raw.is_file():
            return cached
        vertices, faces = read_schema_mesh(raw)
        fitted = fit_to_size(vertices, request.size)
        return self._write_prefab(
            cached.glb.parent,
            cached.prefab_id,
            fitted,
            faces,
            request,
            origin=cached.origin,
            source=cached.source,
            source_id=source_id,
            title=appearance,
            author=cached.author or "Qwen-Image-Edit-2511, TRELLIS.2",
            license_name=cached.license or "MIT",
            page_url=cached.page_url or "https://github.com/microsoft/TRELLIS.2",
            digest=digest,
            version="",
        )

    def _needs_resimplify(self, path: Path, record: dict) -> bool:
        """A library mesh simplified under an older budget is rebuilt from the raw download."""
        try:
            path.resolve().relative_to((self.library_dir / "prefabs").resolve())
        except ValueError:
            return False
        if record.get("decimator") != DECIMATOR:
            return True
        return record.get("triangleBudget") != TRIANGLE_BUDGET

    def _cap_stored_mesh(self, directory: Path) -> None:
        glb = directory / "model.glb"
        meta = directory / "prefab.json"
        if not glb.is_file():
            return
        record = json.loads(meta.read_text(encoding="utf-8")) if meta.is_file() else {}
        recorded = record.get("triangleCount")
        if isinstance(recorded, int) and recorded <= TRIANGLE_BUDGET:
            return
        _vertices, faces = read_schema_mesh(glb)
        if len(faces) <= TRIANGLE_BUDGET and meta.is_file() and recorded != len(faces):
            record["triangleCount"] = int(len(faces))
            meta.write_text(json.dumps(record, indent=2), encoding="utf-8")

    def _remember_source(self, resolved: ResolvedPrefab, version: str) -> None:
        assets = self.sources_catalog["assets"]
        assets[:] = [item for item in assets if item.get("prefabId") != resolved.prefab_id]
        if resolved.origin == "fallback":
            return
        assets.append(
            {
                "prefabId": resolved.prefab_id,
                "source": resolved.source,
                "sourceId": resolved.source_id,
                "url": resolved.page_url,
                "author": resolved.author,
                "license": resolved.license,
                "version": version,
                "retrieved": date.today().isoformat(),
            }
        )

    def _write_resolved_index(self, resolved: dict[str, ResolvedPrefab]) -> None:
        payload = {
            "showId": self.show_id,
            "prefabs": {
                consumer_id: {
                    "prefabId": item.prefab_id,
                    "glb": str(item.glb),
                    "origin": item.origin,
                    "source": item.source,
                    "license": item.license,
                    "title": item.title,
                }
                for consumer_id, item in resolved.items()
            },
        }
        path = self.output_dir / self.show_id / "assets" / "resolved.json"
        self._write_json(path, payload)

    def _build_sets(self, show: dict, resolved: dict[str, ResolvedPrefab]) -> None:
        from coords import schema_to_gltf

        for location_id, location in (show.get("locations") or {}).items():
            spatial = location.get("spatial") or {}
            landmarks = spatial.get("landmarks") or {}
            instances = []
            placed_vertices: list[np.ndarray] = []
            placed_faces: list[np.ndarray] = []
            vertex_base = 0
            for landmark_id, landmark in landmarks.items():
                consumer_id = f"landmark:{location_id}/{landmark_id}"
                item = resolved[consumer_id]
                vertices, faces = read_schema_mesh(item.glb)
                position = tuple(float(value) for value in landmark["position"])
                placed_vertices.append(vertices + np.array(position))
                placed_faces.append(faces + vertex_base)
                vertex_base += len(vertices)
                gltf_position = schema_to_gltf(position)  # type: ignore[arg-type]
                instances.append(
                    {
                        "id": landmark_id,
                        "prefabId": item.prefab_id,
                        "position": list(gltf_position),
                        "schemaPosition": list(position),
                        "sizeMeters": list(item.size),
                        "origin": item.origin,
                        "source": item.source,
                        "license": item.license,
                        "title": item.title,
                    }
                )
            directory = self.output_dir / self.show_id / "sets" / location_id
            directory.mkdir(parents=True, exist_ok=True)
            if placed_vertices:
                write_schema_glb(
                    directory / "set.glb",
                    np.vstack(placed_vertices),
                    np.vstack(placed_faces),
                )
            size = spatial.get("sizeMeters") or [1, 1, 1]
            (directory / "set.json").write_text(
                json.dumps(
                    {
                        "locationId": location_id,
                        "space": "gltf-y-up",
                        "sizeMeters": size,
                        "instances": instances,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def collect_requests(show: dict) -> list[AssetRequest]:
    requests: list[AssetRequest] = []
    for location_id, location in (show.get("locations") or {}).items():
        landmarks = ((location.get("spatial") or {}).get("landmarks")) or {}
        for landmark_id, landmark in landmarks.items():
            requests.append(
                AssetRequest(
                    consumer_id=f"landmark:{location_id}/{landmark_id}",
                    label=landmark_id,
                    size=_vec3(landmark.get("size"), (1.0, 1.0, 1.0)),
                    position=_vec3(landmark.get("position"), (0.0, 0.0, 0.0)),
                    location_id=location_id,
                    appearance=_appearance(landmark.get("appearance")),
                    prefab_id=landmark.get("prefabId"),
                )
            )
    for prop_id, prop in (show.get("props") or {}).items():
        size = _vec3(prop.get("sizeMeters"), (0.3, 0.3, 0.3))
        requests.append(
            AssetRequest(
                consumer_id=f"prop:{prop_id}",
                label=prop_id,
                size=size,
                position=None,
                location_id=None,
                appearance=_appearance(prop.get("appearance")),
                prefab_id=prop.get("prefabId"),
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


def prune_unused_library(library_dir: Path, shows: list[dict]) -> list[str]:
    """Delete library prefabs that no show script pins or still describes."""
    prefab_root = library_dir / "prefabs"
    records: list[tuple[Path, dict]] = []
    if prefab_root.is_dir():
        for path in sorted(prefab_root.rglob("prefab.json")):
            records.append((path, json.loads(path.read_text(encoding="utf-8"))))
    by_hash = {
        str(record["assetHash"]): str(record.get("id") or "")
        for _path, record in records
        if record.get("assetHash") and record.get("id")
    }
    lock_path = library_dir / "lock.json"
    lock = _read_json(lock_path, {"version": 1, "needs": {}})
    needs = lock.get("needs") if isinstance(lock.get("needs"), dict) else {}
    kept: set[str] = set()
    for show in shows:
        for request in collect_requests(show):
            if request.prefab_id:
                kept.add(str(request.prefab_id))
            appearance = " ".join((request.appearance or "").split())
            if not appearance:
                continue
            digest = asset_hash(GENERATED_SOURCE, appearance_source_id(appearance), request.size)
            matched = by_hash.get(digest)
            if matched:
                kept.add(matched)
            pinned = str((needs.get(digest) or {}).get("prefabId") or "")
            if pinned:
                kept.add(pinned)
    removed: list[str] = []
    kept_raw: set[tuple[str, str]] = set()
    for path, record in records:
        prefab_id = str(record.get("id") or "")
        if prefab_id in kept:
            source = str(record.get("source") or "")
            source_id = str(record.get("sourceId") or "")
            if source and source_id:
                kept_raw.add((source, _raw_folder(source_id) if source == GENERATED_SOURCE else _slug(source_id)))
            continue
        shutil.rmtree(path.parent)
        if prefab_id:
            removed.append(prefab_id)
    if prefab_root.is_dir():
        for directory in sorted(prefab_root.rglob("*"), reverse=True):
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
    lock["needs"] = {
        digest: entry
        for digest, entry in needs.items()
        if str((entry or {}).get("prefabId") or "") in kept
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    catalog_path = library_dir / "sources.json"
    catalog = _read_json(catalog_path, {"assets": []})
    assets = catalog.get("assets") if isinstance(catalog.get("assets"), list) else []
    catalog["assets"] = [item for item in assets if str(item.get("prefabId") or "") in kept]
    catalog_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    raw_root = library_dir / "raw"
    if raw_root.is_dir():
        for source_dir in list(raw_root.iterdir()):
            if not source_dir.is_dir():
                continue
            for item in list(source_dir.iterdir()):
                if (source_dir.name, item.name) in kept_raw:
                    continue
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            if not any(source_dir.iterdir()):
                source_dir.rmdir()
    return removed


def apply_lock_override(lock_path: Path, digest: str, prefab_id: str) -> None:
    lock = _read_json(lock_path, {"version": 1, "needs": {}})
    entry = lock.setdefault("needs", {}).get(digest)
    if entry is None:
        raise SystemExit(f"No locked asset {digest}")
    entry["prefabId"] = prefab_id
    entry["origin"] = "override"
    entry["provisional"] = False
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")


def export_credits(catalog_path: Path, destination: Path) -> None:
    catalog = _read_json(catalog_path, {"assets": []})
    lines = [
        "# Credits",
        "",
        "Stage meshes are generated locally. Qwen-Image-Edit-2511 draws the object",
        "and TRELLIS.2 turns that picture into the mesh. Both models are used under",
        "their published licenses (Qwen Apache-2.0, TRELLIS.2 MIT).",
        "",
    ]
    assets = catalog.get("assets") or []
    if not assets:
        lines.append("No generated meshes have been recorded.")
    else:
        for asset in assets:
            author = asset.get("author") or "Unknown"
            license_name = asset.get("license") or "unknown"
            url = asset.get("url") or ""
            lines.append(
                f"- {asset.get('prefabId')} by {author} ({license_name}). Source: {asset.get('source')}. {url}"
            )
    lines.append("")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")


def _appearance(raw) -> str | None:
    if not isinstance(raw, str):
        return None
    text = " ".join(raw.split())
    return text or None


def _raw_folder(source_id: str) -> str:
    return source_id[:16]


def _size_token(size: tuple[float, float, float]) -> str:
    return "-".join(str(int(round(float(value) * 1000))) for value in size)


def _vec3(raw, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if not isinstance(raw, list) or len(raw) != 3:
        return default
    return (float(raw[0]), float(raw[1]), float(raw[2]))


def _slug(value: str) -> str:
    import re

    token = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return token[:80] or "asset"


def _lock_entry(request: AssetRequest, resolved: ResolvedPrefab, version: str, provisional: bool) -> dict:
    return {
        "appearance": request.appearance,
        "sizeMeters": list(request.size),
        "prefabId": resolved.prefab_id,
        "source": resolved.source,
        "sourceId": resolved.source_id,
        "version": version,
        "origin": resolved.origin,
        "provisional": provisional,
    }


def _read_json(path: Path, default: dict) -> dict:
    if not path.is_file():
        return json.loads(json.dumps(default))
    return json.loads(path.read_text(encoding="utf-8"))
