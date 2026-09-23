"""Resolve landmark and prop needs into prefabs, then build one set per location.

Characters stay out of the set. Their clay mannequins are built from
``ShowCharacter.proxy`` at previs time. A rigged CC0 humanoid can later replace
that mannequin by dropping it in ``shows/<id>/assets/``; this resolver does not
fetch one. Text-to-3D is not registered.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from asset_sources import (
    Candidate,
    Need,
    default_sources,
    materialize_mesh,
    normalize_license,
)
from asset_sources import _tokens as tokens
from mesh_io import (
    decimate,
    fit_to_size,
    primitive_mesh,
    proportions_match,
    read_schema_mesh,
    schema_triangles,
    write_schema_glb,
)
from pipeline_paths import LIBRARY_DIR, OUTPUT_DIR, SHOWS_DIR

KIND_CATEGORY = {
    "box": "blocks",
    "column": "columns",
    "pedestal": "pedestals",
    "window": "windows",
    "door": "doors",
    "seat": "seats",
}


@dataclass
class AssetRequest:
    consumer_id: str
    label: str
    kind: str
    size: tuple[float, float, float]
    position: tuple[float, float, float] | None
    location_id: str | None
    need: Need | None
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
        sources=None,
        offline: bool = False,
        refresh: bool = False,
        review: bool = False,
        fetch_limit: int = 4,
        score_fn=None,
        clip_fn=None,
        write_thumbs: bool = False,
    ) -> None:
        self.show_id = show_id
        self.library_dir = library_dir
        self.shows_dir = shows_dir
        self.output_dir = output_dir
        self.sources = (
            sources
            if sources is not None
            else default_sources(library_dir / "raw")
        )
        self.offline = offline
        self.refresh = refresh
        self.review = review
        self.fetch_limit = fetch_limit
        self.score_fn = score_fn or keyword_score
        self.clip_fn = clip_fn
        self.write_thumbs = write_thumbs or review or clip_fn is not None
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
        if request.need is None:
            return self._primitive(request, digest=None, provisional=False)
        digest = need_hash(request.need, request.size)
        cached = self._cached(request, digest)
        if cached is not None and not self.refresh:
            return cached
        if self.offline:
            return self._primitive(request, digest, provisional=True)
        chosen, reviewed = self._search_and_import(request, digest)
        if self.review:
            self._write_review(request, digest, chosen, reviewed)
        if chosen is not None:
            return chosen
        return self._primitive(request, digest, provisional=True)

    def _cached(self, request: AssetRequest, digest: str) -> ResolvedPrefab | None:
        for root, origin in (
            (self._show_assets(), "show"),
            (self.library_dir / "prefabs", "library"),
        ):
            found = self._find_need(root, digest, origin)
            if found is not None:
                return found
        entry = self.lock["needs"].get(digest)
        if not entry or self.refresh or (entry.get("provisional") and not self.offline):
            return None
        return self._find_prefab(str(entry.get("prefabId") or ""))

    def _show_assets(self) -> Path:
        return self.shows_dir / self.show_id / "assets"

    def _search_and_import(
        self, request: AssetRequest, digest: str
    ) -> tuple[ResolvedPrefab | None, list[dict]]:
        assert request.need is not None
        found: list[Candidate] = []
        for source in self.sources:
            try:
                found.extend(source.search(request.need, limit=self.fetch_limit))
            except Exception as exc:
                self.warnings.append(f"{getattr(source, 'name', 'source')}: {exc}")
        eligible = []
        for candidate in found:
            license_name = normalize_license(candidate.license)
            if license_name is None:
                continue
            if candidate.size_meters and not proportions_match(
                np.array(candidate.size_meters), request.size
            ):
                continue
            eligible.append(candidate)
        ranked = sorted(
            eligible,
            key=lambda candidate: (-self.score_fn(request.need, candidate), candidate.source_id),
        )
        chosen: ResolvedPrefab | None = None
        reviewed: list[dict] = []
        for candidate in ranked[: self.fetch_limit]:
            try:
                imported = self._import_candidate(request, candidate, digest)
            except Exception as exc:
                self.warnings.append(
                    f"{request.consumer_id}: skipped {candidate.source}:{candidate.source_id} ({exc})"
                )
                continue
            reviewed.append(
                {
                    "prefabId": imported["resolved"].prefab_id,
                    "title": imported["resolved"].title,
                    "source": imported["resolved"].source,
                    "license": imported["resolved"].license,
                    "author": imported["resolved"].author,
                    "pageUrl": imported["resolved"].page_url,
                    "score": self.score_fn(request.need, candidate) if request.need else 0,
                    "thumb": "thumb.png" if self.write_thumbs else None,
                    "resolved": imported["resolved"],
                }
            )
            if chosen is None:
                chosen = imported["resolved"]
                if not self.review and self.clip_fn is None:
                    break
        if self.clip_fn and reviewed:
            def clip_key(item: dict) -> float:
                thumb = item["resolved"].glb.parent / "thumb.png"
                if not thumb.is_file() or request.need is None:
                    return float(item["score"])
                return float(self.clip_fn(request.need.query, thumb))

            best = max(reviewed, key=clip_key)
            chosen = best["resolved"]
            digest_entry = self.lock["needs"].get(digest)
            if digest_entry is not None:
                digest_entry["prefabId"] = chosen.prefab_id
                digest_entry["source"] = chosen.source
                digest_entry["sourceId"] = chosen.source_id
        for item in reviewed:
            item.pop("resolved", None)
        return chosen, reviewed

    def _import_candidate(self, request: AssetRequest, candidate: Candidate, digest: str) -> dict:
        raw_dir = self.library_dir / "raw" / candidate.source / _slug(candidate.source_id)
        if candidate.local_path:
            fetched = Path(candidate.local_path)
            if self.library_dir not in fetched.parents:
                raw_dir.mkdir(parents=True, exist_ok=True)
                stored = raw_dir / fetched.name
                if fetched.resolve() != stored.resolve():
                    stored.write_bytes(fetched.read_bytes())
                fetched = stored
            fetched = materialize_mesh(fetched)
        else:
            source = next(item for item in self.sources if item.name == candidate.source)
            fetched = materialize_mesh(source.fetch(candidate, raw_dir))
        vertices, faces = read_schema_mesh(fetched)
        extent = vertices.max(axis=0) - vertices.min(axis=0)
        if not proportions_match(extent, request.size):
            raise RuntimeError("proportions are far from the requested size")
        fitted = fit_to_size(vertices, request.size)
        fitted, faces = decimate(fitted, faces)
        category = KIND_CATEGORY.get(request.kind, "props")
        prefab_id = f"{category}/{_slug(candidate.source_id)}"
        directory = self.library_dir / "prefabs" / category / _slug(candidate.source_id)
        resolved = self._write_prefab(
            directory,
            prefab_id,
            fitted,
            faces,
            request,
            origin="downloaded",
            source=candidate.source,
            source_id=candidate.source_id,
            title=candidate.title,
            author=candidate.author,
            license_name=normalize_license(candidate.license) or candidate.license,
            page_url=candidate.page_url,
            digest=digest,
            version=candidate.version,
        )
        self._remember_source(resolved, candidate.version)
        self.lock["needs"][digest] = _lock_entry(request, resolved, candidate.version, provisional=False)
        return {
            "resolved": resolved,
            "review": {
                "prefabId": resolved.prefab_id,
                "title": resolved.title,
                "source": resolved.source,
                "license": resolved.license,
                "author": resolved.author,
                "pageUrl": resolved.page_url,
                "score": self.score_fn(request.need, candidate) if request.need else 0,
                "thumb": "thumb.png" if self.write_thumbs else None,
            },
        }

    def _primitive(self, request: AssetRequest, digest: str | None, provisional: bool) -> ResolvedPrefab:
        prefab_id = fallback_prefab_id(request.kind, request.size)
        directory = (
            self.output_dir / self.show_id / "assets" / "fallback" / prefab_id.split("/", 1)[1]
        )
        existing = directory / "model.glb"
        if not existing.is_file() or self.refresh:
            vertices, faces = primitive_mesh(request.kind, request.size)
            resolved = self._write_prefab(
                directory,
                prefab_id,
                vertices,
                faces,
                request,
                origin="fallback",
                source="primitive",
                source_id=request.kind,
                title=request.kind,
                author="",
                license_name="",
                page_url="",
                digest=digest,
                version="1",
            )
        else:
            found = self._read_prefab(directory / "prefab.json", "fallback")
            if found is None:
                raise RuntimeError(f"Fallback prefab {directory} is incomplete")
            resolved = found
        if digest and provisional:
            self.lock["needs"][digest] = _lock_entry(request, resolved, "1", provisional=True)
        self.warnings.append(f"{request.consumer_id}: used a {request.kind} primitive")
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
            "needHash": digest,
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
                return self._read_prefab(path, record.get("origin") or origin)
        return None

    def _find_need(self, root: Path, digest: str, origin: str) -> ResolvedPrefab | None:
        if not root.is_dir():
            return None
        for path in sorted(root.rglob("prefab.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("needHash") == digest:
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

    def _write_review(
        self,
        request: AssetRequest,
        digest: str,
        chosen: ResolvedPrefab | None,
        reviewed: list[dict],
    ) -> None:
        directory = self.output_dir / self.show_id / "asset-review" / digest
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "needHash": digest,
            "consumerId": request.consumer_id,
            "need": None
            if request.need is None
            else {
                "query": request.need.query,
                "tags": list(request.need.tags),
                "style": request.need.style,
            },
            "sizeMeters": list(request.size),
            "chosen": None if chosen is None else chosen.prefab_id,
            "candidates": reviewed,
        }
        (directory / "candidates.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

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
                    kind=str(landmark.get("kind") or "box"),
                    size=_vec3(landmark.get("size"), (1.0, 1.0, 1.0)),
                    position=_vec3(landmark.get("position"), (0.0, 0.0, 0.0)),
                    location_id=location_id,
                    need=_need(landmark.get("need"), _vec3(landmark.get("size"), (1.0, 1.0, 1.0))),
                    prefab_id=landmark.get("prefabId"),
                )
            )
    for prop_id, prop in (show.get("props") or {}).items():
        size = _vec3(prop.get("sizeMeters"), (0.3, 0.3, 0.3))
        requests.append(
            AssetRequest(
                consumer_id=f"prop:{prop_id}",
                label=prop_id,
                kind="box",
                size=size,
                position=None,
                location_id=None,
                need=_need(prop.get("need"), size),
                prefab_id=prop.get("prefabId"),
            )
        )
    return requests


def need_hash(need: Need, size: tuple[float, float, float]) -> str:
    payload = {
        "query": need.query.strip().lower(),
        "tags": sorted(tag.lower() for tag in need.tags),
        "style": need.style.strip().lower(),
        "sizeMeters": [round(float(value), 4) for value in size],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def keyword_score(need: Need, candidate: Candidate) -> float:
    wanted = tokens(need.query) + [token for tag in need.tags for token in tokens(tag)]
    if not wanted:
        return 0.0
    haystack = set(tokens(" ".join((candidate.title, candidate.source_id, *candidate.tags))))
    unique = list(dict.fromkeys(wanted))
    return sum(1 for token in unique if token in haystack) / len(unique)


def fallback_prefab_id(kind: str, size: tuple[float, float, float]) -> str:
    parts = "-".join(str(int(round(float(value) * 1000))) for value in size)
    return f"fallback/{kind}-{parts}"


def apply_lock_override(lock_path: Path, digest: str, prefab_id: str) -> None:
    lock = _read_json(lock_path, {"version": 1, "needs": {}})
    entry = lock.setdefault("needs", {}).get(digest)
    if entry is None:
        raise SystemExit(f"No locked need {digest}")
    entry["prefabId"] = prefab_id
    entry["origin"] = "override"
    entry["provisional"] = False
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")


def export_credits(catalog_path: Path, destination: Path) -> None:
    catalog = _read_json(catalog_path, {"assets": []})
    lines = [
        "# Credits",
        "",
        "Models fetched into `content-pipeline/library` keep the license they were published under.",
        "CC-BY assets need attribution when a video that uses them is published.",
        "Poly Haven's live API also asks for a visible Powered by Poly Haven credit.",
        "",
    ]
    assets = catalog.get("assets") or []
    if not assets:
        lines.append("No third-party models have been fetched.")
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


def _need(raw, size: tuple[float, float, float]) -> Need | None:
    if not isinstance(raw, dict) or not str(raw.get("query") or "").strip():
        return None
    tags = tuple(str(tag) for tag in raw.get("tags") or [])
    return Need(
        query=str(raw["query"]).strip(),
        tags=tags,
        style=str(raw.get("style") or ""),
        size_meters=size,
    )


def _vec3(raw, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if not isinstance(raw, list) or len(raw) != 3:
        return default
    return (float(raw[0]), float(raw[1]), float(raw[2]))


def _slug(value: str) -> str:
    import re

    token = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return token[:80] or "asset"


def _lock_entry(request: AssetRequest, resolved: ResolvedPrefab, version: str, provisional: bool) -> dict:
    need = request.need
    return {
        "need": None
        if need is None
        else {"query": need.query, "tags": list(need.tags), "style": need.style},
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
