"""Sketchfab is the only model catalog.

Mixamo is not a source. Its terms restrict automated download. Export a rigged
humanoid by hand into ``shows/<id>/assets/`` if you want to replace the mannequin.

Text-to-3D is the ``TextTo3DSource`` class below. It is not registered.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

USER_AGENT = "reelshort-content-pipeline/1.0"
SKETCHFAB_API = "https://api.sketchfab.com"


@dataclass(frozen=True)
class Need:
    query: str
    tags: tuple[str, ...] = ()
    style: str = ""
    size_meters: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class Candidate:
    source: str
    source_id: str
    title: str
    author: str
    license: str
    page_url: str
    tags: tuple[str, ...] = ()
    version: str = ""
    download_url: str | None = None
    local_path: str | None = None
    size_meters: tuple[float, float, float] | None = None


def normalize_license(raw: str | None) -> str | None:
    """Keep CC0 and CC-BY. Drop NC, ND, and SA.

    Sketchfab search rows often omit ``license.slug`` and send the label
    ``CC Attribution``. The model endpoint sends slug ``by``.
    """
    if not raw:
        return None
    text = raw.strip().lower().replace("_", "-")
    if any(flag in text for flag in ("-nc", "noncommercial", "-nd", "noderiv", "-sa", "sharealike")):
        return None
    if "cc0" in text or text in {"zero", "public domain", "cc-zero", "publicdomain"}:
        return "CC0"
    if "attribution" in text or text in {"by", "cc-by", "ccby", "attribution"} or "cc-by" in text:
        return "CC-BY"
    return None


class TextTo3DSource:
    """Extension point for a future text-to-3D model. Not registered, and not called."""

    name = "text-to-3d"

    def search(self, need: Need, limit: int = 4) -> list[Candidate]:
        raise RuntimeError("Text-to-3D is not part of this pipeline")


class SketchfabSource:
    """Downloadable CC0 and CC-BY models. Needs SKETCHFAB_TOKEN."""

    name = "sketchfab"

    def __init__(self, token: str | None = None) -> None:
        self.token = token if token is not None else os.environ.get("SKETCHFAB_TOKEN", "")

    def search(self, need: Need, limit: int = 4) -> list[Candidate]:
        if not self.token:
            return []
        query = urllib.parse.urlencode(
            {
                "type": "models",
                "q": need.query,
                "downloadable": "true",
                "licenses": "cc0,by",
                "count": str(limit),
                "sort_by": "-relevance",
            }
        )
        payload = _get_json(
            f"{SKETCHFAB_API}/v3/search?{query}",
            headers={"Authorization": f"Token {self.token}"},
        )
        time.sleep(0.25)
        found: list[Candidate] = []
        for item in payload.get("results") or []:
            license_name = normalize_license(
                ((item.get("license") or {}).get("slug")) or ((item.get("license") or {}).get("label"))
            )
            if license_name is None or not item.get("uid"):
                continue
            user = item.get("user") or {}
            tags = tuple(tag.get("name", "") for tag in item.get("tags") or [] if isinstance(tag, dict))
            found.append(
                Candidate(
                    source=self.name,
                    source_id=str(item["uid"]),
                    title=str(item.get("name") or item["uid"]),
                    author=str(user.get("username") or user.get("displayName") or "Sketchfab"),
                    license=license_name,
                    page_url=str(item.get("viewerUrl") or f"https://sketchfab.com/3d-models/{item['uid']}"),
                    tags=tuple(tag for tag in tags if tag),
                    download_url=f"{SKETCHFAB_API}/v3/models/{item['uid']}/download",
                )
            )
        return found

    def lookup(self, source_id: str) -> Candidate | None:
        if not self.token or not source_id:
            return None
        payload = _get_json(
            f"{SKETCHFAB_API}/v3/models/{urllib.parse.quote(source_id)}",
            headers={"Authorization": f"Token {self.token}"},
        )
        time.sleep(0.25)
        license_name = normalize_license(
            ((payload.get("license") or {}).get("slug")) or ((payload.get("license") or {}).get("label"))
        )
        if license_name is None or not payload.get("uid") or not payload.get("isDownloadable", True):
            return None
        user = payload.get("user") or {}
        tags = tuple(tag.get("name", "") for tag in payload.get("tags") or [] if isinstance(tag, dict))
        uid = str(payload["uid"])
        return Candidate(
            source=self.name,
            source_id=uid,
            title=str(payload.get("name") or uid),
            author=str(user.get("username") or user.get("displayName") or "Sketchfab"),
            license=license_name,
            page_url=str(payload.get("viewerUrl") or f"https://sketchfab.com/3d-models/{uid}"),
            tags=tuple(tag for tag in tags if tag),
            download_url=f"{SKETCHFAB_API}/v3/models/{uid}/download",
        )

    def fetch(self, candidate: Candidate, directory: Path) -> Path:
        if not self.token or not candidate.download_url:
            raise RuntimeError("Sketchfab download needs SKETCHFAB_TOKEN")
        payload = _get_json(candidate.download_url, headers={"Authorization": f"Token {self.token}"})
        url = None
        kind = None
        for key in ("glb", "gltf", "source"):
            entry = payload.get(key)
            if isinstance(entry, dict) and entry.get("url"):
                url = entry["url"]
                kind = key
                break
        if not url:
            raise RuntimeError(f"Sketchfab returned no glTF archive for {candidate.source_id}")
        directory.mkdir(parents=True, exist_ok=True)
        # The glb entry is one binary. glTF and source entries are archives.
        suffix = ".glb" if kind == "glb" else ".zip"
        destination = directory / f"{candidate.source_id}{suffix}"
        legacy = directory / f"{candidate.source_id}.zip"
        if destination.is_file() and destination.stat().st_size > 0:
            return materialize_mesh(destination)
        if (
            suffix == ".glb"
            and legacy.is_file()
            and _starts_with(legacy, b"glTF")
        ):
            legacy.replace(destination)
        else:
            _download(url, destination)
        return materialize_mesh(destination)


def default_sources(_library_raw: Path) -> list:
    return [SketchfabSource()]


def _tokens(text: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", text.lower()) if len(token) > 2]


def _starts_with(path: Path, magic: bytes) -> bool:
    with path.open("rb") as handle:
        return handle.read(len(magic)) == magic


def materialize_mesh(path: Path) -> Path:
    """Return a mesh path. Extract a real zip; a GLB mislabeled as ``.zip`` stays a GLB."""
    if zipfile.is_zipfile(path):
        extract_dir = path.with_suffix("") if path.suffix.lower() == ".zip" else path.parent / f"{path.stem}-extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as archive:
            archive.extractall(extract_dir)
        for suffix in (".glb", ".gltf", ".obj"):
            matches = sorted(extract_dir.rglob(f"*{suffix}"))
            if matches:
                return matches[0]
        raise RuntimeError(f"No mesh inside {path}")
    if path.suffix.lower() != ".glb" and _starts_with(path, b"glTF"):
        glb_path = path.with_suffix(".glb")
        if glb_path != path:
            path.replace(glb_path)
        return glb_path
    return path


def _get_json(url: str, headers: dict | None = None) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object from {url}")
    return payload


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            destination.write_bytes(response.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Download failed ({exc.code}) for {url}") from exc
