"""Open-source model catalogs.

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
POLYHAVEN_API = "https://api.polyhaven.com"
SKETCHFAB_API = "https://api.sketchfab.com"
SMITHSONIAN_API = "https://api.si.edu/openaccess/api/v1.0"

# Sketchfab's download guidelines require end-user OAuth for unattended apps
# unless Sketchfab grants an exception. A token can search downloadable CC0 and
# CC-BY models. The download call is attempted with that token and skipped when
# the API refuses it. Poly Haven's live API asks for a unique User-Agent and a
# "Powered by Poly Haven" credit; the assets themselves are CC0. Smithsonian
# Open Access media is CC0 and needs an api.data.gov key. Kenney and Quaternius
# packs are CC0 and have no per-item API, so they are indexed from local files.


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
    """Keep CC0 and CC-BY. Drop NC, ND, and SA."""
    if not raw:
        return None
    text = raw.strip().lower().replace("_", "-")
    if any(flag in text for flag in ("-nc", "noncommercial", "-nd", "noderiv", "-sa", "sharealike")):
        return None
    if "cc0" in text or text in {"zero", "public domain", "cc-zero", "publicdomain"}:
        return "CC0"
    if text in {"by", "cc-by", "ccby", "attribution"} or "cc-by" in text:
        return "CC-BY"
    return None


def objaverse_license_allowed(raw: str | None) -> bool:
    """Objaverse objects stay eligible only when the object license is CC0 or CC-BY.

    The dataset as a whole is ODC-By and also contains NC and SA objects, so a
    license check on the individual record is required. The annotation index is
    large; ``ObjaverseSource`` does not download it unless explicitly enabled.
    """
    return normalize_license(raw) in {"CC0", "CC-BY"}


class TextTo3DSource:
    """Extension point for a future text-to-3D model. Not registered, and not called."""

    name = "text-to-3d"

    def search(self, need: Need, limit: int = 4) -> list[Candidate]:
        raise RuntimeError("Text-to-3D is not part of this pipeline")


class ObjaverseSource:
    """Optional catalog. Registered only so an explicit flag can turn it on."""

    name = "objaverse"

    def __init__(self) -> None:
        self._warned = False

    def search(self, need: Need, limit: int = 4) -> list[Candidate]:
        if os.environ.get("OBJAVERSE_ENABLE") != "1":
            return []
        if not self._warned:
            self._warned = True
            print(
                "Objaverse stays unused. Enabling it would download the annotation "
                "index; this pipeline does not bulk-download that dataset.",
                flush=True,
            )
        return []

    def fetch(self, candidate: Candidate, directory: Path) -> Path:
        raise RuntimeError("Objaverse download is disabled")


class LocalPackSource:
    """Index GLB, glTF, and OBJ files already stored under a pack directory."""

    def __init__(self, name: str, root: Path, author: str) -> None:
        self.name = name
        self.root = root
        self.author = author

    def search(self, need: Need, limit: int = 4) -> list[Candidate]:
        if not self.root.is_dir():
            return []
        tokens = _tokens(need.query) + [token for tag in need.tags for token in _tokens(tag)]
        ranked: list[tuple[float, Candidate]] = []
        for path in sorted(self.root.rglob("*")):
            if path.suffix.lower() not in {".glb", ".gltf", ".obj"}:
                continue
            haystack = set(_tokens(path.stem))
            if not tokens:
                continue
            score = sum(1 for token in tokens if token in haystack) / len(tokens)
            if score <= 0:
                continue
            ranked.append(
                (
                    score,
                    Candidate(
                        source=self.name,
                        source_id=path.relative_to(self.root).as_posix(),
                        title=path.stem.replace("_", " ").replace("-", " "),
                        author=self.author,
                        license="CC0",
                        page_url=f"local://{self.name}/{path.relative_to(self.root).as_posix()}",
                        tags=tuple(sorted(haystack)),
                        local_path=str(path),
                    ),
                )
            )
        ranked.sort(key=lambda item: (-item[0], item[1].source_id))
        return [candidate for _score, candidate in ranked[:limit]]

    def fetch(self, candidate: Candidate, directory: Path) -> Path:
        if not candidate.local_path:
            raise RuntimeError(f"{self.name} candidate {candidate.source_id} has no local file")
        source = Path(candidate.local_path)
        destination = directory / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        return destination


class PolyHavenSource:
    """CC0 models. One asset list per process, then one file download for a chosen model."""

    name = "polyhaven"

    def __init__(self) -> None:
        self._assets: dict | None = None

    def search(self, need: Need, limit: int = 4) -> list[Candidate]:
        assets = self._asset_list()
        tokens = _tokens(need.query) + [token for tag in need.tags for token in _tokens(tag)]
        ranked: list[tuple[float, Candidate]] = []
        for asset_id, meta in assets.items():
            if not isinstance(meta, dict):
                continue
            tags = tuple(str(tag) for tag in meta.get("tags") or [])
            categories = tuple(str(tag) for tag in meta.get("categories") or [])
            name = str(meta.get("name") or asset_id)
            haystack = set(_tokens(" ".join((asset_id, name, *tags, *categories))))
            if not tokens:
                continue
            score = sum(1 for token in tokens if token in haystack) / len(tokens)
            if score <= 0:
                continue
            authors = meta.get("authors") or {}
            author = ", ".join(authors) if isinstance(authors, dict) else str(authors)
            dimensions = meta.get("dimensions")
            size = None
            if isinstance(dimensions, list) and len(dimensions) >= 3:
                size = tuple(float(value) / 1000.0 for value in dimensions[:3])
            ranked.append(
                (
                    score,
                    Candidate(
                        source=self.name,
                        source_id=asset_id,
                        title=name,
                        author=author or "Poly Haven",
                        license="CC0",
                        page_url=f"https://polyhaven.com/a/{asset_id}",
                        tags=tags,
                        version=str(meta.get("files_hash") or ""),
                        size_meters=size,  # type: ignore[arg-type]
                    ),
                )
            )
        ranked.sort(key=lambda item: (-item[0], item[1].source_id))
        return [candidate for _score, candidate in ranked[:limit]]

    def fetch(self, candidate: Candidate, directory: Path) -> Path:
        payload = _get_json(f"{POLYHAVEN_API}/files/{urllib.parse.quote(candidate.source_id)}")
        picked = _polyhaven_gltf(payload)
        if picked is None:
            raise RuntimeError(f"Poly Haven has no glTF for {candidate.source_id}")
        directory.mkdir(parents=True, exist_ok=True)
        filename = picked["url"].rstrip("/").split("/")[-1] or f"{candidate.source_id}.gltf"
        destination = directory / filename
        _download(picked["url"], destination)
        includes = picked.get("include") or {}
        if isinstance(includes, dict):
            for include_name, include in includes.items():
                if isinstance(include, dict) and include.get("url"):
                    _download(include["url"], directory / include_name)
        return destination

    def _asset_list(self) -> dict:
        if self._assets is None:
            payload = _get_json(f"{POLYHAVEN_API}/assets?t=models")
            self._assets = payload if isinstance(payload, dict) else {}
        return self._assets


class SketchfabSource:
    """Downloadable CC0 and CC-BY models. Needs SKETCHFAB_TOKEN. No bulk download."""

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

    def fetch(self, candidate: Candidate, directory: Path) -> Path:
        if not self.token or not candidate.download_url:
            raise RuntimeError("Sketchfab download needs SKETCHFAB_TOKEN")
        payload = _get_json(candidate.download_url, headers={"Authorization": f"Token {self.token}"})
        url = None
        for key in ("glb", "gltf", "source"):
            entry = payload.get(key)
            if isinstance(entry, dict) and entry.get("url"):
                url = entry["url"]
                break
        if not url:
            raise RuntimeError(
                f"Sketchfab refused a direct download for {candidate.source_id}. "
                "The download API expects an end-user login unless Sketchfab grants an exception."
            )
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{candidate.source_id}.zip"
        _download(url, destination)
        return materialize_mesh(destination)


class SmithsonianSource:
    """CC0 3D records. Needs SMITHSONIAN_API_KEY from api.data.gov."""

    name = "smithsonian"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("SMITHSONIAN_API_KEY", "")

    def search(self, need: Need, limit: int = 4) -> list[Candidate]:
        if not self.api_key:
            return []
        query = urllib.parse.urlencode(
            {
                "q": f'{need.query} AND online_media_type:"3D Models"',
                "rows": str(limit),
                "api_key": self.api_key,
            }
        )
        payload = _get_json(f"{SMITHSONIAN_API}/search?{query}")
        rows = ((payload.get("response") or {}).get("rows")) or []
        found: list[Candidate] = []
        for row in rows:
            content = row.get("content") or row
            title = _smithsonian_title(content) or need.query
            urls = [item for item in _walk_strings(content) if item.lower().split("?")[0].endswith((".glb", ".gltf"))]
            if not urls:
                continue
            record_id = str(row.get("id") or row.get("url") or title)
            found.append(
                Candidate(
                    source=self.name,
                    source_id=record_id,
                    title=title,
                    author="Smithsonian Open Access",
                    license="CC0",
                    page_url=str(row.get("url") or "https://www.si.edu/openaccess"),
                    download_url=urls[0],
                )
            )
        return found

    def fetch(self, candidate: Candidate, directory: Path) -> Path:
        if not candidate.download_url:
            raise RuntimeError(f"Smithsonian record {candidate.source_id} has no GLB")
        directory.mkdir(parents=True, exist_ok=True)
        name = candidate.download_url.rstrip("/").split("/")[-1].split("?")[0] or "model.glb"
        destination = directory / name
        _download(candidate.download_url, destination)
        return destination


def default_sources(library_raw: Path) -> list:
    return [
        PolyHavenSource(),
        SketchfabSource(),
        SmithsonianSource(),
        LocalPackSource("kenney", library_raw / "kenney", "Kenney"),
        LocalPackSource("quaternius", library_raw / "quaternius", "Quaternius"),
        ObjaverseSource(),
    ]


def _tokens(text: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", text.lower()) if len(token) > 2]


def _polyhaven_gltf(payload: dict) -> dict | None:
    gltf = payload.get("gltf") if isinstance(payload, dict) else None
    if not isinstance(gltf, dict):
        return None
    for _resolution, entry in gltf.items():
        if isinstance(entry, dict) and isinstance(entry.get("gltf"), dict) and entry["gltf"].get("url"):
            return entry["gltf"]
    return None


def _smithsonian_title(content: dict) -> str:
    descriptive = content.get("descriptiveNonRepeating") or {}
    title = descriptive.get("title") or content.get("title") or {}
    if isinstance(title, dict):
        return str(title.get("content") or "")
    return str(title or "")


def _walk_strings(value) -> list[str]:
    found: list[str] = []
    if isinstance(value, str):
        found.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            found.extend(_walk_strings(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_walk_strings(item))
    return found


def materialize_mesh(path: Path) -> Path:
    """Return a mesh path, extracting the first glTF or OBJ when ``path`` is a zip."""
    if path.suffix.lower() != ".zip" and not zipfile.is_zipfile(path):
        return path
    extract_dir = path.with_suffix("") if path.suffix.lower() == ".zip" else path.parent / f"{path.stem}-extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        archive.extractall(extract_dir)
    for suffix in (".glb", ".gltf", ".obj"):
        matches = sorted(extract_dir.rglob(f"*{suffix}"))
        if matches:
            return matches[0]
    raise RuntimeError(f"No mesh inside {path}")


def _get_json(url: str, headers: dict | None = None) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object from {url}")
    return payload


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            destination.write_bytes(response.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Download failed ({exc.code}) for {url}") from exc
