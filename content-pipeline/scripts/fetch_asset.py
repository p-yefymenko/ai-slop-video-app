"""Download one CC0 or CC-BY model into the shared prefab library.

Refuses Mixamo and any host that is not Poly Haven, Sketchfab, or Smithsonian
Open Access. Sketchfab still needs SKETCHFAB_TOKEN, and the download is skipped
when the API refuses an unattended token.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from asset_resolver import AssetRequest, AssetResolver, asset_hash  # noqa: E402
from asset_sources import (  # noqa: E402
    Candidate,
    PolyHavenSource,
    SketchfabSource,
    _download,
    _get_json,
    materialize_mesh,
    normalize_license,
)
from mesh_io import read_schema_mesh  # noqa: E402
from pipeline_paths import LIBRARY_DIR, load_content_env  # noqa: E402

SKETCHFAB_API = "https://api.sketchfab.com"


def main() -> None:
    load_content_env()
    parser = argparse.ArgumentParser(description="Fetch one allowed catalog model.")
    parser.add_argument("url")
    parser.add_argument("--size", help="width,depth,height in meters. Default: the mesh's own extent")
    args = parser.parse_args()
    candidate, fetch = candidate_for_url(args.url.strip())
    raw_dir = LIBRARY_DIR / "raw" / candidate.source / _slug(candidate.source_id)
    fetched = materialize_mesh(fetch(candidate, raw_dir))
    vertices, _faces = read_schema_mesh(fetched)
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    if args.size:
        parts = [float(part) for part in args.size.split(",")]
        if len(parts) != 3:
            raise SystemExit("--size must be width,depth,height")
        size = (parts[0], parts[1], parts[2])
    else:
        size = (float(extent[0]), float(extent[1]), float(extent[2]))
    candidate = Candidate(
        source=candidate.source,
        source_id=candidate.source_id,
        title=candidate.title,
        author=candidate.author,
        license=candidate.license,
        page_url=candidate.page_url,
        tags=candidate.tags,
        version=candidate.version,
        local_path=str(fetched),
        size_meters=size,
    )
    request = AssetRequest(
        consumer_id=f"fetch:{candidate.source_id}",
        label=candidate.source_id,
        size=size,
        position=None,
        location_id=None,
        asset_id=f"{candidate.source}:{candidate.source_id}",
        prefab_id=None,
    )

    class OneShot:
        name = candidate.source

        def lookup(self, source_id: str):
            return candidate if source_id == candidate.source_id else None

        def fetch(self, item, directory):
            return Path(item.local_path)

    resolver = AssetResolver("manual", sources=[OneShot()])
    resolved = resolver.resolve_one(request)
    resolver._write_json(LIBRARY_DIR / "lock.json", resolver.lock)
    resolver._write_json(LIBRARY_DIR / "sources.json", resolver.sources_catalog)
    for warning in resolver.warnings:
        print(warning)
    print(f"{resolved.prefab_id}  {asset_hash(candidate.source, candidate.source_id, size)}")


def candidate_for_url(url: str):
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    if "mixamo" in host:
        raise SystemExit(
            "Mixamo does not allow automated download. Export the file by hand into shows/<id>/assets/."
        )
    if host == "polyhaven.com" or host.endswith(".polyhaven.com"):
        asset_id = [part for part in parsed.path.split("/") if part][-1]
        if not asset_id:
            raise SystemExit(f"Could not read a Poly Haven asset id from {url}")
        candidate = Candidate(
            source="polyhaven",
            source_id=asset_id,
            title=asset_id.replace("_", " "),
            author="Poly Haven",
            license="CC0",
            page_url=f"https://polyhaven.com/a/{asset_id}",
        )
        return candidate, PolyHavenSource().fetch
    if host == "sketchfab.com" or host.endswith(".sketchfab.com"):
        uid = _sketchfab_uid(parsed.path)
        token = os.environ.get("SKETCHFAB_TOKEN", "")
        if not token:
            raise SystemExit("Sketchfab download needs SKETCHFAB_TOKEN in content-pipeline/.env")
        payload = _get_json(
            f"{SKETCHFAB_API}/v3/models/{uid}",
            headers={"Authorization": f"Token {token}"},
        )
        license_name = normalize_license(
            ((payload.get("license") or {}).get("slug")) or ((payload.get("license") or {}).get("label"))
        )
        if license_name is None:
            raise SystemExit(f"Refusing Sketchfab model {uid}: license is not CC0 or CC-BY")
        if not payload.get("isDownloadable", True):
            raise SystemExit(f"Sketchfab model {uid} is not downloadable")
        user = payload.get("user") or {}
        candidate = Candidate(
            source="sketchfab",
            source_id=uid,
            title=str(payload.get("name") or uid),
            author=str(user.get("username") or user.get("displayName") or "Sketchfab"),
            license=license_name,
            page_url=str(payload.get("viewerUrl") or url),
            download_url=f"{SKETCHFAB_API}/v3/models/{uid}/download",
        )
        return candidate, SketchfabSource(token).fetch
    if host == "si.edu" or host.endswith(".si.edu"):
        if not parsed.path.lower().split("?")[0].endswith((".glb", ".gltf")):
            raise SystemExit("Smithsonian fetch accepts a direct .glb or .gltf URL")
        name = parsed.path.rstrip("/").split("/")[-1]
        candidate = Candidate(
            source="smithsonian",
            source_id=name,
            title=name,
            author="Smithsonian Open Access",
            license="CC0",
            page_url=url,
            download_url=url,
        )
        return candidate, _fetch_direct
    raise SystemExit(
        f"Refusing {host or url}. content:fetch-asset accepts Poly Haven, Sketchfab, and Smithsonian Open Access URLs."
    )


def _fetch_direct(candidate: Candidate, directory: Path) -> Path:
    if not candidate.download_url:
        raise RuntimeError("Missing download URL")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / candidate.source_id
    _download(candidate.download_url, destination)
    return destination


def _sketchfab_uid(path: str) -> str:
    parts = [part for part in path.split("/") if part]
    if not parts:
        raise SystemExit("Could not read a Sketchfab model id from the URL")
    tail = parts[-1]
    token = tail.split("-")[-1]
    if len(token) < 16:
        raise SystemExit(f"Could not read a Sketchfab model id from {path}")
    return token


def _slug(value: str) -> str:
    import re

    token = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return token[:80] or "asset"


if __name__ == "__main__":
    main()
