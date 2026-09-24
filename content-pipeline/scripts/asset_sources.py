"""Turn a downloaded or generated file into a mesh path."""

from __future__ import annotations

import zipfile
from pathlib import Path


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
