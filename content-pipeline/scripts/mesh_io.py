"""Triangle meshes stored as glTF Y-up GLB files.

Location meshes are geometry only. A character mesh can carry the plate color
on COLOR_0, one RGB value per vertex.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from coords import gltf_points_to_schema, schema_points_to_gltf

GENERATOR = "reelshort-content-pipeline"
# Default polygon limit. Override with `pnpm run content:assets -- --triangles <count>`.
TRIANGLE_BUDGET = 10_000_000
# ComfyUI DecimateMesh rejects a target above this.
TRIANGLE_BUDGET_MAX = 50_000_000
DECIMATOR = "comfy-decimate-mesh"


def require_triangle_budget(value: int) -> int:
    """The polygon limit passed to DecimateMesh."""
    limit = int(value)
    if limit < 1 or limit > TRIANGLE_BUDGET_MAX:
        raise ValueError(
            f"Polygon limit must be from 1 to {TRIANGLE_BUDGET_MAX:,}. Got {limit}."
        )
    return limit


def box_mesh(size: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Schema-space box. Origin is the center of the base. ``size`` is width, depth, height."""
    half_x, half_y = size[0] / 2.0, size[1] / 2.0
    height = size[2]
    corners = np.array(
        [
            [-half_x, -half_y, 0.0],
            [half_x, -half_y, 0.0],
            [half_x, half_y, 0.0],
            [-half_x, half_y, 0.0],
            [-half_x, -half_y, height],
            [half_x, -half_y, height],
            [half_x, half_y, height],
            [-half_x, half_y, height],
        ],
        dtype=np.float64,
    )
    quads = (
        (0, 1, 2, 3),
        (4, 7, 6, 5),
        (0, 4, 5, 1),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (3, 7, 4, 0),
    )
    faces = []
    for a, b, c, d in quads:
        faces.extend(((a, b, c), (a, c, d)))
    return corners, np.array(faces, dtype=np.int64)


def fit_to_size(
    vertices: np.ndarray, size: tuple[float, float, float]
) -> np.ndarray:
    """Put the base center on the origin and scale the mesh uniformly into ``size``.

    One scale is applied to every axis, so the generated proportions stay intact.
    The mesh fits inside the requested box and touches it on the tightest axis.
    """
    array = np.asarray(vertices, dtype=np.float64)
    minimum = array.min(axis=0)
    maximum = array.max(axis=0)
    center = (minimum + maximum) / 2.0
    center[2] = minimum[2]
    shifted = array - center
    extent = np.maximum(maximum - minimum, 1e-6)
    scale = float(np.min(np.asarray(size, dtype=np.float64) / extent))
    return shifted * scale


def face_schema_forward(vertices: np.ndarray) -> np.ndarray:
    """Turn the photographed side from schema -Y to +Y.

    Pixal3D writes the front toward schema -Y. A stored character faces +Y,
    which is body yaw 0.
    """
    turned = np.array(vertices, dtype=np.float64, copy=True)
    turned[:, 0] *= -1.0
    turned[:, 1] *= -1.0
    return turned


def write_schema_glb(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray | None = None,
) -> None:
    """Write schema-space triangles as a Y-up GLB. ``colors`` are RGB in 0..1."""
    gltf_vertices = schema_points_to_gltf(vertices)
    _write_glb(path, gltf_vertices, faces, colors)


def read_vertex_colors(path: Path) -> np.ndarray | None:
    """RGB per vertex from COLOR_0, or None when the mesh has no plate color."""
    raw = path.read_bytes()
    if raw[:4] != b"glTF":
        return None
    document, blob = _parse_glb(raw)
    primitive = document["meshes"][0]["primitives"][0]
    color_index = primitive.get("attributes", {}).get("COLOR_0")
    if color_index is None:
        return None
    accessor = document["accessors"][color_index]
    width = {"VEC3": 3, "VEC4": 4}[accessor["type"]]
    values = _read_accessor(blob, document["bufferViews"], accessor)
    colors = values.reshape(-1, width)[:, :3]
    if len(colors) == 0:
        return None
    return np.clip(colors, 0.0, 1.0)


def read_schema_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a mesh file and return schema-space vertices and triangle indices."""
    vertices, faces = read_gltf_mesh(path)
    return gltf_points_to_schema(vertices), faces


def read_gltf_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".obj":
        return _read_obj_as_y_up(path)
    raw = path.read_bytes()
    if raw[:4] == b"glTF":
        document, blob = _parse_glb(raw)
        if document.get("asset", {}).get("generator") == GENERATOR:
            return _mesh_from_glb(document, blob)
    try:
        import trimesh
    except ImportError as exc:
        if raw[:4] == b"glTF":
            document, blob = _parse_glb(raw)
            return _mesh_from_glb(document, blob)
        raise RuntimeError(
            "Reading this model needs trimesh. Run `pnpm run content:asset-deps`."
        ) from exc
    try:
        loaded = trimesh.load(path, force="mesh", process=False, skip_materials=True)
    except TypeError:
        loaded = trimesh.load(path, force="mesh", process=False)
    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    if len(faces) == 0:
        raise ValueError(f"{path} has no triangles")
    return vertices, faces


def _read_obj_as_y_up(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices = []
    faces = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("v "):
            parts = line.split()
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif line.startswith("f "):
            indexes = []
            for part in line.split()[1:]:
                indexes.append(int(part.split("/")[0]) - 1)
            for index in range(1, len(indexes) - 1):
                faces.append((indexes[0], indexes[index], indexes[index + 1]))
    if not faces:
        raise ValueError(f"{path} has no faces")
    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int64)


def _write_glb(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray | None = None,
) -> None:
    vertex_data = np.ascontiguousarray(vertices, dtype=np.float32)
    index_data = np.ascontiguousarray(faces.reshape(-1), dtype=np.uint32)
    pieces = [vertex_data.tobytes(), index_data.tobytes()]
    color_data = None
    if colors is not None:
        color_data = np.ascontiguousarray(np.clip(colors, 0.0, 1.0), dtype=np.float32)
        if color_data.shape != (len(vertex_data), 3):
            raise ValueError(
                f"Vertex colors must be {(len(vertex_data), 3)}, got {color_data.shape}"
            )
        pieces.append(color_data.tobytes())
    blob = b"".join(pieces)
    minimum = vertex_data.min(axis=0).tolist()
    maximum = vertex_data.max(axis=0).tolist()
    attributes = {"POSITION": 0}
    accessors = [
        {
            "bufferView": 0,
            "componentType": 5126,
            "count": int(len(vertex_data)),
            "type": "VEC3",
            "min": minimum,
            "max": maximum,
        },
        {
            "bufferView": 1,
            "componentType": 5125,
            "count": int(len(index_data)),
            "type": "SCALAR",
        },
    ]
    buffer_views = [
        {
            "buffer": 0,
            "byteOffset": 0,
            "byteLength": int(vertex_data.nbytes),
            "target": 34962,
        },
        {
            "buffer": 0,
            "byteOffset": int(vertex_data.nbytes),
            "byteLength": int(index_data.nbytes),
            "target": 34963,
        },
    ]
    if color_data is not None:
        attributes["COLOR_0"] = 2
        accessors.append(
            {
                "bufferView": 2,
                "componentType": 5126,
                "count": int(len(color_data)),
                "type": "VEC3",
            }
        )
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": int(vertex_data.nbytes + index_data.nbytes),
                "byteLength": int(color_data.nbytes),
                "target": 34962,
            }
        )
    document = {
        "asset": {"version": "2.0", "generator": GENERATOR},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": attributes, "indices": 1}]}],
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(blob)}],
    }
    json_chunk = _pad(json.dumps(document, separators=(",", ":")).encode("utf-8"), b" ")
    bin_chunk = _pad(blob, b"\x00")
    chunks = _chunk(b"JSON", json_chunk) + _chunk(b"BIN\x00", bin_chunk)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks)


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack("<I4s", len(data), kind) + data


def _pad(data: bytes, pad: bytes) -> bytes:
    extra = (4 - (len(data) % 4)) % 4
    return data + pad * extra


def _parse_glb(raw: bytes) -> tuple[dict, bytes]:
    magic, version, _length = struct.unpack_from("<4sII", raw, 0)
    if magic != b"glTF" or version != 2:
        raise ValueError("Not a glTF 2 GLB")
    offset = 12
    document = None
    blob = b""
    while offset + 8 <= len(raw):
        chunk_length, chunk_type = struct.unpack_from("<I4s", raw, offset)
        offset += 8
        chunk = raw[offset : offset + chunk_length]
        offset += chunk_length
        if chunk_type == b"JSON":
            document = json.loads(chunk.decode("utf-8").strip())
        elif chunk_type == b"BIN\x00":
            blob = chunk
    if document is None:
        raise ValueError("GLB is missing JSON")
    return document, blob


def _mesh_from_glb(document: dict, blob: bytes) -> tuple[np.ndarray, np.ndarray]:
    accessors = document["accessors"]
    views = document["bufferViews"]
    primitive = document["meshes"][0]["primitives"][0]
    vertices = _read_accessor(blob, views, accessors[primitive["attributes"]["POSITION"]])
    indexes = _read_accessor(blob, views, accessors[primitive["indices"]])
    faces = indexes.astype(np.int64).reshape(-1, 3)
    return vertices.reshape(-1, 3), faces


def _read_accessor(blob: bytes, views: list[dict], accessor: dict) -> np.ndarray:
    view = views[accessor["bufferView"]]
    offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    component_type = accessor["componentType"]
    component = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}[
        component_type
    ]
    width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[accessor["type"]]
    count = int(accessor["count"]) * width
    data = blob[offset : offset + count * np.dtype(component).itemsize]
    values = np.frombuffer(data, dtype=component).astype(np.float64)
    if accessor.get("normalized") and component_type == 5121:
        values = values / 255.0
    return values
