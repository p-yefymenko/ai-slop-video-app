"""Triangle meshes stored as glTF Y-up GLB files, without textures."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from coords import gltf_points_to_schema, schema_points_to_gltf

GENERATOR = "reelshort-content-pipeline"
# Clay previs walks every triangle in Python. Downloaded scans stay under this.
TRIANGLE_BUDGET = 25_000


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


def primitive_mesh(kind: str, size: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Last-resort stand-in for one landmark or prop. Schema space, base at the origin."""
    if kind == "column":
        return _prism(size, sides=8)
    if kind == "pedestal":
        return _stack_boxes(
            (
                ((size[0], size[1], size[2] * 0.35), 0.0),
                ((size[0] * 0.62, size[1] * 0.62, size[2] * 0.65), size[2] * 0.35),
            )
        )
    if kind == "seat":
        return _stack_boxes(
            (
                ((size[0], size[1], size[2] * 0.45), 0.0),
                ((size[0], size[1] * 0.16, size[2] * 0.55), size[2] * 0.45),
            ),
            back_offset=size[1] * 0.42,
        )
    if kind == "door":
        return box_mesh((size[0] * 0.92, min(size[1], 0.12), size[2]))
    if kind == "window":
        return _window(size)
    return box_mesh(size)


def fit_to_size(
    vertices: np.ndarray, size: tuple[float, float, float]
) -> np.ndarray:
    """Put the base center on the origin and stretch the mesh onto ``size``."""
    array = np.asarray(vertices, dtype=np.float64)
    minimum = array.min(axis=0)
    maximum = array.max(axis=0)
    center = (minimum + maximum) / 2.0
    center[2] = minimum[2]
    shifted = array - center
    extent = np.maximum(maximum - minimum, 1e-6)
    scale = np.array(size, dtype=np.float64) / extent
    return shifted * scale


def proportions_match(
    extent: np.ndarray,
    size: tuple[float, float, float],
    low: float = 0.35,
    high: float = 2.85,
) -> bool:
    """Compare sorted axis ratios so a long prop is not forced into a cube."""
    actual = np.sort(np.maximum(np.asarray(extent, dtype=np.float64), 1e-6))
    target = np.sort(np.maximum(np.array(size, dtype=np.float64), 1e-6))
    actual = actual / actual.max()
    target = target / target.max()
    ratio = actual / target
    return bool(np.all((ratio >= low) & (ratio <= high)))


def decimate(
    vertices: np.ndarray,
    faces: np.ndarray,
    budget: int = TRIANGLE_BUDGET,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a mesh until it has at most ``budget`` triangles.

    Vertex clustering keeps the silhouette. A stride of the original faces is
    the fallback when clustering cannot get under the budget.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(faces) <= budget:
        return vertices, faces
    best: tuple[np.ndarray, np.ndarray] | None = None
    low = 2
    high = 128
    while low <= high:
        divisions = (low + high) // 2
        clustered_vertices, clustered_faces = _cluster_vertices(vertices, faces, divisions)
        if 0 < len(clustered_faces) <= budget:
            best = (clustered_vertices, clustered_faces)
            low = divisions + 1
        else:
            high = divisions - 1
    if best is not None:
        return _compact(best[0], best[1])
    step = int(np.ceil(len(faces) / budget))
    return _compact(vertices, np.ascontiguousarray(faces[::step]))


def _cluster_vertices(
    vertices: np.ndarray, faces: np.ndarray, divisions: int
) -> tuple[np.ndarray, np.ndarray]:
    minimum = vertices.min(axis=0)
    extent = np.maximum(vertices.max(axis=0) - minimum, 1e-9)
    quantized = np.floor((vertices - minimum) / extent * divisions).astype(np.int64)
    quantized = np.clip(quantized, 0, divisions - 1)
    keys = (
        quantized[:, 0]
        + quantized[:, 1] * divisions
        + quantized[:, 2] * divisions * divisions
    )
    _unique, inverse = np.unique(keys, return_inverse=True)
    count = int(inverse.max()) + 1 if len(inverse) else 0
    if count == 0:
        return vertices[:0], faces[:0]
    clustered = np.zeros((count, 3), dtype=np.float64)
    weights = np.zeros(count, dtype=np.float64)
    np.add.at(clustered, inverse, vertices)
    np.add.at(weights, inverse, 1.0)
    clustered /= weights[:, None]
    remapped = inverse[faces]
    keep = (
        (remapped[:, 0] != remapped[:, 1])
        & (remapped[:, 1] != remapped[:, 2])
        & (remapped[:, 0] != remapped[:, 2])
    )
    return clustered, remapped[keep]


def _compact(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(faces) == 0:
        return vertices[:0], faces
    used = np.unique(faces.reshape(-1))
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return np.ascontiguousarray(vertices[used]), np.ascontiguousarray(remap[faces])


def write_schema_glb(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write schema-space triangles as a Y-up GLB."""
    gltf_vertices = schema_points_to_gltf(vertices)
    _write_glb(path, gltf_vertices, faces)


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


def schema_triangles(vertices: np.ndarray, faces: np.ndarray) -> list[tuple[tuple[float, float, float], ...]]:
    output = []
    for face in faces:
        output.append(tuple(tuple(float(value) for value in vertices[int(index)]) for index in face[:3]))
    return output


def _prism(size: tuple[float, float, float], sides: int) -> tuple[np.ndarray, np.ndarray]:
    radius = min(size[0], size[1]) / 2.0
    height = size[2]
    angles = np.linspace(0.0, np.pi * 2.0, sides, endpoint=False)
    ring = np.stack((np.cos(angles) * radius, np.sin(angles) * radius), axis=1)
    bottom = np.column_stack((ring, np.zeros(sides)))
    top = np.column_stack((ring, np.full(sides, height)))
    vertices = np.vstack((bottom, top))
    faces = []
    for index in range(1, sides - 1):
        faces.append((0, index, index + 1))
        faces.append((sides, sides + index + 1, sides + index))
    for index in range(sides):
        nxt = (index + 1) % sides
        faces.append((index, nxt, sides + nxt))
        faces.append((index, sides + nxt, sides + index))
    return vertices, np.array(faces, dtype=np.int64)


def _stack_boxes(
    parts: tuple[tuple[tuple[float, float, float], float], ...],
    back_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    vertices = []
    faces = []
    for part_index, (part_size, z_offset) in enumerate(parts):
        part_vertices, part_faces = box_mesh(part_size)
        part_vertices = part_vertices.copy()
        part_vertices[:, 2] += z_offset
        if part_index == 1 and back_offset:
            part_vertices[:, 1] += back_offset
        base = len(vertices)
        vertices.extend(part_vertices.tolist())
        faces.extend((base + int(a), base + int(b), base + int(c)) for a, b, c in part_faces)
    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int64)


def _window(size: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    width, depth, height = size
    thickness = max(min(depth, 0.12), width * 0.08)
    frame = max(min(width, height) * 0.16, 0.05)
    parts = (
        ((width, thickness, frame), 0.0),
        ((width, thickness, frame), height - frame),
        ((frame, thickness, height), 0.0),
    )
    vertices, faces = _stack_boxes(parts[:2])
    side, side_faces = box_mesh((frame, thickness, height))
    left = side.copy()
    left[:, 0] -= width / 2.0 - frame / 2.0
    right = side.copy()
    right[:, 0] += width / 2.0 - frame / 2.0
    base = len(vertices)
    vertices = np.vstack((vertices, left, right))
    extra = np.vstack((side_faces + base, side_faces + base + len(side)))
    return vertices, np.vstack((faces, extra))


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


def _write_glb(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    vertex_data = np.ascontiguousarray(vertices, dtype=np.float32)
    index_data = np.ascontiguousarray(faces.reshape(-1), dtype=np.uint32)
    blob = vertex_data.tobytes() + index_data.tobytes()
    minimum = vertex_data.min(axis=0).tolist()
    maximum = vertex_data.max(axis=0).tolist()
    document = {
        "asset": {"version": "2.0", "generator": GENERATOR},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "accessors": [
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
        ],
        "bufferViews": [
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
        ],
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
    component = {5126: np.float32, 5125: np.uint32, 5123: np.uint16}[accessor["componentType"]]
    width = {"SCALAR": 1, "VEC3": 3}[accessor["type"]]
    count = int(accessor["count"]) * width
    data = blob[offset : offset + count * np.dtype(component).itemsize]
    return np.frombuffer(data, dtype=component).astype(np.float64)
