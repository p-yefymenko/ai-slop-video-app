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


# A standing body's height spread is several times its width spread.
LONG_AXIS_RATIO = 2.0


def stand_upright(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Turn a standing body so its long axis is vertical, keeping which way it faces.

    Pixal3D builds the body in the frame of the camera it estimates for the
    plate, which looks slightly down at the person, so the mesh leans by that
    angle. The plate is always a person standing straight with arms at their
    sides, so the body's long axis (area-weighted, so dense hair does not pull
    it) is head to feet. The turn is about a horizontal axis: yaw is unchanged.
    """
    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    a, b, c = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2.0
    centers = (a + b + c) / 3.0
    middle = (centers * area[:, None]).sum(axis=0) / area.sum()
    spread = ((centers - middle) * area[:, None]).T @ (centers - middle)
    values, vectors = np.linalg.eigh(spread)
    if values[-1] < LONG_AXIS_RATIO * values[-2]:
        return points.copy()  # no clear long axis (not a standing body): nothing to stand up
    axis = vectors[:, -1]
    if axis[2] < 0:
        axis = -axis
    up = np.array([0.0, 0.0, 1.0])
    turn = np.cross(axis, up)
    sine, cosine = np.linalg.norm(turn), float(axis @ up)
    if sine < 1e-9:
        return points.copy()
    k = turn / sine
    cross = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    rotation = np.eye(3) + sine * cross + (1 - cosine) * cross @ cross
    return (points - middle) @ rotation.T + middle


def cap_holes(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    max_perimeter: float = 0.04,
    max_vertices: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Cover a small opening with new triangles. Existing vertices stay where they are.

    The cap is a fan to one vertex already on that opening. An opening wider than
    ``max_perimeter`` meters, such as the gap under an arm, is left alone.
    """
    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if len(triangles) == 0:
        return points, triangles
    directed = _boundary_edges(triangles)
    if not directed:
        return points, triangles
    added: list[tuple[int, int, int]] = []
    occupied = {tuple(sorted(int(index) for index in face[:3])) for face in triangles}
    for edges in _boundary_components(directed):
        loop = sorted({vertex for edge in edges for vertex in edge})
        if len(loop) < 3 or len(loop) > max_vertices:
            continue
        perimeter = 0.0
        for src, tgt in edges:
            perimeter += float(np.linalg.norm(points[tgt] - points[src]))
        if perimeter >= max_perimeter:
            continue
        if len(loop) == 3:
            src, tgt = edges[0]
            other = next(vertex for vertex in loop if vertex != src and vertex != tgt)
            _add_cap(points, occupied, added, (tgt, src, other))
            continue
        degree: dict[int, int] = {}
        for src, tgt in edges:
            degree[src] = degree.get(src, 0) + 1
            degree[tgt] = degree.get(tgt, 0) + 1
        endpoints = [vertex for vertex in loop if degree.get(vertex) == 1]
        apex = endpoints[0] if endpoints else loop[0]
        for src, tgt in edges:
            if src == apex or tgt == apex:
                continue
            _add_cap(points, occupied, added, (tgt, src, apex))
    if not added:
        return points, triangles
    return points, np.vstack((triangles, np.asarray(added, dtype=np.int64)))


# A standing person is not as wide as they are tall. Capes that stay wide
# across many slices are left alone; this only flags a mesh the fit could
# not make into a person.
STANDING_WIDTH_RATIO = 0.7


def drop_thin_side_protrusions(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int, int]:
    """Drop a thin sideways sheet that is welded to the body.

    Width is measured per height slice in the mesh's own axes (schema X
    across, Z up). A slice is a sheet when it is much wider than the slices
    around it and that excess is only a few slices tall. Vertices in that
    slice that sit outside the neighboring body are removed, and so are the
    triangles that use them. A cape or pauldron that is wide for a long
    stretch of the body stays.
    """
    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    kept_colors = None if colors is None else np.asarray(colors)
    before_vertices = len(points)
    before_triangles = len(triangles)
    if before_vertices == 0 or before_triangles == 0:
        return points, triangles, kept_colors, 0, 0
    height_axis = points[:, 2]
    height = float(height_axis.max() - height_axis.min())
    if height < 1e-4:
        return points, triangles, kept_colors, 0, 0
    slice_count = 80
    edges = np.linspace(float(height_axis.min()), float(height_axis.max()), slice_count + 1)
    bucket = np.clip(np.digitize(height_axis, edges) - 1, 0, slice_count - 1)
    widths = np.zeros(slice_count)
    occupied = np.zeros(slice_count, dtype=bool)
    for index in range(slice_count):
        sample = points[bucket == index, 0]
        if len(sample) < 8:
            continue
        occupied[index] = True
        widths[index] = float(sample.max() - sample.min())
    spike = np.zeros(slice_count, dtype=bool)
    for index in range(slice_count):
        if not occupied[index]:
            continue
        nearby = [
            widths[other]
            for other in range(max(0, index - 10), min(slice_count, index + 11))
            if abs(other - index) >= 2 and occupied[other]
        ]
        if len(nearby) < 4:
            continue
        local_width = float(np.median(nearby))
        if local_width > 0 and widths[index] > local_width * 1.4:
            spike[index] = True
    start = 0
    while start < slice_count:
        if not spike[start]:
            start += 1
            continue
        end = start
        while end < slice_count and spike[end]:
            end += 1
        if (end - start) / slice_count > 0.08:
            spike[start:end] = False
        start = end
    if not spike.any():
        return points, triangles, kept_colors, 0, 0
    remove = np.zeros(before_vertices, dtype=bool)
    for index in np.flatnonzero(spike):
        neighbors = _neighbor_indices(occupied, spike, index, slice_count)
        if not neighbors:
            continue
        chosen = bucket == index
        remove[chosen] = ~_on_neighbor_surface(points, bucket, chosen, neighbors)
    if not remove.any():
        return points, triangles, kept_colors, 0, 0
    keep = ~remove
    remap = np.full(before_vertices, -1, dtype=np.int64)
    remap[keep] = np.arange(int(keep.sum()), dtype=np.int64)
    surviving = keep[triangles].all(axis=1)
    points = points[keep]
    triangles = remap[triangles[surviving]]
    if kept_colors is not None and len(kept_colors) == before_vertices:
        kept_colors = kept_colors[keep]
    return (
        points,
        triangles,
        kept_colors,
        before_vertices - len(points),
        before_triangles - len(triangles),
    )


def _neighbor_indices(
    occupied: np.ndarray,
    spike: np.ndarray,
    index: int,
    slice_count: int,
) -> list[int]:
    chosen: list[int] = []
    for delta in range(1, 12):
        for other in (index - delta, index + delta):
            if 0 <= other < slice_count and occupied[other] and not spike[other]:
                chosen.append(other)
        if len(chosen) >= 2:
            break
    return chosen


def _on_neighbor_surface(
    points: np.ndarray,
    bucket: np.ndarray,
    chosen: np.ndarray,
    neighbors: list[int],
) -> np.ndarray:
    """True where a spike-slice vertex sits on the body surface around it.

    The sheet through the shoulders has no matching surface in the slices
    above and below, so those vertices fall through. A shoulder that simply
    continues the torso stays.
    """
    sample = points[np.isin(bucket, neighbors)]
    here = points[chosen]
    reach = 0.08 * float(max(sample[:, 0].max() - sample[:, 0].min(), 1e-4))
    origin = sample[:, :2].min(axis=0)
    keys = np.floor((sample[:, :2] - origin) / reach).astype(np.int32)
    occupied_cells = set(zip(keys[:, 0].tolist(), keys[:, 1].tolist()))
    here_keys = np.floor((here[:, :2] - origin) / reach).astype(np.int32)
    keep = np.zeros(len(here), dtype=bool)
    for row, (x_cell, y_cell) in enumerate(zip(here_keys[:, 0].tolist(), here_keys[:, 1].tolist())):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if (x_cell + dx, y_cell + dy) in occupied_cells:
                    keep[row] = True
                    break
            if keep[row]:
                break
    return keep


def standing_proportion_warning(character_id: str, vertices: np.ndarray) -> str | None:
    """Warn when a fitted character is too wide to be a standing person."""
    points = np.asarray(vertices, dtype=np.float64)
    if len(points) == 0:
        return None
    extent = points.max(axis=0) - points.min(axis=0)
    height = float(extent[2])
    if height < 1e-4:
        return None
    width = float(max(extent[0], extent[1]))
    if width / height <= STANDING_WIDTH_RATIO:
        return None
    return (
        f"{character_id}: mesh is {width:.2f} m wide and {height:.2f} m tall, "
        f"which is too wide for a standing person"
    )


def _add_cap(
    points: np.ndarray,
    occupied: set[tuple[int, int, int]],
    added: list[tuple[int, int, int]],
    face: tuple[int, int, int],
) -> None:
    if len(set(face)) < 3:
        return
    key = tuple(sorted(face))
    if key in occupied:
        return
    apex, src, tgt = face[2], face[1], face[0]
    span = np.cross(points[src] - points[apex], points[tgt] - points[apex])
    if float(np.dot(span, span)) < 1e-16:
        return
    occupied.add(key)
    added.append(face)


def _boundary_edges(faces: np.ndarray) -> list[tuple[int, int]]:
    counts: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face in faces:
        a, b, c = (int(index) for index in face[:3])
        for src, tgt in ((a, b), (b, c), (c, a)):
            key = (src, tgt) if src < tgt else (tgt, src)
            counts.setdefault(key, []).append((src, tgt))
    return [directed[0] for directed in counts.values() if len(directed) == 1]


def _boundary_components(edges: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    parent: dict[int, int] = {}

    def find(node: int) -> int:
        root = node
        while parent.get(root, root) != root:
            root = parent.get(root, root)
        while node != root:
            nxt = parent.get(node, node)
            parent[node] = root
            node = nxt
        return root

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    for src, tgt in edges:
        union(src, tgt)
    groups: dict[int, list[tuple[int, int]]] = {}
    for src, tgt in edges:
        groups.setdefault(find(src), []).append((src, tgt))
    return list(groups.values())


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
