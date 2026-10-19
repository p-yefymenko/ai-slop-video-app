"""A character's mesh painted from the photo it was made from.

Pixal3D builds the mesh pixel-aligned to the square crop of the reference photo
it was given (``ImageCropToMask``), seen by a pinhole camera of the crop's field
of view (MoGe's), on the axis in front of the mesh at the distance where one
unit spans the crop's width. The mesh's own colors come out of its texture
stage blurred and stained (a white chin, dark specks, dull fur), and an image
model drawing the start still copies them. Every vertex that camera sees is
given the crop's pixel instead; the rest (sides, back) keep the mesh's colors.

The raw mesh in schema space is Pixal3D's projection frame: ``coords.gltf_to_schema``
maps glTF (x, y, z) to (x, -z, y), which is the frame its projection uses.
"""

from __future__ import annotations

import math

import numpy as np

# Pixal3D decodes its mesh from a 1024-voxel grid across one unit. A vertex on the surface the
# camera sees lies within this depth of that surface where it crosses the vertex's pixel: the
# surface's depth changes by up to about two voxels across one pixel where it slants away.
VISIBLE_DEPTH_TOLERANCE = 2.0 / 1024
# The renderer encodes linear vertex colors with this gamma; a photo's pixels are decoded with it.
DISPLAY_GAMMA = 2.2


def camera_distance(fov_degrees: float) -> float:
    """How far in front of the mesh's center Pixal3D's camera stands: one unit spans the crop."""
    return 0.5 / math.tan(math.radians(fov_degrees) / 2.0)


def crop_camera(fov_degrees: float) -> dict:
    """Pixal3D's camera as a ``render`` camera, in schema space: on -y, looking at the center."""
    return {"position": [0.0, -camera_distance(fov_degrees), 0.0], "lookAt": [0.0, 0.0, 0.0], "verticalFovDegrees": float(fov_degrees)}


def project(vertices: np.ndarray, fov_degrees: float, resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each schema-space vertex's (column, row) in the square crop and its depth from the camera."""
    distance = camera_distance(fov_degrees)
    focal = resolution / (2.0 * math.tan(math.radians(fov_degrees) / 2.0))
    x, y, z = (np.asarray(vertices, dtype=np.float64)[:, axis] for axis in range(3))
    depth = y + distance
    column = focal * x / depth + resolution / 2.0
    row = -focal * z / depth + resolution / 2.0
    return column, row, depth


def paint_from_photo(vertices: np.ndarray, colors: np.ndarray, crop: np.ndarray, fov_degrees: float, surface_depth: np.ndarray) -> np.ndarray:
    """``colors`` (linear, 0-1) with every vertex the crop's camera sees given the crop's pixel.

    ``crop`` is the square sRGB image Pixal3D saw (uint8); ``surface_depth`` is the mesh rendered
    from its camera (``crop_camera``) at the crop's size, the depth of the nearest surface at each
    pixel (inf where there is none)."""
    resolution = crop.shape[0]
    column, row, depth = project(vertices, fov_degrees, resolution)
    col_i, row_i = np.floor(column).astype(np.int64), np.floor(row).astype(np.int64)
    inside = (col_i >= 0) & (col_i < resolution) & (row_i >= 0) & (row_i < resolution)
    seen = inside.copy()
    seen[inside] = depth[inside] <= surface_depth[row_i[inside], col_i[inside]] + VISIBLE_DEPTH_TOLERANCE
    painted = np.array(colors, dtype=np.float32, copy=True)
    photo = crop[row_i[seen], col_i[seen], :3].astype(np.float32) / 255.0
    painted[seen] = photo ** DISPLAY_GAMMA
    return painted


def surface_depth(vertices: np.ndarray, faces: np.ndarray, fov_degrees: float, resolution: int) -> np.ndarray:
    """The mesh rendered from the crop's camera: the depth of the nearest surface at each pixel."""
    from render import render
    from world import Entity, make_mesh

    mesh = make_mesh(f"photo-paint:{len(vertices)}:{fov_degrees}", np.asarray(vertices), np.asarray(faces))
    frame = render([Entity("photo", "landmark", mesh, (0.0, 0.0, 0.0), 0.0)], crop_camera(fov_degrees), resolution, resolution)
    return frame.depth
