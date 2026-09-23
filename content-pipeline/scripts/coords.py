"""Schema space is Z-up, +Y forward, meters. glTF is Y-up, and forward is -Z.

This module is the only place that converts between those spaces.
"""

from __future__ import annotations

import numpy as np

Vec3 = tuple[float, float, float]


def schema_to_gltf(point: Vec3) -> Vec3:
    """Map one schema point into glTF."""
    x, y, z = point
    return (float(x), float(z), float(-y))


def gltf_to_schema(point: Vec3) -> Vec3:
    """Map one glTF point back into schema space."""
    x, y, z = point
    return (float(x), float(-z), float(y))


def schema_points_to_gltf(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    converted = np.empty_like(array)
    converted[:, 0] = array[:, 0]
    converted[:, 1] = array[:, 2]
    converted[:, 2] = -array[:, 1]
    return converted


def gltf_points_to_schema(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    converted = np.empty_like(array)
    converted[:, 0] = array[:, 0]
    converted[:, 1] = -array[:, 2]
    converted[:, 2] = array[:, 1]
    return converted
