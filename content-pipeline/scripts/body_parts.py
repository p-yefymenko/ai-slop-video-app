"""Body parts are the regions of a character mesh. The enum lives in script.ts; keep this list identical."""

from __future__ import annotations

import numpy as np

BODY_PARTS = (
    "hair",
    "face",
    "eyes",
    "neck",
    "torso",
    "arms",
    "hands",
    "legs",
    "feet",
)

# Standing-height bands. Meshes are upright, fitted to height, feet at z=0, front +Y.
_FEET = 0.07
_LEGS = 0.48
_HANDS_LO = 0.42
_HANDS_HI = 0.52
_TORSO = 0.76
_NECK = 0.82
_EYES_LO = 0.86
_EYES_HI = 0.91
_FACE = 0.92
_ARM_X = 0.55
_HAND_X = 0.50

# Viewer debug colors, 0-1 RGB. Keep in sync with content-pipeline/viewer/src/stage.js.
PART_DEBUG_COLORS = {
    "hair": (0.77, 0.24, 0.24),
    "face": (0.94, 0.78, 0.47),
    "eyes": (0.24, 0.47, 0.94),
    "neck": (0.82, 0.47, 0.78),
    "torso": (0.24, 0.71, 0.31),
    "arms": (0.94, 0.63, 0.16),
    "hands": (0.63, 0.31, 0.16),
    "legs": (0.31, 0.63, 0.78),
    "feet": (0.78, 0.78, 0.31),
}


def part_ids_for_vertices(vertices: np.ndarray) -> np.ndarray:
    """Index into BODY_PARTS per vertex. Local schema space: Z up, +Y front, feet at z=0.

    Right is +X. Face and eyes are the front half of the head band, so the back
    of the head is hair from every camera.
    """
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        return np.zeros((0,), dtype=np.int16)
    t = points[:, 2] / max(float(points[:, 2].max()), 1e-6)
    abs_x = np.abs(points[:, 0]) / max(float(np.max(np.abs(points[:, 0]))), 1e-6)
    index = {name: number for number, name in enumerate(BODY_PARTS)}
    ids = np.full(len(points), index["torso"], dtype=np.int16)
    ids[t < _FEET] = index["feet"]
    legs = (t >= _FEET) & (t < _LEGS)
    ids[legs] = index["legs"]
    ids[legs & (t >= _HANDS_LO) & (t < _HANDS_HI) & (abs_x >= _HAND_X)] = index["hands"]
    torso = (t >= _LEGS) & (t < _TORSO)
    ids[torso] = index["torso"]
    ids[torso & (abs_x >= _ARM_X)] = index["arms"]
    ids[(t >= _TORSO) & (t < _NECK)] = index["neck"]
    head = t >= _NECK
    ids[head] = index["hair"]
    if np.any(head):
        forward = points[head, 1]
        front = head & (points[:, 1] >= 0.5 * (float(forward.min()) + float(forward.max())))
        ids[front & (t < _FACE)] = index["face"]
        ids[front & (t >= _EYES_LO) & (t < _EYES_HI)] = index["eyes"]
    return ids
