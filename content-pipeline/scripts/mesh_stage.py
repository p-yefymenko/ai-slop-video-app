"""Stand people on the open mesh and keep the camera out of it.

Script marks are written on an empty floor at z = 0. The fitted mesh has a
raised deck and solid objects. Landmark positions and the mesh bounds say
where those objects are, so a mark inside one is moved to open floor and the
camera is lifted and backed out of the stone.
"""

from __future__ import annotations

import math

import numpy as np

Vec3 = tuple[float, float, float]

CELL = 0.5
OBJECT_CLEARANCE = 1.15
BODY_CLEARANCE = 0.45
DECK_RISE = 0.65
CAMERA_AHEAD = 1.5


class MeshStage:
    """Walkable deck and solid cells for one location mesh."""

    def __init__(
        self,
        minimum: np.ndarray,
        maximum: np.ndarray,
        cell: float,
        deck_z: float,
        solid: np.ndarray,
        top: np.ndarray,
    ) -> None:
        self.minimum = minimum
        self.maximum = maximum
        self.cell = cell
        self.deck_z = deck_z
        self.solid = solid
        self.top = top

    def covers(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (
            float(self.minimum[0]) - margin <= x <= float(self.maximum[0]) + margin
            and float(self.minimum[1]) - margin <= y <= float(self.maximum[1]) + margin
        )

    def blocked(self, x: float, y: float, radius: float) -> bool:
        reach = max(0, int(math.ceil(radius / self.cell)))
        center = self._index(x, y)
        if center is None:
            return False
        iy, ix = center
        height, width = self.solid.shape
        for row in range(iy - reach, iy + reach + 1):
            for col in range(ix - reach, ix + reach + 1):
                if row < 0 or col < 0 or row >= height or col >= width:
                    continue
                if not self.solid[row, col]:
                    continue
                cx, cy = self._center(col, row)
                if math.hypot(cx - x, cy - y) <= radius + self.cell * 0.5:
                    return True
        return False

    def place_feet(self, position: Vec3, avoid: list[Vec3] | None = None) -> Vec3:
        """Feet on the deck, outside landmarks and other raised mesh."""
        x, y, z = (float(position[0]), float(position[1]), float(position[2]))
        if not self.covers(x, y, margin=self.cell):
            x, y = self._clamp(x, y)
        if self._feet_blocked(x, y, avoid):
            found = self._nearest_open(x, y, avoid)
            if found is not None:
                x, y = found
        return (x, y, self.deck_z + z + 0.02)

    def place_camera(
        self, position: Vec3, look_at: Vec3, vertical_fov: float = 40.0
    ) -> tuple[Vec3, Vec3]:
        """Keep an interior lens above the deck and out of solid mesh.

        A camera already outside the mesh bounds is an exterior and stays put.
        """
        if not self.covers(float(position[0]), float(position[1]), margin=0.25):
            return (
                (float(position[0]), float(position[1]), float(position[2])),
                (float(look_at[0]), float(look_at[1]), float(look_at[2])),
            )
        if float(position[2]) > float(self.maximum[2]) + 0.5:
            eye = float(position[2])
            gaze = float(look_at[2])
        else:
            eye = self.deck_z + float(position[2])
            gaze = self.deck_z + float(look_at[2])
        look_feet = self.place_feet((float(look_at[0]), float(look_at[1]), 0.0))
        offset_x = float(position[0]) - float(look_at[0])
        offset_y = float(position[1]) - float(look_at[1])
        pos = [look_feet[0] + offset_x, look_feet[1] + offset_y, eye]
        look = [look_feet[0], look_feet[1], gaze]
        pos[0], pos[1], raised = self._clear_view(
            pos[0], pos[1], look[0], look[1], eye, vertical_fov
        )
        if raised is not None:
            pos[2] = raised
        forward_x = look[0] - pos[0]
        forward_y = look[1] - pos[1]
        length = math.hypot(forward_x, forward_y)
        if length < 1e-4:
            return (pos[0], pos[1], pos[2]), (look[0], look[1], look[2])
        forward_x /= length
        forward_y /= length
        for _step in range(48):
            if self._lens_clear(pos[0], pos[1], pos[2], forward_x, forward_y):
                break
            pos[0] -= forward_x * self.cell
            pos[1] -= forward_y * self.cell
        return (pos[0], pos[1], pos[2]), (look[0], look[1], look[2])

    def _feet_blocked(self, x: float, y: float, avoid: list[Vec3] | None) -> bool:
        if self.blocked(x, y, BODY_CLEARANCE):
            return True
        return _near_points(x, y, avoid, 0.7)

    def _clear_view(
        self,
        x: float,
        y: float,
        look_x: float,
        look_y: float,
        eye: float,
        vertical_fov: float,
    ) -> tuple[float, float, float | None]:
        """Slide a little, then dolly back until solid mesh fits in the frame."""
        if self._sight_clear(x, y, look_x, look_y, eye):
            return x, y, None
        forward_x = look_x - x
        forward_y = look_y - y
        length = math.hypot(forward_x, forward_y) or 1.0
        side_x = -forward_y / length
        side_y = forward_x / length
        for distance in (1.0, 2.0, 3.0, 4.5):
            for sign in (1.0, -1.0):
                candidate_x = x + side_x * sign * distance
                candidate_y = y + side_y * sign * distance
                if self.blocked(candidate_x, candidate_y, BODY_CLEARANCE):
                    continue
                if self._sight_clear(candidate_x, candidate_y, look_x, look_y, eye):
                    return candidate_x, candidate_y, None
        half = max(math.tan(math.radians(vertical_fov) / 2.0), 0.05)
        forward_x /= length
        forward_y /= length
        for step in range(1, 48):
            candidate_x = x - forward_x * step * self.cell
            candidate_y = y - forward_y * step * self.cell
            if self._tops_fit(candidate_x, candidate_y, look_x, look_y, eye, half):
                return candidate_x, candidate_y, None
        return x, y, self._path_peak(x, y, look_x, look_y) + 1.5

    def _tops_fit(
        self,
        x: float,
        y: float,
        look_x: float,
        look_y: float,
        eye: float,
        half_tan: float,
    ) -> bool:
        """True when solid tops between the lens and the look target sit inside the frame."""
        length = math.hypot(look_x - x, look_y - y)
        steps = max(1, int(length / self.cell))
        for step in range(steps):
            remaining = (1.0 - step / steps) * length
            if remaining < 0.8:
                continue
            sample_x = x + (look_x - x) * (step / steps)
            sample_y = y + (look_y - y) * (step / steps)
            top = self._top(sample_x, sample_y, 0.35)
            if top <= eye - 0.25:
                continue
            distance = max(math.hypot(sample_x - x, sample_y - y), 0.3)
            if (top - eye) / distance > half_tan * 0.85:
                return False
        return True

    def _sight_clear(self, x: float, y: float, look_x: float, look_y: float, eye: float) -> bool:
        length = math.hypot(look_x - x, look_y - y)
        if length < 1e-4:
            return self._top(x, y, 0.35) <= eye - 0.25
        steps = max(1, int(length / self.cell))
        for step in range(steps):
            remaining = (1.0 - step / steps) * length
            if remaining < 0.8:
                continue
            sample_x = x + (look_x - x) * (step / steps)
            sample_y = y + (look_y - y) * (step / steps)
            if self._top(sample_x, sample_y, 0.35) > eye - 0.25:
                return False
        return True

    def _path_peak(self, x: float, y: float, look_x: float, look_y: float) -> float:
        length = math.hypot(look_x - x, look_y - y)
        peak = self.deck_z
        steps = max(1, int(length / self.cell))
        for step in range(steps + 1):
            sample_x = x + (look_x - x) * (step / steps)
            sample_y = y + (look_y - y) * (step / steps)
            peak = max(peak, self._top(sample_x, sample_y, 0.35))
        return peak

    def _lens_clear(self, x: float, y: float, eye: float, forward_x: float, forward_y: float) -> bool:
        if self.blocked(x, y, BODY_CLEARANCE) and self._top(x, y, BODY_CLEARANCE) > eye - 0.25:
            return False
        samples = int(CAMERA_AHEAD / self.cell)
        for step in range(1, samples + 1):
            distance = step * self.cell
            ahead_x = x + forward_x * distance
            ahead_y = y + forward_y * distance
            if self._top(ahead_x, ahead_y, 0.35) > eye - 0.25:
                return False
        return True

    def _top(self, x: float, y: float, radius: float = 0.0) -> float:
        reach = max(0, int(math.ceil(radius / self.cell)))
        center = self._index(x, y)
        if center is None:
            return self.deck_z
        iy, ix = center
        height, width = self.solid.shape
        best = self.deck_z
        for row in range(iy - reach, iy + reach + 1):
            for col in range(ix - reach, ix + reach + 1):
                if row < 0 or col < 0 or row >= height or col >= width:
                    continue
                if not self.solid[row, col]:
                    continue
                cx, cy = self._center(col, row)
                if math.hypot(cx - x, cy - y) > radius + self.cell * 0.5:
                    continue
                value = float(self.top[row, col])
                if np.isfinite(value):
                    best = max(best, value)
        return best

    def _nearest_open(self, x: float, y: float, avoid: list[Vec3] | None) -> tuple[float, float] | None:
        start = self._index(x, y)
        if start is None:
            return None
        height, width = self.solid.shape
        best: tuple[float, float] | None = None
        best_distance = float("inf")
        rings = int(8.0 / self.cell)
        for ring in range(rings + 1):
            if best is not None and ring * self.cell > best_distance + self.cell:
                break
            for row in range(start[0] - ring, start[0] + ring + 1):
                for col in range(start[1] - ring, start[1] + ring + 1):
                    if row < 0 or col < 0 or row >= height or col >= width:
                        continue
                    if ring and max(abs(row - start[0]), abs(col - start[1])) != ring:
                        continue
                    cx, cy = self._center(col, row)
                    if self._feet_blocked(cx, cy, avoid):
                        continue
                    distance = math.hypot(cx - x, cy - y)
                    if distance < best_distance:
                        best = (cx, cy)
                        best_distance = distance
        return best

    def _clamp(self, x: float, y: float) -> tuple[float, float]:
        return (
            min(max(x, float(self.minimum[0])), float(self.maximum[0])),
            min(max(y, float(self.minimum[1])), float(self.maximum[1])),
        )

    def _index(self, x: float, y: float) -> tuple[int, int] | None:
        col = int((x - float(self.minimum[0])) / self.cell)
        row = int((y - float(self.minimum[1])) / self.cell)
        height, width = self.solid.shape
        if col < 0 or row < 0 or col >= width or row >= height:
            return None
        return row, col

    def _center(self, col: int, row: int) -> tuple[float, float]:
        return (
            float(self.minimum[0]) + (col + 0.5) * self.cell,
            float(self.minimum[1]) + (row + 0.5) * self.cell,
        )


def build_mesh_stage(vertices: np.ndarray, landmarks: dict | None) -> MeshStage:
    """Deck height and solid cells from mesh vertices plus named landmarks."""
    points = np.asarray(vertices, dtype=np.float64)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    span = np.maximum(maximum - minimum, 1e-6)
    width = max(1, int(math.ceil(span[0] / CELL)))
    height = max(1, int(math.ceil(span[1] / CELL)))
    columns = np.clip(((points[:, 0] - minimum[0]) / CELL).astype(np.int32), 0, width - 1)
    rows = np.clip(((points[:, 1] - minimum[1]) / CELL).astype(np.int32), 0, height - 1)
    top = np.full(height * width, -np.inf, dtype=np.float64)
    count = np.zeros(height * width, dtype=np.int32)
    flat = rows * width + columns
    np.maximum.at(top, flat, points[:, 2])
    np.add.at(count, flat, 1)
    top = top.reshape(height, width)
    count = count.reshape(height, width)
    occupied = count >= 1
    if np.any(occupied):
        deck_z = float(np.percentile(top[occupied], 40))
    else:
        deck_z = float(minimum[2])
    solid = occupied & (top > deck_z + DECK_RISE)
    _paint_landmarks(solid, minimum, landmarks or {}, CELL)
    return MeshStage(minimum, maximum, CELL, deck_z, solid, top)


def _paint_landmarks(solid: np.ndarray, minimum: np.ndarray, landmarks: dict, cell: float) -> None:
    height, width = solid.shape
    reach = max(1, int(math.ceil(OBJECT_CLEARANCE / cell)))
    for landmark in landmarks.values():
        if not isinstance(landmark, dict) or landmark.get("position") is None:
            continue
        x, y = float(landmark["position"][0]), float(landmark["position"][1])
        col = int((x - float(minimum[0])) / cell)
        row = int((y - float(minimum[1])) / cell)
        for iy in range(row - reach, row + reach + 1):
            for ix in range(col - reach, col + reach + 1):
                if iy < 0 or ix < 0 or iy >= height or ix >= width:
                    continue
                cx = float(minimum[0]) + (ix + 0.5) * cell
                cy = float(minimum[1]) + (iy + 0.5) * cell
                if math.hypot(cx - x, cy - y) <= OBJECT_CLEARANCE:
                    solid[iy, ix] = True


def _near_points(x: float, y: float, points: list[Vec3] | None, radius: float) -> bool:
    if not points:
        return False
    return any(math.hypot(x - point[0], y - point[1]) < radius for point in points)
