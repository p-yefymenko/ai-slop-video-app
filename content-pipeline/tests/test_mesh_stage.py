from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from mesh_stage import build_mesh_stage  # noqa: E402


def _slab_and_column() -> np.ndarray:
    floor = np.stack(
        np.meshgrid(np.linspace(-5.0, 5.0, 21), np.linspace(-5.0, 5.0, 21), [0.0, 1.0]),
        axis=-1,
    ).reshape(-1, 3)
    column = np.stack(
        np.meshgrid(
            np.linspace(-0.4, 0.4, 5),
            np.linspace(-0.4, 0.4, 5),
            np.linspace(1.2, 3.8, 6),
        ),
        axis=-1,
    ).reshape(-1, 3)
    return np.vstack((floor, column))


class MeshStageTests(unittest.TestCase):
    def test_feet_leave_a_column_and_stand_on_the_deck(self) -> None:
        stage = build_mesh_stage(
            _slab_and_column(),
            {"pillar": {"position": [0.0, 0.0, 2.5]}},
        )
        self.assertAlmostEqual(stage.deck_z, 1.0, delta=0.2)
        moved = stage.place_feet((0.0, 0.0, 0.0))
        self.assertGreater(abs(moved[0]) + abs(moved[1]), 0.8)
        self.assertAlmostEqual(moved[2], stage.deck_z + 0.02, delta=0.05)
        open_feet = stage.place_feet((3.0, 0.0, 0.0))
        self.assertAlmostEqual(open_feet[0], 3.0, delta=0.3)
        self.assertGreater(open_feet[2], 0.9)

    def test_interior_camera_backs_out_of_the_column(self) -> None:
        stage = build_mesh_stage(_slab_and_column(), {})
        position, look_at = stage.place_camera((0.0, 0.0, 1.6), (0.0, 3.0, 1.4))
        self.assertGreater(math.hypot(position[0], position[1]), 0.8)
        self.assertAlmostEqual(position[2], stage.deck_z + 1.6, delta=0.05)
        self.assertAlmostEqual(look_at[2], stage.deck_z + 1.4, delta=0.05)

    def test_exterior_camera_stays_outside_the_mesh(self) -> None:
        stage = build_mesh_stage(_slab_and_column(), {})
        position, look_at = stage.place_camera((0.0, -20.0, 8.0), (0.0, 0.0, 1.0))
        self.assertEqual(position, (0.0, -20.0, 8.0))
        self.assertEqual(look_at, (0.0, 0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
