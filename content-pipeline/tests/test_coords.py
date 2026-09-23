from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from coords import gltf_to_schema, schema_points_to_gltf, schema_to_gltf  # noqa: E402


class CoordTests(unittest.TestCase):
    def test_schema_point_round_trip(self) -> None:
        schema = (1.0, 2.0, 3.0)
        self.assertEqual(schema_to_gltf(schema), (1.0, 3.0, -2.0))
        self.assertEqual(gltf_to_schema(schema_to_gltf(schema)), schema)

    def test_point_arrays_round_trip(self) -> None:
        points = np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, 8.0]], dtype=np.float64)
        restored = gltf_to_schema_array(schema_points_to_gltf(points))
        np.testing.assert_allclose(restored, points)


def gltf_to_schema_array(points: np.ndarray) -> np.ndarray:
    from coords import gltf_points_to_schema

    return gltf_points_to_schema(points)


if __name__ == "__main__":
    unittest.main()
