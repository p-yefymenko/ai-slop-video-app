from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from label_landmarks import (  # noqa: E402
    RENDER,
    choose_box,
    fuse_points,
    landmark_phrase,
    locate_landmarks,
    pending_landmark_ids,
    set_landmark_position,
    surface_point,
)
from mesh_io import box_mesh  # noqa: E402
from spatial_previs import project, unproject  # noqa: E402


SCRIPT = """{
  "locations": {
    "sun_well_court": {
      "spatial": {
        "landmarks": {
          "sun_well": {},
          "anvil_altar": {},
          "gate_column_l": {}
        }
      }
    }
  },
  "episodes": [
    {
      "scenes": [
        {
          "lookAtId": "sun_well"
        }
      ]
    }
  ]
}
"""


class _BoxGrounder:
    def __init__(self, boxes):
        self.boxes_by_phrase = boxes

    def boxes(self, _image, phrase):
        return list(self.boxes_by_phrase.get(phrase, []))


class LandmarkTests(unittest.TestCase):
    def test_phrase_reads_side_suffixes(self) -> None:
        self.assertEqual(landmark_phrase("gate_column_l"), "left gate column")
        self.assertEqual(landmark_phrase("gate_column_r"), "right gate column")
        self.assertEqual(landmark_phrase("sun_well"), "sun well")

    def test_pending_keeps_a_hand_position_unless_forced(self) -> None:
        landmarks = {"sun_well": {}, "anvil_altar": {"position": [1, 2, 3]}}
        self.assertEqual(pending_landmark_ids(landmarks, False), ["sun_well"])
        self.assertEqual(
            pending_landmark_ids(landmarks, True),
            ["sun_well", "anvil_altar"],
        )

    def test_choose_box_rejects_the_whole_frame_and_a_speck(self) -> None:
        frame = (0.0, 0.0, float(RENDER), float(RENDER))
        speck = (10.0, 10.0, 12.0, 12.0)
        object_box = (100.0, 100.0, 180.0, 180.0)
        self.assertIsNone(choose_box([frame, speck], RENDER, RENDER))
        self.assertEqual(
            choose_box([frame, object_box, (120.0, 120.0, 160.0, 160.0)], RENDER, RENDER),
            (120.0, 120.0, 160.0, 160.0),
        )

    def test_unproject_inverts_project(self) -> None:
        camera = {
            "position": [0.0, -8.0, 3.0],
            "lookAt": [0.0, 0.0, 1.0],
            "verticalFovDegrees": 36.0,
            "rollDegrees": 0.0,
        }
        point = (1.2, 0.4, 1.5)
        projected = project(point, camera, width=RENDER, height=RENDER)
        self.assertIsNotNone(projected)
        assert projected is not None
        back = unproject(projected[0], projected[1], projected[2], camera, width=RENDER, height=RENDER)
        for original, recovered in zip(point, back):
            self.assertAlmostEqual(original, recovered, places=4)

    def test_fuse_keeps_the_largest_cluster(self) -> None:
        found = fuse_points([(0.0, 0.0, 1.0), (0.1, -0.1, 1.05), (9.0, 9.0, 1.0)], 0.5)
        self.assertIsNotNone(found)
        assert found is not None
        self.assertAlmostEqual(found[0], 0.05, places=2)
        self.assertAlmostEqual(found[2], 1.025, places=2)

    def test_script_splice_replaces_only_the_landmark_object(self) -> None:
        updated = set_landmark_position(SCRIPT, "sun_well_court", "sun_well", (1.25, -3.5, 0.4))
        self.assertIn('"lookAtId": "sun_well"', updated)
        self.assertIn('"anvil_altar": {}', updated)
        self.assertIn('"gate_column_l": {}', updated)
        data = json.loads(updated)
        self.assertEqual(
            data["locations"]["sun_well_court"]["spatial"]["landmarks"]["sun_well"],
            {"position": [1.25, -3.5, 0.4]},
        )
        start = SCRIPT.index('"sun_well":')
        end = SCRIPT.index('"anvil_altar"')
        self.assertEqual(updated[:start], SCRIPT[:start])
        self.assertEqual(updated[updated.index('"anvil_altar"') :], SCRIPT[end:])

    def test_surface_point_reads_a_depth_window(self) -> None:
        import numpy as np

        camera = {
            "position": [0.0, -6.0, 2.0],
            "lookAt": [0.0, 0.0, 1.0],
            "verticalFovDegrees": 40.0,
            "rollDegrees": 0.0,
        }
        depth = np.full((32, 32), np.inf)
        depth[8:23, 8:23] = 5.0
        point = surface_point(depth, (8, 8, 22, 22), camera, width=32, height=32)
        self.assertIsNotNone(point)
        assert point is not None
        expected = unproject(15.5, 15.5, 5.0, camera, width=32, height=32)
        for original, recovered in zip(expected, point):
            self.assertAlmostEqual(original, recovered, places=4)

    def test_shaded_box_unprojects_to_the_front_face(self) -> None:
        vertices, faces = box_mesh((4.0, 4.0, 2.0))
        target = (0.0, -2.0, 1.0)
        camera = {
            "position": [0.0, -12.0, 3.0],
            "lookAt": [0.0, 0.0, 1.0],
            "verticalFovDegrees": 36.0,
            "rollDegrees": 0.0,
        }
        projected = project(target, camera, width=RENDER, height=RENDER)
        self.assertIsNotNone(projected)
        assert projected is not None
        half = 24.0
        box = (
            projected[0] - half,
            projected[1] - half,
            projected[0] + half,
            projected[1] + half,
        )
        found = locate_landmarks(
            vertices,
            faces,
            (4.0, 4.0, 2.0),
            ["block"],
            _BoxGrounder({"block": [box]}),
            views=[camera],
        )
        point = found["block"]
        self.assertIsNotNone(point)
        assert point is not None
        for original, recovered in zip(target, point):
            self.assertAlmostEqual(original, recovered, delta=0.35)


if __name__ == "__main__":
    unittest.main()
