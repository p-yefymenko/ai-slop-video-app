from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import spatial_previs  # noqa: E402
from coords import schema_to_gltf  # noqa: E402
from mesh_io import primitive_mesh, write_schema_glb  # noqa: E402
from pipeline_paths import OUTPUT_DIR, set_dir  # noqa: E402
from spatial_previs import (  # noqa: E402
    _scene_surfaces,
    character_pose_joints,
    write_shot_description,
)


class PrevisSetTests(unittest.TestCase):
    def tearDown(self) -> None:
        spatial_previs._MESH_CACHE.clear()
        shutil.rmtree(OUTPUT_DIR / "unit-shot", ignore_errors=True)

    def test_missing_set_keeps_landmark_boxes(self) -> None:
        show, episode, scene = _bare_scene()
        _camera, surfaces, people = _scene_surfaces(show, episode, scene, 0.0)
        landmarks = [item for item in surfaces if item[1] == 176]
        self.assertEqual(len(landmarks), 12)
        self.assertEqual(people, [])

    def test_set_mesh_replaces_landmark_boxes(self) -> None:
        show, episode, scene = _bare_scene()
        directory = set_dir("unit-shot", "room")
        directory.mkdir(parents=True)
        vertices, faces = primitive_mesh("column", (1.0, 1.0, 4.0))
        write_schema_glb(directory / "set.glb", vertices + np.array([0.0, 2.0, 0.0]), faces)
        _camera, surfaces, _people = _scene_surfaces(show, episode, scene, 0.0)
        landmarks = [item for item in surfaces if item[1] == 176]
        self.assertGreater(len(landmarks), 12)

    def test_shot_description_stores_gltf_positions(self) -> None:
        show, episode, scene = _bare_scene()
        destination = write_shot_description(show, episode, scene)
        payload = json.loads(destination.read_text(encoding="utf-8"))
        self.assertEqual(payload["space"], "gltf-y-up")
        self.assertIsNone(payload["set"])
        camera = payload["camera"]["keyframes"][0]
        self.assertEqual(camera["schemaPosition"], [20.0, 0.0, 1.0])
        self.assertEqual(camera["position"], list(schema_to_gltf((20.0, 0.0, 1.0))))
        self.assertEqual(payload["characters"], [])

    def test_proxy_scales_the_standing_mannequin(self) -> None:
        show = {
            "characters": {
                "ada": {"proxy": {"heightMeters": 2.0, "build": "broad"}},
                "bob": {},
            }
        }
        episode = {"spatialTimeline": {"characterTracks": {}}}
        scene = {"locationId": "room", "characterIds": []}
        state = {
            "position": [0.0, 0.0, 0.0],
            "bodyYawDegrees": 0.0,
            "stance": "standing",
        }
        ada = character_pose_joints(show, episode, scene, state, 0.0, "ada")
        bob = character_pose_joints(show, episode, scene, state, 0.0, "bob")
        scale = 2.0 / 1.72
        self.assertAlmostEqual(ada["nose"][2], 1.72 * scale - 0.04 * scale, places=5)
        self.assertAlmostEqual(bob["nose"][2], 1.72 - 0.04, places=5)
        self.assertGreater(ada["right_shoulder"][0], bob["right_shoulder"][0])


def _bare_scene() -> tuple[dict, dict, dict]:
    show = {
        "id": "unit-shot",
        "characters": {},
        "locations": {
            "room": {
                "spatial": {
                    "sizeMeters": [10, 10, 4],
                    "landmarks": {
                        "block": {
                            "kind": "box",
                            "position": [0, 2, 0],
                            "size": [1, 1, 1],
                        }
                    },
                }
            }
        },
    }
    episode = {"episodeNumber": 1, "scenes": []}
    scene = {
        "sceneNumber": 1,
        "locationId": "room",
        "characterIds": [],
        "timeRangeSeconds": [0, 1],
        "camera": {
            "keyframes": [
                {
                    "timeSeconds": 0,
                    "position": [20, 0, 1],
                    "lookAt": [0, 2, 1],
                    "verticalFovDegrees": 40,
                }
            ]
        },
    }
    return show, episode, scene


if __name__ == "__main__":
    unittest.main()
