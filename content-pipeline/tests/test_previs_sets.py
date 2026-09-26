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
from mesh_io import box_mesh, primitive_mesh, write_schema_glb  # noqa: E402
from pipeline_paths import OUTPUT_DIR, location_dir  # noqa: E402
from clay_gpu import ClayBatch  # noqa: E402
from spatial_previs import (  # noqa: E402
    PROXY_HEIGHT,
    PROXY_WIDTH,
    VIEWPORT_GRAY,
    _raster_clay,
    _scene_surfaces,
    _shade_gray,
    _triangle_normal,
    character_pose_joints,
    project,
    write_scene_description,
)


class PrevisSetTests(unittest.TestCase):
    def tearDown(self) -> None:
        spatial_previs._MESH_CACHE.clear()
        for stage in ("plates", "assets", "landmarks", "previs", "frames", "generate"):
            shutil.rmtree(OUTPUT_DIR / stage / "unit-shot", ignore_errors=True)

    def test_missing_set_draws_no_landmark_mesh(self) -> None:
        show, episode, scene = _bare_scene()
        _camera, surfaces, people = _scene_surfaces(show, episode, scene, 0.0)
        landmarks = [item for item in surfaces if item.base == 176]
        self.assertEqual(landmarks, [])
        self.assertEqual(people, [])

    def test_set_mesh_replaces_landmark_boxes(self) -> None:
        show, episode, scene = _bare_scene()
        directory = location_dir("unit-shot", "room")
        directory.mkdir(parents=True)
        vertices, faces = primitive_mesh("column", (1.0, 1.0, 4.0))
        write_schema_glb(directory / "model.glb", vertices + np.array([0.0, 2.0, 0.0]), faces)
        _camera, surfaces, _people = _scene_surfaces(show, episode, scene, 0.0)
        landmarks = [item for item in surfaces if item.base == 176]
        self.assertEqual(len(landmarks), 1)
        self.assertGreater(int(landmarks[0].faces.shape[0]), 0)

    def test_a_character_mesh_stands_on_their_mark(self) -> None:
        show, episode, scene = _bare_scene()
        show["characters"] = {"ada": {"promptBlock": "Adult woman."}}
        scene["characterIds"] = ["ada"]
        episode["spatialTimeline"] = {
            "characterTracks": {
                "ada": [
                    {
                        "timeSeconds": 0,
                        "locationId": "room",
                        "position": [1.0, 2.0, 0.0],
                        "bodyYawDegrees": 0,
                        "stance": "standing",
                    }
                ]
            }
        }
        directory = OUTPUT_DIR / "assets" / "unit-shot" / "characters" / "ada"
        directory.mkdir(parents=True)
        vertices, faces = box_mesh((0.4, 0.3, 1.7))
        write_schema_glb(directory / "model.glb", vertices, faces)
        _camera, surfaces, people = _scene_surfaces(show, episode, scene, 0.0)
        meshes = [item for item in surfaces if item.base == 208]
        self.assertEqual(len(meshes), 1)
        self.assertEqual(meshes[0].offset, (1.0, 2.0, 0.0))
        self.assertEqual([item for item in surfaces if item.base == 214], [])
        self.assertEqual([character_id for character_id, _joints in people], ["ada"])

    def test_shot_description_stores_gltf_positions(self) -> None:
        show, episode, scene = _bare_scene()
        destination = write_scene_description(show, episode, scene)
        payload = json.loads(destination.read_text(encoding="utf-8"))
        self.assertEqual(payload["space"], "gltf-y-up")
        self.assertIsNone(payload["location"])
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

    def test_gpu_clay_uses_the_face_normal(self) -> None:
        triangle = ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0))
        camera = {
            "position": [0.0, -2.0, 5.0],
            "lookAt": [0.7, 0.7, 0.0],
            "verticalFovDegrees": 40.0,
            "rollDegrees": 0.0,
        }
        batch = ClayBatch(
            np.array(triangle, dtype=np.float32),
            np.array([[0, 1, 2]], dtype=np.uint32),
            176,
        )
        image, zbuf = _raster_clay([batch], camera)
        centroid = tuple(sum(corner[index] for corner in triangle) / 3.0 for index in range(3))
        screen = project(centroid, camera)
        self.assertIsNotNone(screen)
        assert screen is not None
        x = int(screen[0])
        y = int(screen[1])
        self.assertGreaterEqual(x, 0)
        self.assertLess(x, PROXY_WIDTH)
        self.assertGreaterEqual(y, 0)
        self.assertLess(y, PROXY_HEIGHT)
        self.assertEqual(image.getpixel((x, y)), _shade_gray(_triangle_normal(triangle), 176))
        self.assertAlmostEqual(float(zbuf[y, x]), screen[2], delta=0.15)
        self.assertEqual(image.getpixel((2, 2)), VIEWPORT_GRAY)
        self.assertFalse(np.isfinite(zbuf[2, 2]))


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
                            "position": [0, 2, 0],
                            "size": [1, 1, 1],
                            "appearance": "One twisted marble column standing alone.",
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
