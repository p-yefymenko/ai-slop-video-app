from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from spatial_previs import (  # noqa: E402
    PROXY_HEIGHT,
    PROXY_WIDTH,
    compile_spatial_video_prompt,
    project,
    render_scene_proxy,
    scene_has_spatial_change,
    timeline_state,
    validate_spatial_episode,
)


class SpatialPrevisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "scripts_input"
            / "crown-of-the-last-dragon.json"
        )
        cls.show = json.loads(path.read_text(encoding="utf-8"))
        cls.episode = cls.show["episodes"][0]

    def test_camera_target_projects_to_frame_center(self) -> None:
        camera = {
            "position": [0.0, -5.0, 1.5],
            "lookAt": [0.0, 0.0, 1.5],
            "verticalFovDegrees": 40.0,
        }
        projected = project((0.0, 0.0, 1.5), camera)
        self.assertIsNotNone(projected)
        assert projected is not None
        self.assertAlmostEqual(projected[0], PROXY_WIDTH / 2.0)
        self.assertAlmostEqual(projected[1], PROXY_HEIGHT / 2.0)

    def test_timeline_interpolates_position_and_yaw(self) -> None:
        state = timeline_state(
            [
                {
                    "timeSeconds": 0,
                    "locationId": "set",
                    "position": [0, 0, 0],
                    "bodyYawDegrees": 170,
                    "stance": "standing",
                },
                {
                    "timeSeconds": 2,
                    "locationId": "set",
                    "position": [2, 0, 0],
                    "bodyYawDegrees": -170,
                    "stance": "standing",
                },
            ],
            1,
        )
        self.assertEqual(state["position"], [1.0, 0.0, 0.0])
        self.assertAlmostEqual(abs(state["bodyYawDegrees"]), 180.0)

    def test_migrated_coverage_has_valid_geometry(self) -> None:
        self.assertEqual(validate_spatial_episode(self.show, self.episode), [])
        migrated = [
            scene
            for scene in self.episode["scenes"]
            if scene["sceneNumber"] in {4, 5, 6, 7}
        ]
        self.assertTrue(all(scene.get("camera") for scene in migrated))
        self.assertTrue(all(scene.get("timeRangeSeconds") for scene in migrated))

    def test_prompt_compiler_does_not_expose_coordinates(self) -> None:
        scene = next(
            scene for scene in self.episode["scenes"] if scene["sceneNumber"] == 5
        )
        prompt = compile_spatial_video_prompt(self.episode, scene)
        self.assertIn("established mark", prompt)
        self.assertNotIn("[", prompt)
        self.assertNotIn("1.5", prompt)
        self.assertFalse(scene_has_spatial_change(self.episode, scene))

    def test_proxy_renderer_writes_vertical_frame(self) -> None:
        scene = next(
            scene for scene in self.episode["scenes"] if scene["sceneNumber"] == 4
        )
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "proxy.png"
            render_scene_proxy(
                self.show,
                self.episode,
                scene,
                scene["timeRangeSeconds"][0],
                destination,
            )
            from PIL import Image

            with Image.open(destination) as image:
                self.assertEqual(image.size, (PROXY_WIDTH, PROXY_HEIGHT))


if __name__ == "__main__":
    unittest.main()
