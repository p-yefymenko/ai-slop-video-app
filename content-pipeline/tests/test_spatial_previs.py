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
    camera_at,
    character_facing_direction,
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
        self.assertTrue(
            all(location.get("spatial") for location in self.show["locations"].values())
        )
        self.assertTrue(all(scene.get("camera") for scene in self.episode["scenes"]))
        self.assertTrue(
            all(scene.get("timeRangeSeconds") for scene in self.episode["scenes"])
        )

    def test_opening_shots_have_distinct_physical_views(self) -> None:
        first, second = self.episode["scenes"][:2]
        first_pose = first["camera"]["keyframes"][0]
        second_pose = second["camera"]["keyframes"][0]
        self.assertNotEqual(first_pose["position"], second_pose["position"])
        self.assertGreater(first_pose["position"][1], -120)
        self.assertGreater(second_pose["position"][2], 100)
        self.assertGreater(len(self.episode["scenes"][3]["camera"]["keyframes"]), 1)

    def test_overhead_and_rolled_cameras_project(self) -> None:
        overhead = project(
            (0.0, 0.0, 0.0),
            {
                "position": [0.0, 0.0, 5.0],
                "lookAt": [0.0, 0.0, 0.0],
                "verticalFovDegrees": 50.0,
            },
        )
        self.assertIsNotNone(overhead)
        assert overhead is not None
        self.assertAlmostEqual(overhead[0], PROXY_WIDTH / 2.0)
        self.assertAlmostEqual(overhead[1], PROXY_HEIGHT / 2.0)
        level = project(
            (1.0, 0.0, 1.5),
            {
                "position": [0.0, -5.0, 1.5],
                "lookAt": [0.0, 0.0, 1.5],
                "verticalFovDegrees": 40.0,
            },
        )
        rolled = project(
            (1.0, 0.0, 1.5),
            {
                "position": [0.0, -5.0, 1.5],
                "lookAt": [0.0, 0.0, 1.5],
                "verticalFovDegrees": 40.0,
                "rollDegrees": 90.0,
            },
        )
        self.assertIsNotNone(level)
        self.assertIsNotNone(rolled)
        assert level is not None and rolled is not None
        self.assertGreater(level[0], PROXY_WIDTH / 2.0)
        self.assertAlmostEqual(level[1], PROXY_HEIGHT / 2.0, delta=2)
        self.assertAlmostEqual(rolled[0], PROXY_WIDTH / 2.0, delta=2)
        self.assertGreater(rolled[1], PROXY_HEIGHT / 2.0)

    def test_offscreen_head_is_allowed(self) -> None:
        scene = next(
            item for item in self.episode["scenes"] if item["sceneNumber"] == 3
        )
        original = scene["camera"]
        scene["camera"] = {
            "keyframes": [
                {
                    "timeSeconds": 4.0,
                    "position": [20.0, -2.2, 1.45],
                    "lookAt": [20.0, -12.0, 1.45],
                    "verticalFovDegrees": 28.0,
                }
            ]
        }
        try:
            errors = validate_spatial_episode(self.show, self.episode)
        finally:
            scene["camera"] = original
        self.assertFalse(any("head is outside" in error for error in errors))
        self.assertEqual(errors, [])

    def test_camera_keyframes_interpolate(self) -> None:
        scene = self.episode["scenes"][3]
        start, finish = scene["timeRangeSeconds"]
        mid = camera_at(scene, (start + finish) / 2.0)
        first = scene["camera"]["keyframes"][0]
        last = scene["camera"]["keyframes"][-1]
        self.assertAlmostEqual(
            mid["position"][0],
            (first["position"][0] + last["position"][0]) / 2.0,
        )

    def test_prompt_compiler_does_not_expose_coordinates(self) -> None:
        scene = next(
            scene for scene in self.episode["scenes"] if scene["sceneNumber"] == 5
        )
        prompt = compile_spatial_video_prompt(scene)
        self.assertIn("visibly lip-syncs every spoken word", prompt)
        self.assertNotIn("established mark", prompt)
        self.assertNotIn("CAMERA:", prompt)
        self.assertNotIn("VISUAL:", prompt)
        self.assertNotIn("[", prompt)
        self.assertNotIn("1.5", prompt)
        self.assertFalse(scene_has_spatial_change(self.episode, scene))

    def test_anchor_pair_uses_reciprocal_profile_references(self) -> None:
        scene = self.episode["scenes"][8]
        self.assertEqual(
            character_facing_direction(self.episode, scene, "kael"), "right"
        )
        self.assertEqual(
            character_facing_direction(self.episode, scene, "malrec"), "left"
        )

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
