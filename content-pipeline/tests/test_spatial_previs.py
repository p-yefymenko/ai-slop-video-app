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
    VIEWPORT_GRAY,
    blockout_sample_times,
    camera_at,
    character_facing_direction,
    character_pose_joints,
    compile_spatial_video_prompt,
    episode_character_state,
    project,
    render_blocked_frame,
    render_scene_proxy,
    render_structure_maps,
    scene_blockouts,
    scene_has_spatial_change,
    timeline_state,
    validate_spatial_episode,
    write_episode_blockout,
)

SHOW_JSON = (
    Path(__file__).resolve().parents[1] / "shows" / "the-iron-bride" / "script.json"
)


class SpatialPrevisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.show = json.loads(SHOW_JSON.read_text(encoding="utf-8"))
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

    def test_episode_has_valid_geometry(self) -> None:
        self.assertEqual(validate_spatial_episode(self.show, self.episode), [])
        self.assertTrue(
            all(location.get("spatial") for location in self.show["locations"].values())
        )
        self.assertTrue(all(scene.get("camera") for scene in self.episode["scenes"]))
        self.assertTrue(
            all(scene.get("timeRangeSeconds") for scene in self.episode["scenes"])
        )

    def test_opening_is_world_then_a_distinct_arena_view(self) -> None:
        first, second = self.episode["scenes"][:2]
        self.assertEqual(first["characterIds"], [])
        first_pose = first["camera"]["keyframes"][0]
        second_pose = second["camera"]["keyframes"][0]
        self.assertNotEqual(first_pose["position"], second_pose["position"])
        self.assertLess(first_pose["position"][1], -40)
        self.assertNotEqual(first["locationId"], second["locationId"])
        self.assertTrue(
            any(len(scene["camera"]["keyframes"]) > 1 for scene in self.episode["scenes"])
        )

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
        scene = next(item for item in self.episode["scenes"] if item["characterIds"])
        original = scene["camera"]
        start = float(scene["timeRangeSeconds"][0])
        scene["camera"] = {
            "keyframes": [
                {
                    "timeSeconds": start,
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
        scene = next(
            item
            for item in self.episode["scenes"]
            if len(item["camera"]["keyframes"]) > 1
        )
        start, finish = scene["timeRangeSeconds"]
        mid = camera_at(scene, (start + finish) / 2.0)
        first = scene["camera"]["keyframes"][0]
        last = scene["camera"]["keyframes"][-1]
        self.assertAlmostEqual(
            mid["position"][0],
            (first["position"][0] + last["position"][0]) / 2.0,
        )

    def test_prompt_compiler_does_not_expose_coordinates(self) -> None:
        scene = next(item for item in self.episode["scenes"] if item.get("speakerId"))
        prompt = compile_spatial_video_prompt(scene)
        self.assertIn("visibly lip-syncs every spoken word", prompt)
        self.assertNotIn("established mark", prompt)
        self.assertNotIn("CAMERA:", prompt)
        self.assertNotIn("VISUAL:", prompt)
        self.assertNotIn("[", prompt)

    def test_standoff_uses_reciprocal_screen_direction(self) -> None:
        scene = next(
            item
            for item in self.episode["scenes"]
            if item["characterIds"][:2] == ["sela", "tomas"]
            or set(item["characterIds"][:2]) == {"sela", "vardan"}
        )
        left = character_facing_direction(self.episode, scene, scene["characterIds"][0], self.show)
        right = character_facing_direction(self.episode, scene, scene["characterIds"][1], self.show)
        self.assertIn(left, {"left", "right"})
        self.assertIn(right, {"left", "right"})

    def test_proxy_renderer_writes_vertical_frame(self) -> None:
        scene = next(
            item
            for item in self.episode["scenes"]
            if len(item["characterIds"]) >= 2
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

    def test_structure_maps_lock_projected_joints(self) -> None:
        scene = next(
            item
            for item in self.episode["scenes"]
            if len(item["characterIds"]) >= 1
        )
        time_seconds = float(scene["timeRangeSeconds"][0])
        character_id = scene["characterIds"][0]
        state = episode_character_state(self.episode, character_id, time_seconds)
        nose = project(
            character_pose_joints(
                self.show, self.episode, scene, state, time_seconds, character_id
            )["nose"],
            camera_at(scene, time_seconds),
        )
        self.assertIsNotNone(nose)
        assert nose is not None
        with tempfile.TemporaryDirectory() as temp:
            depth_path = Path(temp) / "depth.png"
            pose_path = Path(temp) / "pose.png"
            render_structure_maps(
                self.show,
                self.episode,
                scene,
                time_seconds,
                depth_path,
                pose_path,
            )
            from PIL import Image

            with Image.open(pose_path) as pose, Image.open(depth_path) as depth:
                self.assertEqual(pose.size, (PROXY_WIDTH, PROXY_HEIGHT))
                self.assertEqual(depth.size, (PROXY_WIDTH, PROXY_HEIGHT))
                x, y = int(nose[0]), int(nose[1])
                self.assertGreater(sum(pose.getpixel((x, y))), 0)
                self.assertGreater(sum(depth.getpixel((x, y))), 0)
                extrema = depth.getextrema()
                self.assertGreater(extrema[0][1], extrema[0][0])

    def test_interior_blockout_keeps_the_floor(self) -> None:
        scene = next(item for item in self.episode["scenes"] if item["sceneNumber"] == 2)
        image, _zbuf, _people = render_blocked_frame(
            self.show,
            self.episode,
            scene,
            float(scene["timeRangeSeconds"][0]),
        )
        self.assertEqual(image.size, (PROXY_WIDTH, PROXY_HEIGHT))
        bottom = image.crop((0, int(PROXY_HEIGHT * 0.82), PROXY_WIDTH, PROXY_HEIGHT))
        band = list(bottom.get_flattened_data())
        ground = sum(
            1
            for index in range(0, len(band), 3)
            if (band[index], band[index + 1], band[index + 2]) != VIEWPORT_GRAY
        )
        self.assertGreater(ground, (len(band) // 3) // 2)

    def test_blockout_samples_cover_the_shot(self) -> None:
        times = blockout_sample_times(2.0, 4.0)
        self.assertEqual(times[0], 2.0)
        self.assertEqual(times[-1], 4.0)
        self.assertGreaterEqual(len(times), 3)

    def test_episode_blockout_joins_scenes_in_script_order(self) -> None:
        import pipeline_paths
        from unittest.mock import patch

        show = {"id": "demo"}
        episode = {
            "episodeNumber": 1,
            "scenes": [
                {"sceneNumber": 2, "camera": {"position": [0, 0, 0]}, "timeRangeSeconds": [1, 2]},
                {"sceneNumber": 1, "camera": {"position": [0, 0, 0]}, "timeRangeSeconds": [0, 1]},
            ],
        }
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(pipeline_paths, "OUTPUT_DIR", Path(temp)):
                ready, missing = scene_blockouts(show, episode)
                self.assertEqual(ready, [])
                self.assertEqual(missing, [2, 1])
                self.assertIsNone(write_episode_blockout(show, episode))
                for scene_number in (1, 2):
                    path = pipeline_paths.blockout_video_path("demo", 1, scene_number)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"x" * 1024)
                joined: list[list[Path]] = []

                def capture(files: list[Path], dest: Path) -> None:
                    joined.append(list(files))
                    dest.write_bytes(b"joined")

                with patch("ffmpeg_tools.concat_videos", capture):
                    destination = write_episode_blockout(show, episode)
                self.assertEqual(
                    [path.parent.name for path in joined[0]],
                    ["scene_02", "scene_01"],
                )
                assert destination is not None
                self.assertEqual(destination.name, "blockout.mp4")
                self.assertEqual(destination.parent.name, "01_previs")


if __name__ == "__main__":
    unittest.main()
