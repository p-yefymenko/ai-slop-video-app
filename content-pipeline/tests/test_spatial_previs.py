from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from spatial_previs import (  # noqa: E402
    BACKDROP_SKY,
    PROXY_HEIGHT,
    PROXY_WIDTH,
    VIEWPORT_GRAY,
    _edge_image,
    blockout_sample_times,
    camera_at,
    character_facing_direction,
    character_pose_joints,
    face_points_at_camera,
    compile_spatial_video_prompt,
    episode_character_state,
    facing_yaw_degrees,
    project,
    render_blocked_frame,
    render_scene_proxy,
    render_structure_maps,
    scene_blockouts,
    scene_has_spatial_change,
    timeline_state,
    _yaw_axes,
    validate_spatial_episode,
    visible_face_ids,
    visible_place_line,
    landmark_in_view,
    write_episode_blockout,
    _face_ellipse,
    _face_mask,
    _scene_surfaces,
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

    def test_front_aims_at_look_at_and_the_face_cameras_sit_there(self) -> None:
        state = {"position": [-1.4, -2.0, 0.0], "bodyYawDegrees": 0}
        forward, _right = _yaw_axes(facing_yaw_degrees(state, (0.0, 6.0, 0.0)))
        self.assertGreater(forward[1], 0.9)
        backward, _right = _yaw_axes(facing_yaw_degrees(state, (-1.4, -8.0, 0.0)))
        self.assertLess(backward[1], -0.9)
        self.assertEqual(facing_yaw_degrees({"bodyYawDegrees": 35}, None), 35.0)
        for scene_number, character_id in ((3, "sela"), (12, "sela"), (17, "sela"), (18, "tomas")):
            scene = next(item for item in self.episode["scenes"] if item["sceneNumber"] == scene_number)
            time_seconds = float(scene["timeRangeSeconds"][0])
            character = episode_character_state(self.episode, character_id, time_seconds)
            camera = camera_at(scene, time_seconds)
            feet = character["position"]
            to_camera = (
                camera["position"][0] - feet[0],
                camera["position"][1] - feet[1],
            )
            face = _yaw_axes(
                facing_yaw_degrees(
                    character,
                    (0.0, 6.0, 0.0) if character.get("lookAtId") == "vardan" else (0.0, 4.5, 0.0),
                )
            )[0]
            self.assertGreater(
                face[0] * to_camera[0] + face[1] * to_camera[1],
                0.0,
                f"scene {scene_number} camera is behind {character_id}",
            )

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
        camera, _batches, people = _scene_surfaces(
            self.show, self.episode, scene, time_seconds
        )
        joints = dict(people)[character_id]
        nose = project(joints["nose"], camera)
        self.assertIsNotNone(nose)
        assert nose is not None
        with tempfile.TemporaryDirectory() as temp:
            depth_path = Path(temp) / "depth.png"
            pose_path = Path(temp) / "pose.png"
            edge_path = Path(temp) / "edges.png"
            normal_path = Path(temp) / "normal.png"
            render_structure_maps(
                self.show,
                self.episode,
                scene,
                time_seconds,
                depth_path,
                pose_path,
                edge_destination=edge_path,
                normal_destination=normal_path,
            )
            from PIL import Image

            with Image.open(pose_path) as pose, Image.open(depth_path) as depth, Image.open(edge_path) as edges, Image.open(normal_path) as normal:
                self.assertEqual(pose.size, (PROXY_WIDTH, PROXY_HEIGHT))
                self.assertEqual(depth.size, (PROXY_WIDTH, PROXY_HEIGHT))
                self.assertEqual(edges.size, (PROXY_WIDTH, PROXY_HEIGHT))
                self.assertEqual(normal.size, (PROXY_WIDTH, PROXY_HEIGHT))
                x, y = int(nose[0]), int(nose[1])
                self.assertGreater(sum(pose.getpixel((x, y))), 0)
                self.assertGreater(sum(depth.getpixel((x, y))), 0)
                extrema = depth.getextrema()
                self.assertGreater(extrema[0][1], extrema[0][0])
                self.assertGreater(edges.convert("L").getextrema()[1], 0)
                self.assertGreater(normal.convert("L").getextrema()[1], 0)

    def test_a_depth_step_becomes_an_edge(self) -> None:
        import numpy as np

        depth = np.full((8, 8), np.inf)
        depth[:, :4] = 2.0
        depth[:, 4:] = 10.0
        image = _edge_image(depth)
        self.assertEqual(image.getpixel((4, 4)), (255, 255, 255))
        self.assertEqual(image.getpixel((1, 1)), (0, 0, 0))

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
                self.assertEqual(destination.parent.name, "1")
                self.assertEqual(destination.parents[2].name, "previs")

    def test_backdrop_labels_sky_ground_and_the_horizon(self) -> None:
        from spatial_previs import BACKDROP_GROUND, BACKDROP_SKY, BACKDROP_SURROUND, render_backdrop

        colors = {"sky": BACKDROP_SKY, "ground": BACKDROP_GROUND, "surround": BACKDROP_SURROUND}

        looking_down = {
            "position": [5.0, 0.0, 2.0],
            "lookAt": [5.0, 0.0, 0.0],
            "verticalFovDegrees": 40.0,
            "rollDegrees": 0.0,
        }
        empty = np.full((64, 48), np.inf, dtype=np.float32)
        wide = render_backdrop(looking_down, empty, {"sizeMeters": [100.0, 100.0, 10.0]}, colors)
        self.assertEqual(wide.getpixel((24, 32)), BACKDROP_GROUND)
        tight = render_backdrop(looking_down, empty, {"sizeMeters": [0.2, 0.2, 10.0]}, colors)
        self.assertEqual(tight.getpixel((24, 32)), BACKDROP_SURROUND)
        looking_up = {
            "position": [0.0, 0.0, 2.0],
            "lookAt": [0.0, 0.0, 4.0],
            "verticalFovDegrees": 40.0,
            "rollDegrees": 0.0,
        }
        sky = render_backdrop(looking_up, empty, {"sizeMeters": [100.0, 100.0, 10.0]}, colors)
        self.assertEqual(sky.getpixel((24, 32)), BACKDROP_SKY)
        blocked = empty.copy()
        blocked[32, 24] = 1.0
        covered = render_backdrop(looking_down, blocked, {"sizeMeters": [100.0, 100.0, 10.0]}, colors)
        self.assertEqual(covered.getpixel((24, 32)), (0, 0, 0))
        holed = empty.copy()
        holed[20:44, 10:38] = 1.0
        holed[30:33, 20:24] = np.inf
        sealed = render_backdrop(looking_down, holed, {"sizeMeters": [100.0, 100.0, 10.0]}, colors)
        self.assertEqual(sealed.getpixel((21, 31)), (0, 0, 0))

    def test_backdrop_depth_leaves_the_open_floor_empty(self) -> None:
        from clay_gpu import ClayBatch
        from spatial_previs import FLOOR_BASE, backdrop_depth

        floor = ClayBatch(
            np.zeros((3, 3), dtype=np.float32),
            np.zeros((1, 3), dtype=np.uint32),
            FLOOR_BASE,
        )
        camera = {
            "position": [0.0, -5.0, 1.5],
            "lookAt": [0.0, 0.0, 1.5],
            "verticalFovDegrees": 40.0,
        }
        depth = backdrop_depth([floor], camera)
        self.assertTrue(np.isinf(depth).all())

    def test_place_line_names_only_landmarks_in_this_camera(self) -> None:
        location = self.show["locations"]["sun_well_court"]
        well = location["spatial"]["landmarks"]["sun_well"]
        close = next(item for item in self.episode["scenes"] if item["sceneNumber"] == 15)
        close_camera = camera_at(close, float(close["timeRangeSeconds"][0]))
        self.assertFalse(landmark_in_view(well, close_camera))
        close_line = visible_place_line(location, close_camera).lower()
        self.assertNotIn("sun well", close_line)
        self.assertNotIn("sun-well", close_line)
        wide = next(item for item in self.episode["scenes"] if item["sceneNumber"] == 2)
        wide_camera = camera_at(wide, float(wide["timeRangeSeconds"][0]))
        self.assertTrue(landmark_in_view(well, wide_camera))
        self.assertIn("white-gold", visible_place_line(location, wide_camera).lower())

    def test_place_line_mentions_backdrop_regions_in_the_guide(self) -> None:
        from PIL import Image

        location = self.show["locations"]["sun_well_court"]
        camera = camera_at(
            next(item for item in self.episode["scenes"] if item["sceneNumber"] == 15),
            40.0,
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "backdrop.png"
            Image.new("RGB", (64, 64), BACKDROP_SKY).save(path)
            line = visible_place_line(location, camera, path)
        self.assertIn("Storm sky", line)
        self.assertNotIn("sun well", line.lower())
        self.assertNotIn("sun-well", line.lower())

    def test_place_line_omits_landmarks_hidden_by_a_person(self) -> None:
        from PIL import Image

        location = self.show["locations"]["sun_well_court"]
        wide = next(item for item in self.episode["scenes"] if item["sceneNumber"] == 2)
        camera = camera_at(wide, float(wide["timeRangeSeconds"][0]))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "clothes.png"
            Image.new("RGB", (PROXY_WIDTH, PROXY_HEIGHT), (200, 180, 160)).save(path)
            line = visible_place_line(location, camera, people_path=path)
        self.assertNotIn("white-gold", line.lower())

    def test_place_line_omits_a_landmark_that_is_mostly_hidden(self) -> None:
        location = self.show["locations"]["sun_well_court"]
        scene = next(item for item in self.episode["scenes"] if item["sceneNumber"] == 5)
        camera = camera_at(scene, float(scene["timeRangeSeconds"][0]))
        probes = visible_place_line(location, camera)
        self.assertIn("worn steps", probes.lower())
        shown = visible_place_line(
            location,
            camera,
            shown={
                "sun_well": 0.05,
                "throne_dais": 0.01,
                "anvil_altar": 0.0,
                "gate_column_l": 0.0,
                "gate_column_r": 0.0,
            },
        )
        self.assertIn("white-gold", shown.lower())
        self.assertNotIn("worn steps", shown.lower())
        self.assertNotIn("dais", shown.lower())

    def test_empty_location_landmark_names_its_appearance(self) -> None:
        location = self.show["locations"]["sky_forge_exterior"]
        scene = next(item for item in self.episode["scenes"] if item["sceneNumber"] == 1)
        line = visible_place_line(location, camera_at(scene, float(scene["timeRangeSeconds"][0])))
        self.assertIn("hanging citadel of black iron", line.lower())
        self.assertIn("chain pylon of black iron", line.lower())
        self.assertNotIn("is in frame", line.lower())

    def test_shown_fraction_is_the_front_surface_of_that_object(self) -> None:
        from spatial_previs import _batch_from_triangles, _quad, _raster_clay, shown_fraction

        camera = {
            "position": [0.0, -4.0, 1.2],
            "lookAt": [0.0, 2.0, 0.5],
            "verticalFovDegrees": 40.0,
        }
        front = _batch_from_triangles(
            _quad((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 1.6), (-1.0, 0.0, 1.6)),
            176,
        )
        back = _batch_from_triangles(
            _quad((-0.6, 2.0, 0.2), (0.6, 2.0, 0.2), (0.6, 2.0, 1.2), (-0.6, 2.0, 1.2)),
            180,
        )
        assert front is not None and back is not None
        _image, scene_z = _raster_clay([front, back], camera)
        _image, front_z = _raster_clay([front], camera)
        _image, back_z = _raster_clay([back], camera)
        self.assertGreater(shown_fraction(front_z, scene_z), 0.9)
        self.assertLess(shown_fraction(back_z, scene_z), 0.15)
        from spatial_previs import screen_coverage

        self.assertGreater(screen_coverage(front_z, scene_z), 0.02)
        self.assertLess(screen_coverage(back_z, scene_z), screen_coverage(front_z, scene_z))

    def test_landmark_prompt_uses_screen_fraction_and_logs_skips(self) -> None:
        from spatial_previs import describe_landmarks

        location = self.show["locations"]["sun_well_court"]
        scene = next(item for item in self.episode["scenes"] if item["sceneNumber"] == 8)
        camera = camera_at(scene, float(scene["timeRangeSeconds"][0]))
        sentence, records = describe_landmarks(
            location,
            camera,
            shown={
                "sun_well": 0.08,
                "throne_dais": 0.01,
                "anvil_altar": 0.0,
                "gate_column_l": 0.004,
                "gate_column_r": 0.004,
            },
            min_screen_fraction=0.02,
        )
        self.assertIn("white-gold", sentence.lower())
        self.assertNotIn("worn steps", sentence.lower())
        by_id = {record["landmarkId"]: record for record in records}
        self.assertTrue(by_id["sun_well"]["sent"])
        self.assertEqual(by_id["sun_well"]["screenFraction"], 0.08)
        self.assertFalse(by_id["throne_dais"]["sent"])
        self.assertEqual(by_id["throne_dais"]["skipReason"], "screen fraction 0.010 < 0.02")
        self.assertEqual(by_id["anvil_altar"]["skipReason"], "no visible pixels")

    def test_flat_clothes_keep_hair_and_skin_and_drop_a_small_stain(self) -> None:
        from spatial_previs import _flatten_figure_colors

        image = np.zeros((80, 60, 3), dtype=np.uint8)
        mask = np.zeros((80, 60), dtype=bool)
        mask[10:70, 15:45] = True
        image[mask] = (198, 178, 149)
        image[10:28, 15:45] = (12, 12, 18)
        image[40:44, 28:32] = (30, 20, 15)
        image[50:66, 15:28] = (170, 130, 100)
        flat = _flatten_figure_colors(image, mask)
        self.assertLess(int(flat[16, 24, 0]), 40)
        self.assertGreater(int(flat[36, 24, 0]), 160)
        self.assertGreater(int(flat[41, 29, 0]), 160)
        self.assertLess(int(flat[58, 20, 1]), 150)

    def test_place_in_front_of_a_person_hides_those_clothes_pixels(self) -> None:
        from spatial_previs import _place_hides_person

        person = np.full((8, 8), 4.0, dtype=np.float32)
        place = np.full((8, 8), np.inf, dtype=np.float32)
        place[2:6, 2:6] = 1.0
        hidden = _place_hides_person(person, place)
        self.assertTrue(bool(hidden[4, 4]))
        self.assertFalse(bool(hidden[0, 0]))
        behind = np.full((8, 8), 6.0, dtype=np.float32)
        self.assertFalse(bool(_place_hides_person(person, behind)[4, 4]))

    def test_clothes_cutout_draws_the_vertex_color_on_black(self) -> None:
        from clay_gpu import ClayBatch, raster_clay
        from spatial_previs import NEAR_CLIP, _camera_basis

        camera = {
            "position": [0.0, -2.0, 0.0],
            "lookAt": [0.0, 0.0, 0.0],
            "verticalFovDegrees": 40.0,
            "rollDegrees": 0.0,
        }
        vertices = np.array(
            [[-0.4, 0.0, -0.4], [0.4, 0.0, -0.4], [0.0, 0.0, 0.4]],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2]], dtype=np.uint32)
        colors = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (3, 1))
        image, _depth = raster_clay(
            [ClayBatch(vertices, faces, 1, colors=colors)],
            _camera_basis(camera, viewport_height=64),
            width=64,
            height=64,
            near=NEAR_CLIP,
            background=(0, 0, 0),
            shading="albedo",
        )
        pixels = np.asarray(image)
        self.assertGreater(int(pixels[:, :, 0].max()), 200)
        self.assertLess(int(pixels[:, :, 1].max()), 40)
        self.assertEqual(int(pixels[0, 0, 0]), 0)

    def test_scene_five_masks_faces_toward_the_camera(self) -> None:
        scene = next(item for item in self.episode["scenes"] if item["sceneNumber"] == 5)
        time_seconds = float(scene["timeRangeSeconds"][0])
        camera = camera_at(scene, time_seconds)
        people = []
        for character_id in scene["characterIds"]:
            state = episode_character_state(self.episode, character_id, time_seconds)
            joints = character_pose_joints(
                self.show, self.episode, scene, state, time_seconds, character_id
            )
            people.append((character_id, joints))
        zbuf = np.full((PROXY_HEIGHT, PROXY_WIDTH), 1.0e6, dtype=np.float32)
        visible = visible_face_ids(camera, people, zbuf)
        self.assertCountEqual(visible, ["vardan", "nira", "kesh", "rhel"])
        turned_away = next(joints for character_id, joints in people if character_id == "sela")
        self.assertFalse(face_points_at_camera(camera, turned_away))

    def test_close_face_mask_covers_the_face_and_leaves_the_sky_black(self) -> None:
        scene = next(item for item in self.episode["scenes"] if item["sceneNumber"] == 3)
        time_seconds = float(scene["timeRangeSeconds"][0])
        camera = camera_at(scene, time_seconds)
        state = episode_character_state(self.episode, "sela", time_seconds)
        joints = character_pose_joints(
            self.show, self.episode, scene, state, time_seconds, "sela"
        )
        box = _face_ellipse(joints, camera)
        assert box is not None
        left, top, right, bottom = box
        self.assertGreater(top, 80)
        self.assertGreater(bottom, 400)
        zbuf = np.full((PROXY_HEIGHT, PROXY_WIDTH), 2.0, dtype=np.float32)
        zbuf[:40, :] = np.inf
        mask = np.asarray(
            _face_mask(camera, [("sela", joints)], ["sela"], zbuf).convert("L")
        )
        self.assertGreater(int(mask[(top + bottom) // 2, (left + right) // 2]), 200)
        self.assertEqual(int(mask[15, (left + right) // 2]), 0)
        self.assertEqual(int(mask[:40].max()), 0)

        wide = next(item for item in self.episode["scenes"] if item["sceneNumber"] == 5)
        wide_time = float(wide["timeRangeSeconds"][0])
        wide_camera = camera_at(wide, wide_time)
        wide_state = episode_character_state(self.episode, "vardan", wide_time)
        wide_joints = character_pose_joints(
            self.show, self.episode, wide, wide_state, wide_time, "vardan"
        )
        wide_eye = project(wide_joints["left_eye"], wide_camera)
        wide_box = _face_ellipse(wide_joints, wide_camera)
        assert wide_eye is not None and wide_box is not None
        self.assertLess(wide_box[1], wide_eye[1])
        self.assertGreater(wide_box[3], wide_eye[1])


if __name__ == "__main__":
    unittest.main()
