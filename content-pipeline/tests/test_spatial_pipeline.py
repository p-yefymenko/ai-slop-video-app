from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import generate_batch as pipeline  # noqa: E402
from spatial_previs import (  # noqa: E402
    camera_at,
    spatial_target_screen_position,
    visible_landmark_line,
    visible_place_line,
    visible_setting_line,
)

SHOW_JSON = (
    Path(__file__).resolve().parents[1] / "shows" / "the-iron-bride" / "script.json"
)


class SpatialPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.show = pipeline.load_show(SHOW_JSON)
        cls.episode = cls.show["episodes"][0]
        cls.qwen = json.loads(
            (cls.root / "workflows" / "qwen_image_edit.json").read_text()
        )
        cls.qwen_spatial = json.loads(
            (cls.root / "workflows" / "qwen_image_edit_spatial.json").read_text()
        )
        cls.ltx = json.loads(
            (cls.root / "workflows" / "ltx_gemma_api.json").read_text()
        )

    def test_loader_preserves_spatial_ground_truth(self) -> None:
        self.assertIsNotNone(self.show["locations"]["sun_well_court"]["spatial"])
        self.assertIsNotNone(self.episode["spatialTimeline"])
        scene = next(
            item
            for item in self.episode["scenes"]
            if len(item["camera"]["keyframes"]) > 1
        )
        self.assertEqual(len(scene["timeRangeSeconds"]), 2)
        self.assertIn("verticalFovDegrees", scene["camera"]["keyframes"][0])

    def test_show_json_contains_no_renderer_templates_or_legacy_prose_state(self) -> None:
        raw = json.loads(SHOW_JSON.read_text(encoding="utf-8"))
        self.assertNotIn("prompts", raw)
        obsolete = {
            "beatType",
            "coverageRole",
            "continuityIn",
            "continuityOut",
            "addresseeId",
            "screenDirection",
            "shotType",
            "durationSeconds",
            "endGuideFrame",
            "logline",
            "dramaticQuestion",
            "coverageReferenceSceneNumber",
            "focusTargetId",
            "motionMode",
            "endPosition",
            "endLookAt",
        }
        self.assertTrue(obsolete.isdisjoint(raw["episodes"][0]))
        for scene in raw["episodes"][0]["scenes"]:
            self.assertTrue(obsolete.isdisjoint(scene))
            if "speakerId" in scene:
                self.assertIsInstance(scene["speakerId"], str)

    def _inject_png(self, directory: Path, name: str) -> Path:
        path = directory / name
        pipeline.write_black_png(path, 8, 8)
        return path

    def test_two_shot_restyles_the_blockout_and_not_a_wireframe(self) -> None:
        scene = next(
            item
            for item in self.episode["scenes"]
            if len(item["characterIds"]) == 2
        )
        graph = pipeline.clone_workflow(self.qwen_spatial)
        with tempfile.TemporaryDirectory() as temp:
            temp_dir = Path(temp)
            characters = [
                {
                    "id": character_id,
                    "image_path": self._inject_png(temp_dir, f"{character_id}.png"),
                }
                for character_id in scene["characterIds"]
            ]
            pipeline.inject_qwen_spatial_refs(
                graph,
                characters,
                "scene_depth.png",
                "scene_faces.png",
                [("scene_pose.png", "Pose"), ("scene_edges.png", "Edges")],
            )
        loaders = [
            node for node in graph.values() if node.get("class_type") == "LoadImage"
        ]
        self.assertEqual(len(loaders), 6)
        self.assertFalse(any("proxy" in node["inputs"]["image"] for node in loaders))
        self.assertEqual(graph["6"]["inputs"]["image"], "scene_depth.png")
        self.assertEqual(graph["40"]["inputs"]["image"], "scene_pose.png")
        self.assertEqual(graph["41"]["inputs"]["image"], "scene_edges.png")
        blockout = pipeline._qwen_encoder(graph, "Blockout instruction")
        assert blockout is not None
        self.assertEqual(
            set(blockout["inputs"]) & {"image1", "image2", "image3"},
            {"image1", "image2", "image3"},
        )

    def test_loader_allows_group_dialogue_and_more_than_two_people(self) -> None:
        groups = [
            scene
            for scene in self.episode["scenes"]
            if len(scene["characterIds"]) >= 3 and scene.get("speakerId")
        ]
        self.assertTrue(groups)

    def test_crowd_still_attaches_two_identities_only(self) -> None:
        graph = pipeline.clone_workflow(self.qwen_spatial)
        with tempfile.TemporaryDirectory() as temp:
            temp_dir = Path(temp)
            characters = [
                {
                    "id": f"person_{index}",
                    "image_path": self._inject_png(temp_dir, f"person_{index}.png"),
                }
                for index in range(4)
            ]
            pipeline.inject_qwen_spatial_refs(
                graph, characters, "scene_blockout.png", "scene_faces.png"
            )
        loaders = [
            node for node in graph.values() if node.get("class_type") == "LoadImage"
        ]
        self.assertEqual(len(loaders), 4)
        face = pipeline._qwen_encoder(graph, "Face instruction")
        assert face is not None
        self.assertEqual(set(face["inputs"]) & {"image2", "image3"}, {"image2", "image3"})

    def test_single_uses_identity_only(self) -> None:
        scene = next(
            item
            for item in self.episode["scenes"]
            if len(item["characterIds"]) == 1
        )
        graph = pipeline.clone_workflow(self.qwen_spatial)
        with tempfile.TemporaryDirectory() as temp:
            temp_dir = Path(temp)
            pipeline.inject_qwen_spatial_refs(
                graph,
                [
                    {
                        "id": scene["characterIds"][0],
                        "image_path": self._inject_png(temp_dir, "identity.png"),
                    }
                ],
                "scene_blockout.png",
                "scene_faces.png",
            )
        loaders = [
            node for node in graph.values() if node.get("class_type") == "LoadImage"
        ]
        self.assertEqual(len(loaders), 3)
        face = pipeline._qwen_encoder(graph, "Face instruction")
        assert face is not None
        self.assertNotIn("image3", face["inputs"])
        self.assertNotIn("spatialEnvironmentStill", self.show["prompts"])
        self.assertNotIn("spatialGroupEndStill", self.show["prompts"])
        self.assertNotIn("characterProfile", self.show["prompts"])

    def test_environment_shot_uses_depth_and_edges(self) -> None:
        graph = pipeline.clone_workflow(self.qwen_spatial)
        pipeline.inject_qwen_spatial_refs(
            graph,
            [],
            "scene_depth.png",
            None,
            [("scene_edges.png", "Edges")],
        )
        loaders = [
            node for node in graph.values() if node.get("class_type") == "LoadImage"
        ]
        self.assertEqual(
            [node["inputs"]["image"] for node in loaders],
            ["scene_depth.png", "scene_edges.png"],
        )
        self.assertEqual(graph["12"]["inputs"]["images"], ["11", 0])
        self.assertNotIn("30", graph)
        self.assertIn("spatialBlockout", self.show["prompts"])
        self.assertIn("spatialFaces", self.show["prompts"])
        self.assertIn("spatialBackdrop", self.show["prompts"])
        self.assertNotIn("spatialStill", self.show["prompts"])

    def test_end_guides_follow_spatial_change(self) -> None:
        moving = next(
            scene
            for scene in self.episode["scenes"]
            if pipeline.scene_needs_end_guide(self.episode, scene)
        )
        static_insert = next(
            scene
            for scene in self.episode["scenes"]
            if not scene.get("speakerId")
            and len(scene["characterIds"]) <= 1
            and not pipeline.scene_needs_end_guide(self.episode, scene)
        )
        static_dialogue = next(
            scene
            for scene in self.episode["scenes"]
            if scene.get("speakerId")
            and not pipeline.scene_needs_end_guide(self.episode, scene)
        )
        self.assertTrue(pipeline.scene_needs_end_guide(self.episode, moving))
        self.assertFalse(pipeline.scene_needs_end_guide(self.episode, static_insert))
        self.assertFalse(pipeline.scene_needs_end_guide(self.episode, static_dialogue))

    def test_dialogue_uses_multimodal_guidance(self) -> None:
        graph = pipeline.inject_prompt(self.ltx, "test", "test-key")
        pipeline.inject_dialogue_multimodal_guider(graph)
        self.assertEqual(graph["17"]["class_type"], "MultimodalGuider")
        self.assertEqual(graph["29"]["inputs"]["modality_scale"], 3.0)
        self.assertEqual(graph["30"]["inputs"]["modality"], "AUDIO")

    def test_landmark_without_a_position_is_not_a_screen_point(self) -> None:
        coverage = next(
            scene
            for scene in self.episode["_allScenes"]
            if scene["locationId"] == "sun_well_court" and not scene["characterIds"]
        )
        show = copy.deepcopy(self.show)
        show["locations"]["sun_well_court"]["spatial"]["landmarks"]["sun_well"] = {}
        with self.assertRaises(ValueError) as caught:
            spatial_target_screen_position(show, self.episode, coverage, "sun_well")
        self.assertIn("no position", str(caught.exception))

    def test_a_landmark_position_projects_inside_the_frame(self) -> None:
        coverage = next(
            scene
            for scene in self.episode["_allScenes"]
            if scene["locationId"] == "sun_well_court" and not scene["characterIds"]
        )
        show = copy.deepcopy(self.show)
        show["locations"]["sun_well_court"]["spatial"]["landmarks"]["sun_well"] = {
            "position": [0, 0, 1]
        }
        x, y = spatial_target_screen_position(show, self.episode, coverage, "sun_well")
        self.assertGreater(x, 0)
        self.assertLess(x, 768)
        self.assertGreater(y, 0)
        self.assertLess(y, 1360)

    def test_end_still_does_not_edit_the_previs_drawing(self) -> None:
        self.assertNotIn("spatialEndStill", pipeline.PROMPT_KEYS)
        self.assertNotIn("spatialEndStill", self.show["prompts"])
        self.assertFalse(hasattr(pipeline, "inject_qwen_end_refs"))
        graph = pipeline.clone_workflow(self.qwen_spatial)
        pipeline.inject_qwen_spatial_refs(
            graph,
            [],
            "scene_depth.png",
            None,
            [("scene_edges.png", "Edges")],
        )
        loaders = [
            node for node in graph.values() if node.get("class_type") == "LoadImage"
        ]
        self.assertEqual(
            [node["inputs"]["image"] for node in loaders],
            ["scene_depth.png", "scene_edges.png"],
        )

    def test_people_prompt_keeps_the_person_volume_in_the_depth(self) -> None:
        text = pipeline.structure_pictures(["Depth", "Clothes", "Edges"], people_count=1)
        self.assertTrue(text.startswith("Picture 1 is depth:"))
        self.assertIn("black is empty space", text)
        self.assertIn("Picture 2 is the person with flat color fills", text)
        self.assertIn("Picture 3 is outlines", text)
        self.assertNotIn("Keep those outlines", text)
        self.assertNotIn("Copy those colors", text)
        self.assertNotIn("It contains no people", text)
        empty_edges = pipeline.structure_pictures(["Depth", "Edges"])
        self.assertEqual(empty_edges, pipeline.structure_pictures(["Depth", "Edges"]))
        self.assertIn("Picture 2 is outlines", empty_edges)
        self.assertNotIn("It contains no people", empty_edges)
        self.assertNotIn("Black is an asset", empty_edges)
        self.assertNotIn("Do not", empty_edges)
        pose = pipeline.structure_pictures(["Depth", "Pose", "Edges"])
        self.assertIn("Picture 2 is an OpenPose skeleton", pose)
        self.assertNotIn("Do not", pose)
        backdrop = pipeline.structure_pictures(["Depth", "Edges", "Backdrop"])
        self.assertIn(
            "Picture 3 is flat sky, ground, and surround colors; the black shapes are the structures.",
            backdrop,
        )
        self.assertNotIn("36 42 58", backdrop)
        self.assertNotIn("Black is an asset", backdrop)
        self.assertNotIn("Do not", backdrop)

    def test_people_shots_pass_clothes_even_when_they_fill_the_frame(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            depth = root / "start_depth.png"
            clothes = root / "start_clothes.png"
            pose = root / "start_pose.png"
            edges = root / "start_edges.png"
            normal = root / "start_normal.png"
            size = (96, 170)
            import random

            random.seed(1)
            pose_image = Image.new("RGB", size)
            pose_image.putdata(
                [(random.randrange(256),) * 3 for _ in range(size[0] * size[1])]
            )
            pose_image.save(pose)
            for path in (depth, edges, normal):
                Image.new("RGB", size, (40, 40, 40)).save(path)
            close_clothes = Image.new("RGB", size)
            close_clothes.putdata(
                [
                    (
                        200 + random.randrange(40),
                        180 + random.randrange(40),
                        150 + random.randrange(40),
                    )
                    for _ in range(size[0] * size[1])
                ]
            )
            close_clothes.save(clothes)
            close = pipeline.spatial_still_pictures(
                has_characters=True,
                depth_path=depth,
                clothes_path=clothes,
                pose_path=pose,
                edge_path=edges,
            )
            self.assertEqual([title for _path, title in close], ["Depth", "Clothes", "Edges"])
            wide = Image.new("RGB", size, (0, 0, 0))
            for x in range(20):
                for y in range(20):
                    wide.putpixel(
                        (x, y),
                        (random.randrange(256), random.randrange(256), random.randrange(256)),
                    )
            wide.save(clothes)
            held = pipeline.spatial_still_pictures(
                has_characters=True,
                depth_path=depth,
                clothes_path=clothes,
                pose_path=pose,
                edge_path=edges,
            )
            self.assertEqual([title for _path, title in held], ["Depth", "Clothes", "Edges"])
            empty = pipeline.spatial_still_pictures(
                has_characters=False,
                depth_path=depth,
                clothes_path=clothes,
                pose_path=pose,
                edge_path=edges,
            )
            self.assertEqual([title for _path, title in empty], ["Depth", "Edges"])

    def test_empty_prompt_lets_the_depth_set_the_camera(self) -> None:
        text = pipeline.structure_pictures(["Depth", "Edges"])
        self.assertTrue(text.startswith("Picture 1 is depth: brighter is closer, black is empty space."))
        self.assertIn("Picture 2 is outlines", text)
        self.assertNotIn("OpenPose", text)
        self.assertNotIn("Do not", text)

    def test_spatial_still_generates_from_the_depth(self) -> None:
        graph = pipeline.clone_workflow(self.qwen_spatial)
        scene = next(item for item in self.episode["scenes"] if len(item["characterIds"]) >= 2)
        with tempfile.TemporaryDirectory() as temp:
            temp_dir = Path(temp)
            characters = [
                {
                    "id": character_id,
                    "image_path": self._inject_png(temp_dir, f"{character_id}.png"),
                }
                for character_id in scene["characterIds"][:2]
            ]
            pipeline.inject_qwen_spatial_refs(
                graph,
                characters,
                "scene_depth.png",
                "scene_faces.png",
                [("scene_pose.png", "Pose"), ("scene_edges.png", "Edges")],
            )
            pipeline.inject_seed(graph, 17)
        self.assertEqual(graph["4"]["inputs"]["lora_name"], "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")
        self.assertEqual(graph["5"]["inputs"]["shift"], 3.1)
        self.assertEqual(graph["9"]["class_type"], "VAEEncode")
        self.assertEqual(graph["9"]["inputs"]["pixels"], ["6", 0])
        self.assertEqual(graph["10"]["inputs"]["steps"], 4)
        self.assertEqual(graph["10"]["inputs"]["cfg"], 1.0)
        self.assertEqual(graph["10"]["inputs"]["denoise"], 1.0)
        self.assertEqual(graph["10"]["inputs"]["latent_image"], ["9", 0])
        self.assertEqual(graph["10"]["inputs"]["positive"], ["7", 0])
        self.assertEqual(graph["10"]["inputs"]["seed"], 17)
        self.assertEqual(graph["6"]["inputs"]["image"], "scene_depth.png")
        self.assertEqual(graph["40"]["inputs"]["image"], "scene_pose.png")
        self.assertEqual(graph["41"]["inputs"]["image"], "scene_edges.png")
        self.assertEqual(graph["30"]["inputs"]["denoise"], 1.0)
        self.assertEqual(graph["30"]["inputs"]["seed"], 17)
        self.assertEqual(graph["23"]["class_type"], "SetLatentNoiseMask")
        self.assertEqual(graph["12"]["inputs"]["images"], ["31", 0])
        self.assertFalse(hasattr(pipeline, "set_pose_control_strength"))

    def test_end_guide_is_added_before_av_sampling_and_cropped(self) -> None:
        graph = pipeline.inject_prompt(self.ltx, "test", "test-key")
        pipeline.inject_scene_length(graph, duration_seconds=5)
        pipeline.inject_start_frame(graph, "start.png")
        pipeline.inject_end_frame(graph, "end.png")
        self.assertEqual(graph["27"]["class_type"], "LTXVAddGuide")
        self.assertEqual(graph["24"]["inputs"]["video_latent"], ["27", 2])
        self.assertEqual(graph["23"]["inputs"]["frames_number"], 129)
        self.assertEqual(graph["28"]["class_type"], "LTXVCropGuides")
        self.assertEqual(graph["25"]["inputs"]["av_latent"], ["6", 0])
        self.assertEqual(graph["28"]["inputs"]["latent"], ["25", 0])
        self.assertEqual(graph["8"]["inputs"]["samples"], ["28", 2])
        self.assertEqual(graph["17"]["inputs"]["conditioning"], ["27", 0])

    def test_dialogue_end_guide_does_not_pass_conditioning(self) -> None:
        graph = pipeline.inject_prompt(self.ltx, "test", "test-key")
        pipeline.inject_dialogue_multimodal_guider(graph)
        pipeline.inject_scene_length(graph, duration_seconds=5)
        pipeline.inject_start_frame(graph, "start.png")
        pipeline.inject_end_frame(graph, "end.png")
        self.assertEqual(graph["17"]["class_type"], "MultimodalGuider")
        self.assertNotIn("conditioning", graph["17"]["inputs"])
        self.assertEqual(graph["17"]["inputs"]["positive"], ["27", 0])
        self.assertEqual(graph["17"]["inputs"]["negative"], ["27", 1])

    def test_visible_faces_are_painted_in_pairs_not_script_order(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mask = root / "start_faces.png"
            Image.new("L", (4, 4), 0).save(mask)
            ids = ["vardan", "nira", "kesh", "rhel"]
            (root / "start_faces.json").write_text(json.dumps(ids), encoding="utf-8")
            import random

            random.seed(0)
            spot_pixels = [random.randrange(1, 256) for _ in range(320 * 320)]
            characters = []
            for character_id in ["sela", "tomas", *ids]:
                portrait = root / f"test_face_{character_id}.png"
                Image.new("RGB", (4, 4), (255, 0, 0)).save(portrait)
                spot = root / f"start_faces_{character_id}.png"
                saved = Image.new("L", (320, 320))
                saved.putdata(spot_pixels)
                saved.save(spot)
                characters.append({"id": character_id, "image_path": portrait})
            groups = pipeline.identity_face_groups(mask, characters)
            self.assertEqual(
                [[item["id"] for item in group] for group in groups],
                [["vardan", "nira"], ["kesh", "rhel"]],
            )
            graph = pipeline.clone_workflow(self.qwen_spatial)
            pipeline.inject_qwen_spatial_refs(graph, groups[1], "shot.png", "faces.png")
            pipeline._face_pass_reads_loaded_shot(graph)
        face = pipeline._qwen_encoder(graph, "Face instruction")
        assert face is not None
        self.assertEqual(face["inputs"]["image1"], ["6", 0])
        self.assertEqual(graph["22"]["inputs"]["pixels"], ["6", 0])
        self.assertEqual(graph["6"]["inputs"]["image"], "shot.png")

    def test_backdrop_label_is_a_fourth_picture_and_not_a_face_input(self) -> None:
        graph = pipeline.clone_workflow(self.qwen_spatial)
        pipeline.inject_qwen_spatial_refs(
            graph,
            [],
            "scene_depth.png",
            None,
            [("scene_edges.png", "Edges")],
            backdrop_name="scene_backdrop.png",
        )
        blockout = pipeline._qwen_encoder(graph, "Blockout instruction")
        assert blockout is not None
        self.assertEqual(blockout["class_type"], "TextEncodeQwenBackdrop")
        self.assertEqual(blockout["inputs"]["image3"], ["42", 0])
        self.assertNotIn("image4", blockout["inputs"])
        self.assertEqual(graph["42"]["inputs"]["image"], "scene_backdrop.png")
        self.assertNotIn("70", graph)
        self.assertNotIn("32", graph)

    def test_clothes_shot_keeps_the_cutout_and_drops_the_backdrop(self) -> None:
        graph = pipeline.clone_workflow(self.qwen_spatial)
        pipeline.inject_qwen_spatial_refs(
            graph,
            [],
            "scene_depth.png",
            None,
            [("scene_clothes.png", "Clothes"), ("scene_edges.png", "Edges")],
            backdrop_name="scene_backdrop.png",
        )
        blockout = pipeline._qwen_encoder(graph, "Blockout instruction")
        assert blockout is not None
        self.assertEqual(blockout["class_type"], "TextEncodeQwenImageEditPlus")
        self.assertEqual(blockout["inputs"]["image2"], ["40", 0])
        self.assertNotIn("42", graph)
        self.assertNotIn("image4", blockout["inputs"])
        self.assertNotIn("latent2", blockout["inputs"])

    def test_face_pass_is_off(self) -> None:
        self.assertFalse(pipeline.FACE_PASS)

    def test_backdrop_followup_fills_empty_space_from_the_plate(self) -> None:
        graph = pipeline.clone_workflow(self.qwen_spatial)
        pipeline.inject_backdrop_followup(
            graph, "shot.png", "empty.png", "scene_backdrop.png"
        )
        encoder = pipeline._qwen_encoder(graph, "Backdrop instruction")
        assert encoder is not None
        self.assertEqual(encoder["class_type"], "TextEncodeQwenBackdrop")
        self.assertEqual(encoder["inputs"]["image1"], ["6", 0])
        self.assertEqual(encoder["inputs"]["image2"], ["42", 0])
        self.assertNotIn("image3", encoder["inputs"])
        self.assertNotIn("image4", encoder["inputs"])
        self.assertEqual(graph["6"]["inputs"]["image"], "shot.png")
        self.assertEqual(graph["20"]["inputs"]["image"], "empty.png")
        self.assertEqual(graph["42"]["inputs"]["image"], "scene_backdrop.png")
        self.assertEqual(graph["22"]["inputs"]["pixels"], ["6", 0])
        self.assertEqual(graph["12"]["inputs"]["images"], ["31", 0])
        self.assertNotIn("10", graph)
        self.assertNotIn("11", graph)
        self.assertNotIn("24", graph)
        self.assertNotIn("25", graph)

    def test_invented_empty_space_is_replaced_with_the_backdrop_plate(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            still = Image.new("RGB", (4, 2), (200, 180, 160))
            still.save(root / "still.png")
            plate = Image.new("RGB", (4, 2), (16, 42, 62))
            plate.putpixel((3, 0), (0, 0, 0))
            plate.putpixel((3, 1), (0, 0, 0))
            plate.save(root / "plate.png")
            dest = root / "stripped.png"
            pipeline.strip_invented_backdrop(root / "still.png", root / "plate.png", dest)
            out = Image.open(dest).convert("RGB")
            self.assertEqual(out.getpixel((0, 0)), (16, 42, 62))
            self.assertEqual(out.getpixel((3, 0)), (200, 180, 160))
            mask = pipeline.empty_space_mask(plate)
            self.assertEqual(mask.getpixel((0, 0)), 255)
            self.assertEqual(mask.getpixel((3, 0)), 0)

    def test_backdrop_pass_numbers_the_plate_as_picture_two(self) -> None:
        text = pipeline.structure_pictures(["Backdrop"], start=2)
        self.assertTrue(
            text.startswith(
                "Picture 2 is flat sky, ground, and surround colors; the black shapes are the structures."
            )
        )
        self.assertNotIn("Do not", text)
        self.assertIn("spatialBackdrop", pipeline.PROMPT_KEYS)

    def test_backdrop_sentence_omits_regions_this_camera_does_not_show(self) -> None:
        from PIL import Image

        court = self.show["locations"]["sun_well_court"]["backdrop"]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "backdrop.png"
            Image.new("RGB", (64, 64), tuple(court["skyColor"])).save(path)
            close = next(item for item in self.episode["scenes"] if item["sceneNumber"] == 15)
            text = visible_place_line(
                self.show["locations"]["sun_well_court"],
                camera_at(close, float(close["timeRangeSeconds"][0])),
                path,
            )
        self.assertIn("storm sky", text.lower())
        self.assertNotIn("iron floor", text.lower())
        self.assertNotIn("open ocean", text.lower())

    def test_blockout_pass_is_saved_beside_the_face_pass(self) -> None:
        dest = Path("scene_03_start.png")
        self.assertEqual(
            pipeline.kept_pass_path(dest, "blockout"),
            Path("scene_03_start_blockout.png"),
        )
        self.assertEqual(
            pipeline.kept_pass_path(dest, "face1"),
            Path("inputs") / "scene_03_start" / "face1.png",
        )
        files = [
            {"filename": "reelshort_blockout_00001_.png"},
            {"filename": "reelshort_start_00002_.png"},
        ]
        self.assertTrue(
            pipeline._has_output_prefixes(files, ["reelshort_blockout", "reelshort_start"])
        )
        self.assertEqual(
            pipeline.output_named(files, "reelshort_blockout")["filename"],
            "reelshort_blockout_00001_.png",
        )
        graph = pipeline.clone_workflow(self.qwen_spatial)
        pipeline._save_blockout_pass(graph)
        self.assertEqual(graph["32"]["inputs"]["images"], ["11", 0])
        self.assertEqual(graph["32"]["inputs"]["filename_prefix"], "reelshort_blockout")
        self.assertEqual(graph["12"]["inputs"]["images"], ["31", 0])

    def test_people_blockout_does_not_name_anyone_or_their_clothes(self) -> None:
        location = self.show["locations"]["sun_well_court"]
        scene = next(item for item in self.episode["scenes"] if item["sceneNumber"] == 6)
        prompt = pipeline.still_prompt(
            ["Depth", "Clothes", "Edges"],
            people=pipeline.still_people_line(scene["characterIds"]),
            landmarks=visible_place_line(
                location,
                camera_at(scene, float(scene["timeRangeSeconds"][0])),
            ),
            people_count=len(scene["characterIds"]),
        )
        self.assertIn("Exactly 6 people are visible", prompt)
        self.assertNotIn(location["promptBlock"], prompt)
        self.assertIn("white-gold", prompt)
        self.assertNotIn("sela", prompt.lower())
        self.assertNotIn("vardan", prompt.lower())
        self.assertNotIn(scene["imagePrompt"], prompt)
        self.assertNotIn("sun-priest plate", prompt)
        self.assertNotIn("It contains no people", prompt)
        empty = next(item for item in self.episode["scenes"] if not item["characterIds"])
        self.assertEqual(pipeline.still_people_line([]), "")
        self.assertNotIn("No people", prompt)
        self.assertNotIn(empty["imagePrompt"], prompt)
        solo = next(item for item in self.episode["scenes"] if item["sceneNumber"] == 11)
        self.assertFalse(
            pipeline.head_in_view(
                self.show,
                self.episode,
                solo,
                "sela",
                float(solo["timeRangeSeconds"][0]),
            )
        )
        with tempfile.TemporaryDirectory() as temp:
            from PIL import Image

            clothes = Path(temp) / "clothes.png"
            image = Image.new("RGB", (30, 30), (0, 0, 0))
            for y in range(30):
                for x in range(20):
                    image.putpixel((x, y), (220, 196, 160))
                for x in range(20, 26):
                    image.putpixel((x, y), (112, 72, 55))
            image.save(clothes)
            solo_line = pipeline.still_people_line(
                solo["characterIds"],
                self.show["characters"],
                clothes=True,
                clothes_path=clothes,
                head_visible=False,
            )
        self.assertIn("Exactly one person is visible", solo_line)
        self.assertIn("adult woman", solo_line)
        self.assertIn("sand-colored wrap with darker sand stains", solo_line)
        self.assertIn("warm brown skin", solo_line)
        self.assertIn("picture 2", solo_line)
        self.assertNotIn("ivory", solo_line.lower())
        self.assertNotIn("olive", solo_line.lower())
        self.assertNotIn("collar", solo_line.lower())
        self.assertNotIn("soot", solo_line.lower())
        self.assertNotIn("sela", solo_line.lower())
        self.assertNotIn(
            "sun-priest plate",
            pipeline.still_people_line(
                scene["characterIds"],
                self.show["characters"],
                clothes=True,
            ),
        )

    def test_every_shot_uses_the_same_prompt_skeleton(self) -> None:
        from PIL import Image

        keep = "Keep the shape, position, and occlusion from the pictures."
        prompts = []
        cases = (
            (1, ["Depth", "Edges", "Backdrop"]),
            (5, ["Depth", "Clothes", "Edges"]),
            (11, ["Depth", "Clothes", "Edges"]),
        )
        for number, pictures in cases:
            scene = next(item for item in self.episode["scenes"] if item["sceneNumber"] == number)
            location = self.show["locations"][scene["locationId"]]
            camera = camera_at(scene, float(scene["timeRangeSeconds"][0]))
            with tempfile.TemporaryDirectory() as temp:
                plate = Path(temp) / "backdrop.png"
                image = Image.new("RGB", (90, 90), tuple(location["backdrop"]["skyColor"]))
                ground = tuple(location["backdrop"]["groundColor"])
                surround = tuple(location["backdrop"]["surroundColor"])
                for y in range(30, 60):
                    for x in range(90):
                        image.putpixel((x, y), ground)
                for y in range(60, 90):
                    for x in range(90):
                        image.putpixel((x, y), surround)
                image.save(plate)
                landmarks = visible_landmark_line(location, camera)
                setting = visible_setting_line(location, plate, landmarks.lower())
            prompt = pipeline.still_prompt(
                pictures,
                people=pipeline.still_people_line(
                    scene["characterIds"],
                    self.show["characters"],
                    clothes="Clothes" in pictures,
                ),
                landmarks=landmarks,
                setting=setting,
                people_count=len(scene["characterIds"]),
            )
            prompts.append(prompt)
            self.assertTrue(prompt.startswith("Photorealistic vertical 9:16 film frame. Picture 1 is depth:"))
            self.assertEqual(prompt.count("Picture "), len(pictures))
            self.assertIn(keep, prompt)
            self.assertNotIn("Do not", prompt)
            self.assertNotIn("No people", prompt)
            self.assertNotIn("Setting:", prompt)
            self.assertNotIn(scene.get("imagePrompt") or "___missing___", prompt)
            if setting:
                self.assertLess(prompt.index(keep), prompt.index("Behind and around:"))
            if landmarks:
                self.assertLess(prompt.index(keep), prompt.index(landmarks[:24]))
        empty, wide, solo = prompts
        self.assertIn("black shapes are the structures", empty)
        self.assertNotIn("Exactly", empty)
        self.assertIn("Exactly 6 people are visible", wide)
        self.assertNotIn("black shapes are the structures", wide)
        self.assertIn("Picture 2 is these people with flat color fills", wide)
        self.assertIn("Exactly one person is visible", solo)
        self.assertIn("Picture 2 is the person with flat color fills", solo)
        self.assertNotIn("black shapes are the structures", solo)

    def test_generation_log_copies_the_prompt_and_attached_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            dest = Path(temp) / "scene_02_start.png"
            depth = Path(temp) / "start_depth.png"
            depth.write_bytes(b"depth-bytes")
            pipeline.begin_generation_log(dest)
            pipeline.append_generation_log(
                dest,
                "blockout",
                "An open basalt court around a sun-well of white-gold fire.",
                [("Depth", depth)],
            )
            log = json.loads(pipeline.still_log_path(dest).read_text(encoding="utf-8"))
            self.assertEqual(log["still"], "scene_02_start.png")
            self.assertEqual(log["passes"][0]["prompt"], "An open basalt court around a sun-well of white-gold fire.")
            copied = pipeline.still_inputs_dir(dest) / "blockout_depth.png"
            self.assertTrue(copied.is_file())
            self.assertEqual(copied.read_bytes(), b"depth-bytes")
            self.assertEqual(log["passes"][0]["images"][0]["file"], "blockout_depth.png")
            self.assertEqual(
                pipeline.still_log_path(dest),
                Path(temp) / "inputs" / "scene_02_start" / "log.json",
            )
            self.assertFalse((Path(temp) / "scene_02_start_log.json").exists())

    def test_still_prompt_names_only_landmarks_this_camera_sees(self) -> None:
        location = self.show["locations"]["sun_well_court"]
        close = next(item for item in self.episode["scenes"] if item["sceneNumber"] == 15)
        close_line = visible_place_line(
            location, camera_at(close, float(close["timeRangeSeconds"][0]))
        )
        self.assertNotIn("sun well", close_line.lower())
        self.assertNotIn("sun-well", close_line.lower())
        self.assertNotIn("iron floor", close_line.lower())
        self.assertNotIn(location["promptBlock"], close_line)
        well = next(item for item in self.episode["scenes"] if item["sceneNumber"] == 2)
        well_line = visible_place_line(
            location, camera_at(well, float(well["timeRangeSeconds"][0]))
        )
        self.assertIn("white-gold", well_line.lower())
        self.assertNotIn(location["promptBlock"], well_line)

    def test_face_prompt_names_identity_and_not_the_place(self) -> None:
        location = self.show["locations"]["sun_well_court"]
        scene = next(item for item in self.episode["scenes"] if item["sceneNumber"] == 3)
        prompt = pipeline.face_prompt_for(
            self.show,
            {
                "characterCount": "1",
                "characterIds": "sela",
                "locationPromptBlock": location["promptBlock"],
                "imagePrompt": scene["imagePrompt"],
            },
            [{"id": "sela"}],
        )
        self.assertIn("Picture 2 = the face of sela only", prompt)
        self.assertNotIn(location["promptBlock"], prompt)
        self.assertNotIn(scene["imagePrompt"], prompt)
        self.assertNotIn("sun-well", prompt)


if __name__ == "__main__":
    unittest.main()
