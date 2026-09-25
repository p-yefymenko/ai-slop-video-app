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
from spatial_previs import spatial_target_screen_position  # noqa: E402

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
                graph, characters, "scene_blockout.png", "scene_faces.png"
            )
        loaders = [
            node for node in graph.values() if node.get("class_type") == "LoadImage"
        ]
        self.assertEqual(len(loaders), 4)
        self.assertFalse(any("proxy" in node["inputs"]["image"] for node in loaders))
        self.assertEqual(graph["6"]["inputs"]["image"], "scene_blockout.png")
        blockout = pipeline._qwen_encoder(graph, "Blockout instruction")
        assert blockout is not None
        self.assertEqual(
            set(blockout["inputs"]) & {"image1", "image2", "image3"},
            {"image1"},
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

    def test_environment_shot_restyles_the_blockout_only(self) -> None:
        graph = pipeline.clone_workflow(self.qwen_spatial)
        pipeline.inject_qwen_spatial_refs(graph, [], "scene_blockout.png", None)
        loaders = [
            node for node in graph.values() if node.get("class_type") == "LoadImage"
        ]
        self.assertEqual(
            [node["inputs"]["image"] for node in loaders],
            ["scene_blockout.png"],
        )
        self.assertEqual(graph["12"]["inputs"]["images"], ["11", 0])
        self.assertNotIn("30", graph)
        self.assertIn("spatialBlockout", self.show["prompts"])
        self.assertIn("spatialFaces", self.show["prompts"])
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
        with self.assertRaises(ValueError) as caught:
            spatial_target_screen_position(self.show, self.episode, coverage, "sun_well")
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
        pipeline.inject_qwen_spatial_refs(graph, [], "scene_blockout.png", None)
        loaders = [
            node for node in graph.values() if node.get("class_type") == "LoadImage"
        ]
        self.assertEqual(
            [node["inputs"]["image"] for node in loaders],
            ["scene_blockout.png"],
        )

    def test_spatial_still_restyles_the_clay_latent(self) -> None:
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
                graph, characters, "scene_blockout.png", "scene_faces.png"
            )
            pipeline.inject_seed(graph, 17)
        self.assertEqual(graph["4"]["inputs"]["lora_name"], "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")
        self.assertEqual(graph["5"]["inputs"]["shift"], 3.1)
        self.assertEqual(graph["9"]["class_type"], "VAEEncode")
        self.assertEqual(graph["9"]["inputs"]["pixels"], ["6", 0])
        self.assertEqual(graph["10"]["inputs"]["steps"], 4)
        self.assertEqual(graph["10"]["inputs"]["cfg"], 1.0)
        self.assertEqual(graph["10"]["inputs"]["denoise"], 0.65)
        self.assertEqual(graph["10"]["inputs"]["latent_image"], ["9", 0])
        self.assertEqual(graph["10"]["inputs"]["positive"], ["7", 0])
        self.assertEqual(graph["10"]["inputs"]["seed"], 17)
        self.assertEqual(graph["30"]["inputs"]["denoise"], 1.0)
        self.assertEqual(graph["30"]["inputs"]["seed"], 17)
        self.assertEqual(graph["23"]["class_type"], "SetLatentNoiseMask")
        self.assertEqual(graph["12"]["inputs"]["images"], ["31", 0])
        self.assertNotIn("40", graph)
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


if __name__ == "__main__":
    unittest.main()
