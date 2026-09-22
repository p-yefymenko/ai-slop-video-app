from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import generate_batch as pipeline  # noqa: E402
from spatial_previs import spatial_target_screen_position  # noqa: E402


class SpatialPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.show = pipeline.load_show(
            cls.root / "scripts_input" / "crown-of-the-last-dragon.json"
        )
        cls.episode = cls.show["episodes"][0]
        cls.qwen = json.loads(
            (cls.root / "workflows" / "qwen_image_edit.json").read_text()
        )
        cls.ltx = json.loads(
            (cls.root / "workflows" / "ltx_gemma_api.json").read_text()
        )

    def test_loader_preserves_spatial_ground_truth(self) -> None:
        self.assertIsNotNone(
            self.show["locations"]["eclipse_throne_hall"]["spatial"]
        )
        self.assertIsNotNone(self.episode["spatialTimeline"])
        scene = self.episode["scenes"][3]
        self.assertEqual(scene["timeRangeSeconds"], [6.0, 8.0])
        self.assertEqual(scene["camera"]["verticalFovDegrees"], 70.0)

    def test_show_json_contains_no_renderer_templates_or_legacy_prose_state(self) -> None:
        raw = json.loads(
            (
                self.root / "scripts_input" / "crown-of-the-last-dragon.json"
            ).read_text(encoding="utf-8")
        )
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
        }
        self.assertTrue(obsolete.isdisjoint(raw["episodes"][0]))
        for scene in raw["episodes"][0]["scenes"]:
            self.assertTrue(obsolete.isdisjoint(scene))
            if "speakerId" in scene:
                self.assertIsInstance(scene["speakerId"], str)

    def test_master_uses_two_identities_and_proxy(self) -> None:
        scene = self.episode["scenes"][3]
        graph = pipeline.clone_workflow(self.qwen)
        pipeline.inject_qwen_spatial_refs(
            graph,
            pipeline.resolve_scene_characters(self.show, scene),
            pipeline.proxy_frame_path(self.show["id"], 1, 4, "start"),
        )
        loaders = [
            node for node in graph.values() if node.get("class_type") == "LoadImage"
        ]
        self.assertEqual(len(loaders), 3)

    def test_single_uses_identity_and_proxy(self) -> None:
        scene = self.episode["scenes"][4]
        graph = pipeline.clone_workflow(self.qwen)
        pipeline.inject_qwen_spatial_refs(
            graph,
            pipeline.resolve_scene_characters(self.show, scene),
            pipeline.proxy_frame_path(self.show["id"], 1, 5, "start_condition"),
        )
        loaders = [
            node for node in graph.values() if node.get("class_type") == "LoadImage"
        ]
        self.assertEqual(len(loaders), 2)
        self.assertNotIn("setPlate", self.show["prompts"])
        self.assertNotIn("spatialCoverageStill", self.show["prompts"])

    def test_environment_shot_uses_spatial_proxy(self) -> None:
        scene = self.episode["scenes"][0]
        graph = pipeline.clone_workflow(self.qwen)
        pipeline.inject_qwen_environment_proxy(
            graph,
            pipeline.proxy_frame_path(self.show["id"], 1, scene["sceneNumber"], "start_condition"),
        )
        loaders = [
            node for node in graph.values() if node.get("class_type") == "LoadImage"
        ]
        self.assertEqual(len(loaders), 1)
        self.assertIn("spatialEnvironmentStill", self.show["prompts"])
        self.assertNotIn("characterProfile", self.show["prompts"])

    def test_safe_shots_default_to_camera_only(self) -> None:
        insert = self.episode["scenes"][7]
        dialogue = self.episode["scenes"][9]
        self.assertEqual(pipeline.scene_motion_mode(insert), "cameraOnly")
        self.assertEqual(pipeline.scene_motion_mode(dialogue), "generative")

    def test_dialogue_uses_multimodal_guidance(self) -> None:
        graph = pipeline.inject_prompt(self.ltx, "test", "test-key")
        pipeline.inject_dialogue_multimodal_guider(graph)
        self.assertEqual(graph["17"]["class_type"], "MultimodalGuider")
        self.assertEqual(graph["29"]["inputs"]["modality_scale"], 3.0)
        self.assertEqual(graph["30"]["inputs"]["modality"], "AUDIO")

    def test_face_landmarks_detect_wrong_screen_direction(self) -> None:
        face = [0.0] * 15
        face[4], face[6], face[8] = 100.0, 200.0, 80.0
        self.assertEqual(pipeline.face_facing_direction(face), "left")
        face[8] = 220.0
        self.assertEqual(pipeline.face_facing_direction(face), "right")

    def test_prop_insert_projects_established_crown(self) -> None:
        coverage = self.episode["_allScenes"][3]
        x, y = spatial_target_screen_position(
            self.show, self.episode, coverage, "ash_crown"
        )
        self.assertGreater(x, 0)
        self.assertLess(x, 768)
        self.assertGreater(y, 0)
        self.assertLess(y, 1360)

    def test_two_character_end_refs_use_start_and_proxy(self) -> None:
        scene = self.episode["scenes"][3]
        self.assertEqual(pipeline.scene_motion_mode(scene), "cameraOnly")
        self.assertFalse(pipeline.scene_needs_end_guide(self.episode, scene))
        graph = pipeline.clone_workflow(self.qwen)
        output_dir = pipeline.OUTPUT_DIR / self.show["id"] / "1"
        pipeline.inject_qwen_group_end_refs(
            graph,
            pipeline.start_still_path(output_dir, scene["sceneNumber"]),
            pipeline.proxy_frame_path(
                self.show["id"], 1, scene["sceneNumber"], "end_condition"
            ),
        )
        loaders = [
            node for node in graph.values() if node.get("class_type") == "LoadImage"
        ]
        self.assertEqual(len(loaders), 2)

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


if __name__ == "__main__":
    unittest.main()
