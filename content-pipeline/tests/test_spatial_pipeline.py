from __future__ import annotations

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
    Path(__file__).resolve().parents[1] / "scripts_input" / "the-iron-bride.json"
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

    def test_two_shot_uses_two_identities_and_proxy(self) -> None:
        scene = next(
            item
            for item in self.episode["scenes"]
            if len(item["characterIds"]) == 2
        )
        graph = pipeline.clone_workflow(self.qwen)
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
                self._inject_png(temp_dir, "proxy.png"),
            )
        loaders = [
            node for node in graph.values() if node.get("class_type") == "LoadImage"
        ]
        self.assertEqual(len(loaders), 3)

    def test_loader_allows_group_dialogue_and_more_than_two_people(self) -> None:
        groups = [
            scene
            for scene in self.episode["scenes"]
            if len(scene["characterIds"]) >= 3 and scene.get("speakerId")
        ]
        self.assertTrue(groups)

    def test_crowd_still_attaches_two_identities_and_proxy(self) -> None:
        graph = pipeline.clone_workflow(self.qwen)
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
                graph,
                characters,
                self._inject_png(temp_dir, "proxy.png"),
            )
        loaders = [
            node for node in graph.values() if node.get("class_type") == "LoadImage"
        ]
        self.assertEqual(len(loaders), 3)

    def test_single_uses_identity_and_proxy(self) -> None:
        scene = next(
            item
            for item in self.episode["scenes"]
            if len(item["characterIds"]) == 1
        )
        graph = pipeline.clone_workflow(self.qwen)
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
                self._inject_png(temp_dir, "proxy.png"),
            )
        loaders = [
            node for node in graph.values() if node.get("class_type") == "LoadImage"
        ]
        self.assertEqual(len(loaders), 2)
        self.assertNotIn("spatialEnvironmentStill", self.show["prompts"])
        self.assertNotIn("spatialGroupEndStill", self.show["prompts"])
        self.assertNotIn("characterProfile", self.show["prompts"])

    def test_environment_shot_uses_spatial_proxy(self) -> None:
        graph = pipeline.clone_workflow(self.qwen)
        with tempfile.TemporaryDirectory() as temp:
            pipeline.inject_qwen_spatial_refs(
                graph,
                [],
                self._inject_png(Path(temp), "proxy.png"),
            )
        loaders = [
            node for node in graph.values() if node.get("class_type") == "LoadImage"
        ]
        self.assertEqual(len(loaders), 1)
        self.assertIn("spatialStill", self.show["prompts"])
        self.assertNotIn("spatialEnvironmentStill", self.show["prompts"])

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

    def test_prop_insert_projects_sun_well(self) -> None:
        coverage = next(
            scene
            for scene in self.episode["_allScenes"]
            if scene["locationId"] == "sun_well_court" and not scene["characterIds"]
        )
        x, y = spatial_target_screen_position(
            self.show, self.episode, coverage, "sun_well"
        )
        self.assertGreater(x, 0)
        self.assertLess(x, 768)
        self.assertGreater(y, 0)
        self.assertLess(y, 1360)

    def test_end_still_uses_end_proxy_not_start_photo(self) -> None:
        self.assertNotIn("spatialEndStill", pipeline.PROMPT_KEYS)
        self.assertNotIn("spatialEndStill", self.show["prompts"])
        self.assertFalse(hasattr(pipeline, "inject_qwen_end_refs"))
        graph = pipeline.clone_workflow(self.qwen)
        with tempfile.TemporaryDirectory() as temp:
            proxy = self._inject_png(Path(temp), "scene_01_end_condition.png")
            pipeline.inject_qwen_spatial_refs(graph, [], proxy)
        loaders = [
            node for node in graph.values() if node.get("class_type") == "LoadImage"
        ]
        self.assertEqual(
            [node["inputs"]["image"] for node in loaders],
            ["proxy_scene_01_end_condition.png"],
        )

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
