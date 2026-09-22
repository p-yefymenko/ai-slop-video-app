from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import generate_batch as pipeline  # noqa: E402


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

    def test_single_uses_identity_master_and_proxy(self) -> None:
        scene = self.episode["scenes"][4]
        graph = pipeline.clone_workflow(self.qwen)
        proxy = pipeline.proxy_frame_path(self.show["id"], 1, 5, "start")
        pipeline.inject_qwen_spatial_refs(
            graph,
            pipeline.resolve_scene_characters(self.show, scene),
            proxy,
            proxy,
            (600.0, 600.0),
        )
        loaders = [
            node for node in graph.values() if node.get("class_type") == "LoadImage"
        ]
        self.assertEqual(len(loaders), 3)

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
