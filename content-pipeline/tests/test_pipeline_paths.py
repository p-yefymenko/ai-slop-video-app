from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from pipeline_paths import (  # noqa: E402
    clay_frame_path,
    discover_show_scripts,
    guide_path,
    show_id_for_script,
    start_still_path,
)


class PipelinePathTests(unittest.TestCase):
    def test_show_script_is_discovered_by_folder_name(self) -> None:
        scripts = discover_show_scripts("return-of-the-wolf")
        self.assertEqual(len(scripts), 1)
        self.assertEqual(show_id_for_script(scripts[0]), "return-of-the-wolf")
        self.assertEqual(scripts[0].name, "script.json")

    def test_stage_paths_follow_the_command(self) -> None:
        self.assertEqual(
            clay_frame_path("return-of-the-wolf", 1, 2, "start").parts[-6:],
            ("output", "previs", "return-of-the-wolf", "1", "scene_02", "start.png"),
        )
        self.assertEqual(
            guide_path("return-of-the-wolf", 1, 2, "start", "faces").name,
            "start_faces.png",
        )
        self.assertIn("frames", start_still_path("return-of-the-wolf", 1, 2).parts)


if __name__ == "__main__":
    unittest.main()
