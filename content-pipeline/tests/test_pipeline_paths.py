from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from pipeline_paths import (  # noqa: E402
    clay_frame_path,
    discover_show_scripts,
    guide_path,
    legacy_output_moves,
    show_id_for_script,
    start_still_path,
)


class PipelinePathTests(unittest.TestCase):
    def test_show_script_is_discovered_by_folder_name(self) -> None:
        scripts = discover_show_scripts("the-iron-bride")
        self.assertEqual(len(scripts), 1)
        self.assertEqual(show_id_for_script(scripts[0]), "the-iron-bride")
        self.assertEqual(scripts[0].name, "script.json")

    def test_stage_paths_keep_characters_outside_the_episode(self) -> None:
        self.assertEqual(
            clay_frame_path("the-iron-bride", 1, 2, "start").parts[-4:],
            ("1", "01_previs", "scene_02", "start.png"),
        )
        self.assertEqual(
            guide_path("the-iron-bride", 1, 2, "start", "faces").name,
            "start_faces.png",
        )
        self.assertIn("02_postvis", start_still_path("the-iron-bride", 1, 2).parts)

    def test_legacy_episode_files_map_onto_stage_folders(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            episode = Path(temp) / "the-iron-bride" / "1"
            previs = episode / "previs"
            previs.mkdir(parents=True)
            names = [
                previs / "scene_02_blockout.mp4",
                previs / "scene_02_start_blockout.png",
                previs / "scene_02_end_blockout.png",
                previs / "scene_02_start_faces.png",
                previs / "scene_02_start_condition.png",
                previs / "contact_sheet.png",
                episode / "scene_02_start.png",
                episode / "scene_02.mp4",
                episode / "episode.mp4",
                episode / "manifest.json",
            ]
            for path in names:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")
            moves = dict(legacy_output_moves(episode))
            self.assertEqual(moves[previs / "scene_02_blockout.mp4"].name, "blockout.mp4")
            self.assertEqual(
                moves[previs / "scene_02_start_blockout.png"],
                episode / "01_previs" / "scene_02" / "start.png",
            )
            self.assertEqual(
                moves[previs / "scene_02_start_faces.png"],
                episode / "01_previs" / "scene_02" / "guides" / "start_faces.png",
            )
            self.assertEqual(
                moves[previs / "scene_02_start_condition.png"],
                episode / "01_previs" / "scene_02" / "guides" / "start_condition.png",
            )
            self.assertEqual(
                moves[episode / "scene_02_start.png"],
                episode / "02_postvis" / "stills" / "scene_02_start.png",
            )
            self.assertEqual(
                moves[episode / "scene_02.mp4"],
                episode / "03_postvis" / "clips" / "scene_02.mp4",
            )
            self.assertEqual(
                moves[episode / "episode.mp4"],
                episode / "04_edit" / "episode.mp4",
            )
            self.assertNotIn(episode / "manifest.json", moves)


if __name__ == "__main__":
    unittest.main()
