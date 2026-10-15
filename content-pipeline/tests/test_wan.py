"""The Wan chain's logic: frame counts, the depth video, the reference sheet, the words, the voice track, the graphs."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import wan  # noqa: E402
from test_world import observation, scene, show_with  # noqa: E402

PRONOUNS = {"he", "him", "his", "she", "her", "they", "them", "it", "someone"}


def standing(x: float = 0.0, y: float = 0.0, yaw: float = 180.0) -> list[dict]:
    return [{"timeSeconds": 0, "locationId": "room", "position": [x, y, 0], "bodyYawDegrees": yaw}]


def two_people() -> dict:
    show = show_with({"ada": standing(), "bo": standing(0.4, -2.5, 0.0)})
    show["characters"]["bo"] = {"body": "old man, broad build", "attributes": ["a red fox-fur mantle"], "heightMeters": 1.8}
    return show


def over_the_shoulder(**extra) -> dict:
    return scene(
        shot={"type": "overShoulder", "subjects": ["ada"], "size": "mcu", "angle": "low"},
        performances={"ada": {"action": "speaks with a half-smile", "expression": "lips slightly parted"}},
        **extra,
    )


class FrameTests(unittest.TestCase):
    def test_a_take_covers_the_scene_in_4n_plus_1_frames(self) -> None:
        self.assertEqual(wan.frames_for(5.43), 89)
        self.assertEqual(wan.frames_for(1.0), 17)
        self.assertEqual(wan.frames_for(0.0), 1)
        for seconds in (0.5, 2.3, 3.7, 6.77):
            frames = wan.frames_for(seconds)
            self.assertEqual((frames - 1) % 4, 0)
            self.assertGreaterEqual(frames / wan.FPS, seconds - 1 / wan.FPS, "never shorter than the scene")


class DepthTests(unittest.TestCase):
    def test_one_fixed_log_scale_for_every_shot(self) -> None:
        from shot_inputs import depth_gray

        gray = depth_gray(np.array([[0.1, 0.5, 60.0, 1000.0, np.inf]], dtype=np.float32))
        self.assertEqual(gray.tolist(), [[255, 255, 48, 48, 0]], "nearer than 0.5 m is brightest, past 60 m darkest, nothing is black")
        middle = depth_gray(np.array([[np.sqrt(0.5 * 60.0)]], dtype=np.float32))[0, 0]
        self.assertAlmostEqual(int(middle), (48 + 255) / 2, delta=1, msg="halfway in log distance is halfway in gray")

    def test_a_person_is_a_flat_silhouette_at_their_own_distance(self) -> None:
        from shot_inputs import flatten_people

        depth = np.array([[2.0, 2.5, 3.0, 9.0]], dtype=np.float32)
        ids = np.array([[1, 1, 1, 2]], dtype=np.uint8)
        flat = flatten_people(depth, ids, [1])
        self.assertEqual(flat.tolist(), [[2.5, 2.5, 2.5, 9.0]], "the face relief is gone; the landmark keeps its shape")


class ReferenceSheetTests(unittest.TestCase):
    def test_the_sheet_is_who_the_script_says_the_shot_is_about(self) -> None:
        from describe import reference_people

        show = two_people()
        self.assertEqual(reference_people(show, over_the_shoulder()), ["ada"], "the shoulder in front is left to the depth video")
        acting = scene(performances={"ada": {"action": "waits"}, "bo": {"action": "nods"}})
        self.assertEqual(reference_people(show, acting), [], "only subjects, never every performer")
        self.assertEqual(reference_people(show, scene()), [])

    def test_photos_side_by_side_on_white_as_drawn(self) -> None:
        from shot_inputs import reference_sheet

        with tempfile.TemporaryDirectory() as folder:
            photo = Image.new("RGB", (200, 400), (40, 30, 20))
            photo.save(Path(folder) / "ada.png")
            sheet = reference_sheet([Path(folder) / "ada.png"] * 2, 480, 832, Path(folder) / "sheet.png")
            pixels = np.asarray(Image.open(sheet))
            self.assertEqual(pixels.shape, (832, 480, 3))
            self.assertTrue((pixels[0] == 255).all(), "the sheet itself is white")
            self.assertEqual(tuple(pixels[416, 120]), (40, 30, 20), "the left photo is pasted unchanged")
            self.assertEqual(tuple(pixels[416, 360]), (40, 30, 20), "and so is the right one")


class WordsTests(unittest.TestCase):
    def test_performers_by_body_only_and_nobody_else(self) -> None:
        from describe import video_prompt

        show = two_people()
        episode = show["episodes"][0]
        seen = observation(
            {"ada": 5000, "bo": 9000},
            sky={"kind": "backdrop", "pixels": 20000},
            surround={"kind": "backdrop", "pixels": 0},
            bench={"kind": "landmark", "pixels": 3000},
        )
        prompt = video_prompt(show, episode, over_the_shoulder(), seen)
        self.assertTrue(prompt.startswith("Low-angle medium close-up with shallow depth of field, cinematic"))
        self.assertNotIn("shoulder", prompt, "the shoulder in front is left to the depth video")
        self.assertNotIn("old man", prompt, "nobody without a performance is named")
        self.assertIn("An adult woman, brown skin: speaks with a half-smile. Face: lips slightly parted.", prompt)
        self.assertNotIn("black hair", prompt, "no look beyond the body: the frame may not show it")
        self.assertNotIn("bench", prompt, "no landmarks or props: the frame may show only part of one")
        self.assertIn("Above: a gray sky.", prompt)
        self.assertNotIn("open fields", prompt, "only the backdrop in shot")
        self.assertIn("Soft overcast daylight.", prompt)
        self.assertNotIn("ada", prompt.lower().replace("daylight", ""), "no ids reach the model")
        self.assertFalse(PRONOUNS & set(prompt.lower().replace(",", " ").replace(".", " ").replace(":", " ").split()), "no pronouns")
        self.assertNotIn("Speaking", prompt, "a picture prompt says nothing of speech")
        for word in ("left", "right", "center", "facing", "profile"):
            self.assertNotIn(word, prompt, "no screen side or facing")

    def test_a_performer_outside_the_start_frame_comes_into_it(self) -> None:
        from describe import video_prompt

        show = two_people()
        entering = scene(performances={"bo": {"action": "walks in"}})
        prompt = video_prompt(show, show["episodes"][0], entering, observation({"ada": 5000}))
        self.assertIn("Coming into the frame, an old man, broad build: walks in.", prompt)

    def test_a_talking_take_is_about_the_first_speaker_with_a_performance(self) -> None:
        from describe import talking_speaker, video_prompt

        show = show_with({"ada": standing()})
        lines = [
            {"speakerId": "ada", "line": "Go.", "delivery": "low", "startSeconds": 0.2},
            {"speakerId": "ada", "line": "Now.", "delivery": "harder"},
        ]
        talking = scene(lines=lines, performances={"ada": {"action": "speaks", "expression": "jaw set"}})
        self.assertEqual(talking_speaker(talking), "ada")
        self.assertIsNone(talking_speaker(scene(lines=lines)), "a voice from off screen")
        prompt = video_prompt(show, show["episodes"][0], talking, observation({"ada": 5000}), "ada")
        self.assertIn("Speaking low; then harder, the head moving naturally with the words.", prompt)
        self.assertNotIn("Go.", prompt, "the words are the voice's, never the prompt's")

    def test_a_target_is_named_by_body(self) -> None:
        from describe import video_prompt

        show = two_people()
        grab = scene(performances={"ada": {"action": "seizes the wrist of {target}", "target": "bo"}})
        prompt = video_prompt(show, show["episodes"][0], grab, observation({"ada": 5000, "bo": 9000}))
        self.assertIn(": seizes the wrist of the old man, broad build.", prompt)

    def test_the_effects_in_shot_are_named(self) -> None:
        from describe import video_prompt

        show = show_with({"ada": standing()})
        start = observation({"ada": 5000})
        start["effects"] = [{"id": "bench", "appearance": "white-gold flames"}]
        self.assertIn("Moving all through the take: white-gold flames.", video_prompt(show, show["episodes"][0], scene(), start))

    def test_the_camera_in_words(self) -> None:
        from describe import camera_sentence

        self.assertIn("handheld and nearly still", camera_sentence(scene()))
        flying = scene(cameraMotion="flies low over the treetops")
        flying["camera"]["keyframes"].append({**flying["camera"]["keyframes"][0], "timeSeconds": 2, "position": [0, -2, 3]})
        self.assertEqual(camera_sentence(flying), "The camera flies low over the treetops, in one smooth continuous move.")

    def test_a_reference_photo_is_drawn_from_every_attribute(self) -> None:
        from describe import reference_prompt

        prompt = reference_prompt({"body": "adult woman, brown skin", "attributes": ["52 years old", "black hair"]})
        self.assertIn("one person, adult woman, brown skin. Exactly as described, item by item: 52 years old; black hair.", prompt)
        self.assertIn("pure white", prompt)


class VoiceTests(unittest.TestCase):
    def test_one_speakers_lines_in_a_row_are_one_speech(self) -> None:
        from voice import speeches

        lines = [
            {"speakerId": "ada", "line": "Strong?", "delivery": "dry", "startSeconds": 0.2},
            {"speakerId": "ada", "line": "Just  a bloodsucker.", "delivery": "flat", "startSeconds": 0.7},
            {"speakerId": "bo", "line": "Wait.", "delivery": "sharp", "startSeconds": 3.0},
        ]
        self.assertEqual(speeches({"lines": lines}), [
            {"speakerId": "ada", "text": "Strong? Just a bloodsucker.", "start": 0.2},
            {"speakerId": "bo", "text": "Wait.", "start": 3.0},
        ])

    def test_speech_is_laid_out_over_exactly_the_scene(self) -> None:
        from voice import lay_out

        rate = 100
        track, problems = lay_out([(0.5, np.full(50, 0.5, dtype=np.float32))], 2.0, rate)
        self.assertEqual(len(track), 200)
        self.assertEqual(float(track[49]), 0.0)
        self.assertEqual(float(track[50]), 0.5)
        self.assertEqual(problems, [])
        _track, problems = lay_out([(0.0, np.ones(150, dtype=np.float32)), (1.0, np.ones(150, dtype=np.float32))], 2.0, rate)
        self.assertEqual(len(problems), 2, "the first runs into the second, the second past the scene")
        silent, problems = lay_out([], 1.5, rate)
        self.assertEqual((len(silent), float(np.abs(silent).max()), problems), (150, 0.0, []))

    def test_a_voice_is_made_again_when_its_description_changes(self) -> None:
        from comfy import write_log
        from voice import design_entry, voice_is_current

        character = {"voice": "an old man's voice, low and slow"}
        with tempfile.TemporaryDirectory() as folder:
            dest = Path(folder) / "voice.wav"
            self.assertFalse(voice_is_current(character, dest), "not made yet")
            dest.write_bytes(b"\0" * 4096)
            write_log(dest, {**design_entry(character), "seed": 1})
            self.assertTrue(voice_is_current(character, dest))
            self.assertFalse(voice_is_current({"voice": "a young woman's voice"}, dest))


class GraphTests(unittest.TestCase):
    def test_a_still_is_the_first_frame_of_a_short_take(self) -> None:
        graph = wan.vace_graph("words", "bad", "480", wan.STILL_FRAMES, 5, "depth.mp4", None, still=True)
        vace = graph["11"]["inputs"]
        self.assertEqual((vace["width"], vace["height"], vace["length"], vace["strength"]), (480, 832, 9, wan.VACE_STRENGTH))
        self.assertNotIn("reference_image", vace, "nobody to match: no reference")
        self.assertEqual(graph["17"]["inputs"]["length"], 1)
        self.assertEqual(graph["18"]["inputs"]["filename_prefix"], wan.STILL_PREFIX)
        self.assertEqual(graph["12"]["inputs"]["end_at_step"], graph["13"]["inputs"]["start_at_step"], "the experts split the steps")
        take = wan.vace_graph("words", "bad", "720", 89, 5, "depth.mp4", "sheet.png", still=False)
        self.assertEqual(take["11"]["inputs"]["reference_image"], ["16", 0])
        self.assertEqual(take["18"]["inputs"]["filename_prefix"], wan.VACE_PREFIX)
        self.assertEqual(take["11"]["inputs"]["width"], 720)

    def test_a_talking_take_carries_its_voice_and_drops_the_bad_first_frames(self) -> None:
        graph = wan.s2v_graph("words", "bad", "480", 89, 5, "still.png", "voice.wav")
        self.assertEqual(graph["16"]["inputs"]["batch_index"], wan.S2V_DROP_FRAMES)
        self.assertEqual(graph["17"]["inputs"]["audio"], ["8", 0])
        self.assertEqual(graph["17"]["inputs"]["frame_rate"], wan.FPS)
        self.assertEqual(graph["12"]["inputs"]["steps"], wan.S2V_STEPS)

    def test_rife_triples_to_48_with_the_sound(self) -> None:
        graph = wan.rife_graph("take.mp4")
        self.assertEqual(graph["3"]["inputs"]["multiplier"] * wan.FPS, 2 * wan.OUTPUT_FPS)
        self.assertEqual(graph["4"]["inputs"]["frame_rate"], 48)
        self.assertEqual(graph["4"]["inputs"]["audio"], ["2", 1])


class RemakeTests(unittest.TestCase):
    def test_an_output_is_current_only_while_its_inputs_are_the_same(self) -> None:
        from comfy import write_log
        from generate_batch import made_from

        with tempfile.TemporaryDirectory() as folder:
            dest = Path(folder) / "scene_04_start.png"
            self.assertFalse(made_from(dest, {"prompt": "a"}))
            dest.write_bytes(b"x" * 2048)
            write_log(dest, {"prompt": "a", "size": "480", "seed": 3})
            self.assertTrue(made_from(dest, {"prompt": "a", "size": "480"}))
            self.assertFalse(made_from(dest, {"prompt": "a", "size": "720"}), "another size is another still")
            self.assertFalse(made_from(dest, {"prompt": "b", "size": "480"}), "new words are a new still")
            self.assertEqual(json.loads((Path(folder) / "inputs" / "scene_04_start" / "log.json").read_text())["seed"], 3)


if __name__ == "__main__":
    unittest.main()
