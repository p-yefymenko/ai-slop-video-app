"""The Wan chain's logic: frame counts, the depth video, the reference sheet, the words, the voice track, the graphs."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
import unittest.mock
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

    def test_a_picture_is_held_to_its_control_image(self) -> None:
        import asset_generate

        graph = asset_generate.picture_graph("words", "bad", (960, 1664), 5, "depth.png")
        self.assertEqual(graph["11"]["inputs"]["image"], "depth.png")
        self.assertEqual(graph["8"]["inputs"]["positive"], ["13", 0], "held by the ControlNet")
        self.assertEqual((graph["7"]["inputs"]["width"], graph["7"]["inputs"]["height"]), (960, 1664))

    def test_the_collage_puts_each_stage_on_a_row(self) -> None:
        from collage import LABEL_HEIGHT, write_collage

        drawing = Image.new("RGB", (200, 400), (0, 0, 0))
        with tempfile.TemporaryDirectory() as folder:
            dest = write_collage([[("a", drawing), ("b", drawing)], [("c", Image.new("RGB", (100, 100)))]], Path(folder) / "collage.png", 50)
            with Image.open(dest) as sheet:
                self.assertEqual(sheet.size, (100, 100 + 50 + 2 * LABEL_HEIGHT))

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
        self.assertTrue(prompt.startswith("Cinematic live-action footage, photorealistic."), "the framing is the still's and the depth video's, not words")
        self.assertNotIn("close-up", prompt)
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

    def test_an_over_the_shoulder_shot_says_so_and_names_nobody_in_front(self) -> None:
        from describe import video_prompt

        show = two_people()
        seen = observation({"ada": 5000, "bo": 9000})
        seen["entities"]["bo"]["depth"] = 1.2
        prompt = video_prompt(show, show["episodes"][0], over_the_shoulder(), seen)
        self.assertNotIn("old man", prompt, "named, the person in front was given the subject's action")
        single = scene(shot={"type": "single", "subjects": ["ada"], "size": "mcu"}, performances={"ada": {"action": "speaks"}})
        self.assertNotIn("shoulder", video_prompt(show, show["episodes"][0], single, seen), "only a shot authored as over the shoulder")

    def test_how_far_round_from_a_persons_front_the_camera_stands(self) -> None:
        from shot_inputs import view_degrees

        class Person:
            offset, yaw_degrees = (0.0, 0.0, 0.0), 0.0  # facing +Y

        self.assertAlmostEqual(view_degrees(Person, {"position": [0.0, 3.0, 1.5]}), 0.0)
        self.assertAlmostEqual(view_degrees(Person, {"position": [0.0, -3.0, 1.5]}), 180.0)
        self.assertAlmostEqual(view_degrees(Person, {"position": [-3.0, 0.0, 1.5]}), 90.0)

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

    def test_speech_is_laid_out_over_the_scene_or_as_long_as_it_runs(self) -> None:
        from voice import SPEECH_TAIL_SECONDS, lay_out

        rate = 100
        track, notes = lay_out([(0.5, np.full(50, 0.5, dtype=np.float32))], 2.0, rate)
        self.assertEqual(len(track), 200, "speech that fits leaves the scene its scripted length")
        self.assertEqual(float(track[49]), 0.0)
        self.assertEqual(float(track[50]), 0.5)
        self.assertEqual(notes, [])
        track, notes = lay_out([(0.0, np.ones(150, dtype=np.float32)), (1.0, np.full(150, 0.5, dtype=np.float32))], 2.0, rate)
        self.assertEqual(float(track[149]), 1.0)
        self.assertEqual(float(track[150]), 0.5, "the second speech waits for the first to end, never over it")
        self.assertEqual(len(track), round((3.0 + SPEECH_TAIL_SECONDS) * rate), "the scene runs to a beat after the last word")
        self.assertEqual(len(notes), 2)
        silent, notes = lay_out([], 1.5, rate)
        self.assertEqual((len(silent), float(np.abs(silent).max()), notes), (150, 0.0, []))

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


class SheetTests(unittest.TestCase):
    def test_the_layout_shows_each_mesh_in_its_own_colors_shaded_as_the_clay(self) -> None:
        from shot_inputs import CLAY_LIGHTEST, colored_layout

        lit = int(CLAY_LIGHTEST)
        clay = np.array([[[lit, lit, lit], [lit // 2, lit // 2, lit // 2], [150, 150, 150]]], dtype=np.uint8)
        albedo = np.array([[[200, 100, 50], [200, 100, 50], [9, 9, 9]]], dtype=np.uint8)
        ids = np.array([[1, 1, 2]], dtype=np.uint8)
        out = colored_layout(clay, albedo, ids, [1])
        self.assertEqual(tuple(out[0, 0]), (200, 100, 50), "a surface turned to the light keeps its own colors")
        self.assertEqual(tuple(out[0, 1]), (100, 50, 25), "the clay's shading kept")
        self.assertEqual(tuple(out[0, 2]), (150, 150, 150), "a mesh without colors stays clay")

    def test_the_layout_words_give_no_looks_and_faces_only_toward_the_camera(self) -> None:
        from describe import layout_prompt

        show = two_people()
        show["characters"]["ada"]["attributes"] = [{"face": "brown eyes"}, "black hair"]
        seen = observation({"ada": 5000, "bo": 9000}, sky={"kind": "backdrop", "pixels": 20000})
        shot = over_the_shoulder()
        words = layout_prompt(show, shot, seen, {"ada": 10.0, "bo": 20.0}, (720, 1280))
        self.assertIn("720 by 1280 pixels", words)
        self.assertIn("The face of the adult woman, brown skin: lips slightly parted.", words)
        self.assertNotIn("old man", words, "a person without a performance is not named")
        self.assertNotIn("black hair", words, "looks come from the layout's colors, never words")
        self.assertNotIn("fox-fur", words)
        self.assertNotIn("brown eyes", words)
        self.assertNotIn("close-up", words, "the layout shows the framing; no words for it")
        self.assertNotIn("left", words)
        self.assertNotIn("right", words)
        away = layout_prompt(show, shot, seen, {"ada": 170.0, "bo": 20.0}, (720, 1280))
        self.assertNotIn("The face of", away, "no face words with the back toward the camera")

    def test_a_person_partly_in_the_frame_gets_only_the_authored_words_for_what_it_shows(self) -> None:
        from describe import layout_prompt

        show = two_people()
        shot = over_the_shoulder()
        shot["shot"]["partlySeen"] = {"bo": ["a red fox-fur mantle"], "ada": ["black hair"]}
        seen = observation({"ada": 5000, "bo": 9000})
        words = layout_prompt(show, shot, seen, {"bo": 20.0}, (720, 1280))
        self.assertIn("Of a person only partly in the frame, it shows: a red fox-fur mantle.", words)
        self.assertNotIn("old man", words, "tied to no one: a body's words would get the rest drawn")
        self.assertNotIn("black hair", words, "nobody out of the start frame")

    def test_a_words_only_still_on_a_moving_camera_says_no_size_or_angle(self) -> None:
        from describe import plain_still_prompt

        seen = observation({}, sky={"kind": "backdrop", "pixels": 20000})
        held = over_the_shoulder()
        self.assertIn("Low-angle medium close-up", plain_still_prompt(two_people(), held, seen, (720, 1280)))
        moving = over_the_shoulder()
        moving["camera"]["keyframes"].append({"timeSeconds": 2, "position": [0, -4, 1.5], "lookAt": [0, 0, 1.6], "verticalFovDegrees": 40})
        words = plain_still_prompt(two_people(), moving, seen, (720, 1280))
        self.assertNotIn("close-up", words, "the size is the whole shot's, not its first frame's")
        self.assertNotIn("low-angle", words.lower())
        self.assertIn("pixels. Cinematic live-action footage, photorealistic.", words)
        moving["shot"] = {"type": "action"}
        self.assertNotIn("wide", plain_still_prompt(two_people(), moving, seen, (720, 1280)).lower(), "a shot type's words are a size too")

    def test_a_still_is_found_in_whatever_format_it_was_saved(self) -> None:
        from generate_batch import saved_still

        with tempfile.TemporaryDirectory() as folder:
            still = Path(folder) / "scene_01_start.png"
            self.assertEqual(saved_still(still), still)
            still.with_suffix(".jpg").write_bytes(b"x")
            self.assertEqual(saved_still(still), still.with_suffix(".jpg"))
            still.write_bytes(b"x")
            with self.assertRaises(SystemExit):
                saved_still(still)

    def test_a_still_with_none_of_the_3d_scene_in_frame_is_words_alone(self) -> None:
        from describe import plain_still_prompt

        seen = observation({}, sky={"kind": "backdrop", "pixels": 20000})
        words = plain_still_prompt(two_people(), scene(shot={"type": "establishing"}), seen, (720, 1280))
        self.assertTrue(words.startswith("One photorealistic live-action film frame, 720 by 1280 pixels. Wide establishing shot"), "a held camera's still says its shot")
        self.assertNotIn("tinted", words)


class GraphTests(unittest.TestCase):
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

    def test_rife_fills_a_25_fps_take_to_at_least_twice_the_cut(self) -> None:
        graph = wan.rife_graph("take.mp4", wan.TALK_FPS)
        self.assertEqual(graph["4"]["inputs"]["frame_rate"], 50)

    def test_a_take_with_one_person_talks_to_one_voice(self) -> None:
        graph = wan.talk_graph("words", "bad", "480", 69, 5, "still.png", "voice.wav", None)
        node = graph["w0_talk"]["inputs"]
        self.assertEqual(node["mode"], "single_speaker")
        self.assertNotIn("mode.mask_1", node)
        self.assertEqual(graph["15"]["inputs"]["audio"], ["9", 0], "the take carries the voice, never the silence")
        self.assertEqual(graph["15"]["inputs"]["frame_rate"], wan.TALK_FPS)

    def test_with_anyone_else_in_frame_the_voice_is_tied_to_the_speaker_and_silence_to_the_rest(self) -> None:
        graph = wan.talk_graph("words", "bad", "480", 69, 5, "still.png", "voice.wav", ("speaker.png", "others.png", "silence.wav"))
        node = graph["w0_talk"]["inputs"]
        self.assertEqual(node["mode"], "two_speakers")
        self.assertEqual(node["audio_encoder_output_1"], ["10", 0])
        self.assertEqual(graph["10"]["inputs"]["audio"], ["9", 0])
        self.assertEqual(graph["9"]["inputs"]["audio"], "voice.wav")
        self.assertEqual(graph[node["mode.mask_1"][0]]["inputs"]["image"], ["16", 0])
        self.assertEqual(graph["16"]["inputs"]["image"], "speaker.png")
        self.assertEqual(graph["18"]["inputs"]["image"], "others.png")
        self.assertEqual(graph[graph[node["mode.audio_encoder_output_2"][0]]["inputs"]["audio"][0]]["inputs"]["audio"], "silence.wav")
        self.assertEqual(graph["15"]["inputs"]["audio"], ["9", 0])

    def test_a_long_take_is_made_in_windows_each_going_on_from_the_last_frames(self) -> None:
        self.assertEqual(wan.talk_windows(69), [69])
        self.assertEqual(wan.talk_windows(81), [81])
        self.assertEqual(wan.talk_windows(113), [81, 41])
        self.assertEqual(wan.talk_windows(153), [81, 81])
        for length in range(82, 400, 4):
            windows = wan.talk_windows(length)
            made = windows[0] + sum(frames - wan.TALK_MOTION_FRAMES for frames in windows[1:])
            self.assertGreaterEqual(made, length)
            self.assertLess(made - length, 4, "the last window no longer than needed")
            self.assertTrue(all(frames % 4 == 1 and frames <= wan.TALK_WINDOW for frames in windows))
        graph = wan.talk_graph("words", "bad", "480", 113, 5, "still.png", "voice.wav", ("s.png", "o.png", "silence.wav"))
        second = graph["w1_talk"]["inputs"]
        self.assertEqual(second["previous_frames"], ["w0_decode", 0])
        self.assertEqual(second["mode.mask_1"], graph["w0_talk"]["inputs"]["mode.mask_1"])
        self.assertEqual(graph["w1_new"]["inputs"]["batch_index"], ["w1_talk", 4], "the repeated frames are cut off")
        self.assertEqual(graph["15"]["inputs"]["images"], ["w1_take", 0])

    def test_each_window_is_guided_toward_the_words_and_the_voice_each_at_its_own_scale(self) -> None:
        graph = wan.talk_graph("words", "bad", "480", 113, 5, "still.png", "voice.wav", None)
        for window in ("w0", "w1"):
            guide = graph[f"{window}_guide"]
            self.assertEqual(guide["class_type"], "InfiniteTalkGuider")
            self.assertEqual((guide["inputs"]["text_cfg"], guide["inputs"]["audio_cfg"]), (wan.TALK_TEXT_CFG, wan.TALK_AUDIO_CFG))
            self.assertEqual(guide["inputs"]["model"], [f"{window}_talk", 0])
            self.assertEqual(graph[f"{window}_sample"]["inputs"]["guider"], [f"{window}_guide", 0])
            self.assertEqual(graph[f"{window}_steps"]["inputs"]["steps"], wan.TALK_STEPS)

    def test_frames_cover_the_scene_at_the_takes_rate(self) -> None:
        self.assertEqual(wan.frames_for(2.63, wan.TALK_FPS), 69)


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
