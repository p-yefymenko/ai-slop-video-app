"""The world, the visibility rule, the words, and the checks."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import world  # noqa: E402
from body_parts import BODY_PARTS  # noqa: E402
from checks import check_scene  # noqa: E402
from describe import drawn_prompt, effect_prompt, ltx_prompt, people_places, still_prompt  # noqa: E402
from mesh_io import write_schema_glb  # noqa: E402
from observe import in_shot, passes, visible  # noqa: E402


def column(radius: float = 0.2, height: float = 1.7, rings: int = 80, around: int = 24):
    """A closed upright cylinder with enough vertices to carry every body part."""
    points = [
        [radius * np.sin(2 * np.pi * i / around), radius * np.cos(2 * np.pi * i / around), z]
        for z in np.linspace(0.0, height, rings)
        for i in range(around)
    ]
    points += [[0.0, 0.0, 0.0], [0.0, 0.0, height]]
    faces = []
    for ring in range(rings - 1):
        for i in range(around):
            a, b = ring * around + i, ring * around + (i + 1) % around
            faces += [[a, b, b + around], [a, b + around, a + around]]
    bottom, top = len(points) - 2, len(points) - 1
    for i in range(around):
        faces += [[bottom, (i + 1) % around, i], [top, (rings - 1) * around + i, (rings - 1) * around + (i + 1) % around]]
    return np.asarray(points, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def show_with(tracks: dict, props: dict | None = None, prop_tracks: dict | None = None) -> dict:
    return {
        "id": "demo",
        "characters": {
            "ada": {
                "body": "adult woman, brown skin",
                "attributes": ["52 years old", "black hair"],
                "heightMeters": 1.7,
            }
        },
        "props": props or {},
        "locations": {
            "room": {
                "look": {"materials": "worn gray stone", "light": "soft overcast daylight"},
                "backdrop": {
                    "sky": "a gray sky",
                    "skyColor": [40, 48, 64],
                    "ground": "stone paving, no walls",
                    "groundColor": [48, 44, 40],
                    "surround": "open fields",
                    "surroundColor": [24, 56, 40],
                },
                "soundscape": {"ambience": "wind", "space": "open air"},
                "spatial": {
                    "sizeMeters": [8, 10, 4],
                    "landmarks": {
                        "bench": {"position": [0, 2, 0], "size": [1, 1, 1], "appearance": "One stone bench, a single object"}
                    },
                },
            },
            "hall": {"spatial": {"sizeMeters": [8, 8, 4], "landmarks": {}}},
        },
        "episodes": [
            {
                "spatialTimeline": {"durationSeconds": 10, "characterTracks": tracks, "propTracks": prop_tracks or {}},
                "scenes": [],
            }
        ],
    }


def scene(position=(0, -4, 1.5), performances=None, **extra) -> dict:
    return {
        "sceneNumber": 1,
        "locationId": "room",
        "timeRangeSeconds": [0, 2],
        "camera": {"keyframes": [{"timeSeconds": 0, "position": list(position), "lookAt": [0, 0, 1.2], "verticalFovDegrees": 40}]},
        "performances": performances or {},
        "sound": {"events": [], "bed": "present", "music": {"kind": "none"}},
        **extra,
    }


def observation(regions: dict[str, dict[str, list[int]]], **entries) -> dict:
    found = {}
    for entity_id, parts in regions.items():
        shown = sum(v for v, _e in parts.values())
        found[entity_id] = {"kind": "character", "pixels": [shown, shown], "screenX": 380.0, "depth": 4.0, "regions": parts}
    found.update(entries)
    return {"timeSeconds": 0.0, "camera": {"position": [0, -4, 1.5]}, "entities": found}


FRONT = {part: [5000, 6000] for part in BODY_PARTS}
BEHIND = {**FRONT, "face": [1800, 42000], "eyes": [900, 30000]}


class TimelineTests(unittest.TestCase):
    def test_only_movement_the_camera_sees_changes_the_shot(self) -> None:
        drifting = [
            {"timeSeconds": 0, "locationId": "room", "position": [3, 3, 0], "bodyYawDegrees": 0},
            {"timeSeconds": 2, "locationId": "room", "position": [3.5, 3, 0], "bodyYawDegrees": 0},
        ]
        show = show_with({"ada": drifting})
        episode = show["episodes"][0]
        self.assertFalse(world.scene_has_spatial_change(episode, scene(), set()), "out of shot")
        self.assertTrue(world.scene_has_spatial_change(episode, scene(), {"ada"}), "in shot")

    def test_depth_control_only_where_the_clay_shows_the_shot(self) -> None:
        standing = [{"timeSeconds": 0, "locationId": "room", "position": [0, 0, 0], "bodyYawDegrees": 0}]
        walking = [*standing, {"timeSeconds": 2, "locationId": "room", "position": [0.5, 0, 0], "bodyYawDegrees": 0}]
        dolly = {"keyframes": [
            {"timeSeconds": 0, "position": [0, -4, 1.5], "lookAt": [0, 0, 1.2], "verticalFovDegrees": 40},
            {"timeSeconds": 2, "position": [0, -3, 1.5], "lookAt": [0, 0, 1.2], "verticalFovDegrees": 40},
        ]}
        still_episode = show_with({"ada": standing})["episodes"][0]
        fits = lambda episode, **extra: world.depth_control_fits(episode, scene(**extra), {"ada"})
        glance = {"ada": {"action": "glances aside", "parts": ["eyes", "face"]}}
        wave = {"ada": {"action": "waves", "parts": ["arms", "hands"]}}
        self.assertTrue(fits(still_episode, camera=dolly, performances=glance), "moving camera, still outline")
        self.assertFalse(fits(still_episode, performances=glance), "locked camera: nothing for the clay to hold")
        self.assertFalse(fits(still_episode, camera=dolly, performances=wave), "a clay arm cannot wave")
        self.assertFalse(fits(show_with({"ada": walking})["episodes"][0], camera=dolly), "a clay body slides")
        self.assertFalse(fits(still_episode, camera=dolly, performances=glance, speakerId="ada"), "a clay face cannot talk")

    def test_walk_turn_and_location_change(self) -> None:
        track = [
            {"timeSeconds": 0, "locationId": "room", "position": [0, 0, 0], "bodyYawDegrees": 350},
            {"timeSeconds": 2, "locationId": "room", "position": [2, 0, 0], "bodyYawDegrees": 10},
            {"timeSeconds": 4, "locationId": "hall", "position": [1, 1, 0], "bodyYawDegrees": 0},
            {"timeSeconds": 6, "locationId": None},
        ]
        middle = world.character_state(track, 1.0)
        self.assertEqual(middle["position"], [1.0, 0.0, 0.0])
        self.assertAlmostEqual(middle["bodyYawDegrees"] % 360, 0.0)
        self.assertEqual(world.character_state(track, 3.9)["locationId"], "room")
        self.assertEqual(world.character_state(track, 4.0)["locationId"], "hall")
        self.assertIsNone(world.character_state(track, 9.0)["locationId"])


class EntityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        vertices, faces = column()
        for path in ("characters/ada", "room/bench", "props/cup"):
            write_schema_glb(root / "demo" / path / "model.glb", vertices, faces)
        self.patch = patch.object(world, "stage_dir", lambda _stage, show_id: root / show_id)
        self.patch.start()
        world._MESHES.clear()

    def tearDown(self) -> None:
        self.patch.stop()
        self.temp.cleanup()

    def test_an_effect_is_a_box_on_its_host_and_never_in_the_world(self) -> None:
        here = show_with({"ada": [{"timeSeconds": 0, "locationId": "room", "position": [0, 0, 0], "bodyYawDegrees": 0}]})
        flame = {"appearance": "white-gold flames", "size": [1, 1, 2], "offset": [0, 0, 0.5]}
        here["locations"]["room"]["spatial"]["landmarks"]["bench"]["effects"] = [flame]
        entities = world.entities_at(here, here["episodes"][0], scene(), 0)
        self.assertEqual([e.id for e in entities], ["bench", "ada", "floor"], "an effect is never rendered or measured")
        box = world.effect_entity(entities[0], 0)
        low, high = box.bounds()
        np.testing.assert_allclose(low, [-0.5, 1.5, 0.5])
        np.testing.assert_allclose(high, [0.5, 2.5, 2.5])

    def test_presence_comes_from_the_timeline(self) -> None:
        here = show_with({"ada": [{"timeSeconds": 0, "locationId": "room", "position": [0, 0, 0], "bodyYawDegrees": 0}]})
        ids = [e.id for e in world.entities_at(here, here["episodes"][0], scene(), 0)]
        self.assertEqual(ids, ["bench", "ada", "floor"])
        away = show_with({"ada": [{"timeSeconds": 0, "locationId": "hall", "position": [0, 0, 0], "bodyYawDegrees": 0}]})
        self.assertNotIn("ada", [e.id for e in world.entities_at(away, away["episodes"][0], scene(), 0)])
        gone = show_with({"ada": [{"timeSeconds": 0, "locationId": None}]})
        self.assertNotIn("ada", [e.id for e in world.entities_at(gone, gone["episodes"][0], scene(), 0)])
        outside = world.entities_at(here, here["episodes"][0], scene(position=(0, -20, 1.5)), 0)
        self.assertNotIn("floor", [e.id for e in outside])

    def test_look_at_turns_the_body_and_a_held_prop_follows_the_hand(self) -> None:
        tracks = {"ada": [{"timeSeconds": 0, "locationId": "room", "position": [0, 0, 0], "bodyYawDegrees": 0, "lookAtId": "bench"}]}
        cup = {"cup": {"appearance": "One clay cup", "size": [0.1, 0.1, 0.1]}}
        held = {"cup": [{"timeSeconds": 0, "heldByCharacterId": "ada", "heldInHand": "left"}]}
        show = show_with(tracks, cup, held)
        entities = {e.id: e for e in world.entities_at(show, show["episodes"][0], scene(), 0)}
        self.assertAlmostEqual(entities["ada"].yaw_degrees, 0.0)
        self.assertIn("cup", entities)
        show["episodes"][0]["spatialTimeline"]["characterTracks"]["ada"][0]["lookAtId"] = None
        show["episodes"][0]["spatialTimeline"]["characterTracks"]["ada"][0]["bodyYawDegrees"] = 90
        entities = {e.id: e for e in world.entities_at(show, show["episodes"][0], scene(), 0)}
        self.assertAlmostEqual(entities["ada"].yaw_degrees, 90.0)
        low, high = entities["ada"].bounds()
        np.testing.assert_allclose((high - low)[2], 1.7, atol=1e-5)
        hand = entities["cup"].offset
        self.assertAlmostEqual(hand[2], 0.45 * 1.7, delta=0.06)

    def test_missing_mesh_is_an_error(self) -> None:
        show = show_with({"ada": [{"timeSeconds": 0, "locationId": "room", "position": [0, 0, 0], "bodyYawDegrees": 0}]})
        show["locations"]["room"]["spatial"]["landmarks"]["well"] = {"position": [1, 1, 0], "size": [1, 1, 1], "appearance": "x"}
        with self.assertRaises(world.MissingMesh):
            world.entities_at(show, show["episodes"][0], scene(), 0)


class VisibilityTests(unittest.TestCase):
    def test_checks_share_one_rule_with_their_own_minimum(self) -> None:
        self.assertTrue(passes(5000, 6000, 250))
        self.assertFalse(passes(1800, 42000, 250), "a face seen from behind is a sliver of its extent")
        self.assertFalse(passes(100, 100, 250), "too small for this purpose")
        seen = observation({"ada": BEHIND})
        self.assertTrue(visible(seen, "ada", None, 250))
        self.assertTrue(visible(seen, "ada", ["hair"], 250))
        self.assertFalse(visible(seen, "ada", ["face"], 250))
        self.assertFalse(visible(seen, "nobody", None, 250))

    def test_in_shot_has_no_size_cutoff(self) -> None:
        seen = observation({"ada": {"hair": [1, 9000]}})
        self.assertTrue(in_shot(seen, "ada"))
        self.assertFalse(in_shot(observation({"ada": {"hair": [0, 9000]}}), "ada"))
        self.assertFalse(in_shot(seen, "nobody"))


class DescribeTests(unittest.TestCase):
    def test_a_person_is_drawn_whole_from_their_own_words(self) -> None:
        show = show_with({})
        shot = scene(performances={"ada": {"action": "waits", "parts": ["torso"], "expression": "jaw set"}})
        prompt = drawn_prompt(show, shot, "character", "ada", ["Depth", "OwnColor", "Edges", "Appearance"])
        self.assertIn("Adult woman, brown skin, 52 years old, black hair, jaw set.", prompt)
        self.assertIn("Picture 4 is how it looks", prompt)
        self.assertIn("One person, alone", prompt)
        self.assertNotIn("ada", prompt)

    def test_a_landmark_is_drawn_whole_from_its_own_words(self) -> None:
        show = show_with({})
        prompt = drawn_prompt(show, scene(), "landmark", "bench", ["Depth", "OwnColor", "Edges", "Appearance"])
        self.assertIn("One object, alone", prompt)
        self.assertIn("The light: soft overcast daylight, the same on everything in the shot.", prompt)
        self.assertIn("One stone bench.", prompt)

    def test_the_shot_prompt_names_no_object_only_the_setting(self) -> None:
        show = show_with({})
        seen = observation(
            {"ada": FRONT},
            bench={"kind": "landmark", "pixels": [9000, 10000], "regions": {"whole": [9000, 10000]}},
            sky={"kind": "backdrop", "pixels": [20000, 20000], "regions": {"whole": [20000, 20000]}},
            ground={"kind": "backdrop", "pixels": [40, 40], "regions": {"whole": [40, 40]}},
            surround={"kind": "backdrop", "pixels": [0, 0], "regions": {"whole": [0, 0]}},
        )
        prompt, _log = still_prompt(show, scene(), seen, ["Depth", "Composite", "Edges"], 2)
        self.assertIn("already painted are finished and stay exactly as they are", prompt)
        self.assertIn("Everything standing in the shot is already painted", prompt)
        self.assertNotIn("bench", prompt)
        self.assertNotIn("52 years old", prompt)
        self.assertIn("A gray sky.", prompt)
        self.assertIn("The light: soft overcast daylight", prompt)
        self.assertIn("stone paving", prompt.lower(), "any pixel of the ground in the shot is said")
        self.assertNotIn("open fields", prompt)

    def test_ltx_names_people_by_where_the_start_frame_shows_them(self) -> None:
        show = show_with({})
        shot = scene(performances={"ada": {"action": "waits", "parts": ["torso"]}})
        start = observation({"ada": FRONT})
        self.assertEqual(people_places(start), {"ada": "In the center"})
        prompt = ltx_prompt(show, shot, start)
        self.assertIn("In the center, adult woman, brown skin: waits.", prompt)
        self.assertIn("No music plays.", prompt)
        start["effects"] = [{"id": "bench", "appearance": "white-gold flames", "guide": "effect_bench_0"}]
        self.assertIn("Moving all through the take: white-gold flames.", ltx_prompt(show, shot, start))
        empty = ltx_prompt(show, scene(), observation({}))
        self.assertIn("Preserve its set, composition", empty)
        self.assertNotIn("people", empty, "an empty shot never mentions people")

    def test_an_action_points_only_at_what_the_start_frame_shows(self) -> None:
        show = show_with({})
        reach = {"ada": {"action": "reaches toward {target}", "parts": ["arms"], "target": "bench"}}
        start = observation({"ada": FRONT})
        self.assertIn("reaches toward the stone bench.", ltx_prompt(show, scene(performances=reach), start))
        start["effects"] = [{"id": "bench", "appearance": "white-gold flames"}]
        self.assertIn("reaches toward the white-gold flames.", ltx_prompt(show, scene(performances=reach), start))

    def test_an_effect_is_drawn_with_its_host(self) -> None:
        show = show_with({})
        bench = show["locations"]["room"]["spatial"]["landmarks"]["bench"]
        bench["effects"] = [{"appearance": "white-gold flames", "size": [1, 1, 1], "offset": [0, 0, 1]}]
        solid = drawn_prompt(show, scene(), "landmark", "bench", ["Depth", "OwnColor", "Edges", "Appearance"])
        self.assertIn("One stone bench.", solid)
        self.assertNotIn("flames", solid, "the solid drawing has no effects")
        prompt = effect_prompt(show, scene(), "landmark", "bench", ["Base", "Look"])
        self.assertIn("Picture 2 is how it looks with its effects.", prompt)
        self.assertIn("Picture 1 is the object on pure black.", prompt)
        self.assertIn("Add only its white-gold flames", prompt)
        self.assertIn("pure black", prompt)
        plate = world.plate_path(show, "landmark", "bench", "room")
        self.assertEqual(world.look_path(show, "landmark", "bench", "room"), plate.with_name("look.png"))
        del bench["effects"]
        self.assertEqual(world.look_path(show, "landmark", "bench", "room"), plate)


class CheckTests(unittest.TestCase):
    def test_the_script_must_match_what_is_visible(self) -> None:
        show = show_with({})
        start = observation({"ada": BEHIND})
        unlisted = check_scene(show, show["episodes"][0], scene(), [(0.0, start, [])])
        self.assertFalse(any("ada" in e for e in unlisted), "someone caught by the camera without a performance holds still")
        held = ltx_prompt(show, scene(), start)
        self.assertIn("In the center, adult woman, brown skin: stands still", held)
        shot = scene(
            speakerId="ada",
            performances={"ada": {"action": "talks to {target}", "parts": ["face"], "expression": "jaw set", "target": "bench"}},
            sound={"events": [{"text": "a breath", "source": {"landmarkId": "bench"}}], "bed": "faint", "music": {"kind": "none"}},
        )
        errors = "\n".join(check_scene(show, show["episodes"][0], shot, [(0.0, start, [])]))
        self.assertIn("parts names face, but their face is never visible in the shot (start: 1800 px visible of 42000 px)", errors)
        self.assertIn("expression is set, but their face is not visible", errors)
        self.assertIn("speaker ada's face is 1800 px visible of 42000 px in the start frame; a speaking face needs at least", errors)
        self.assertIn("comes from bench, which is not visible", errors)
        self.assertIn("performances.ada.target is bench, which is not visible in the start frame", errors)
        absent = check_scene(show, show["episodes"][0], shot, [(0.0, observation({}), [])])
        self.assertTrue(any("performances.ada is set, but ada is never visible" in e for e in absent))


class RenderTests(unittest.TestCase):
    """The rule on real pixels: a box person seen from the back shows no face."""

    def test_face_from_front_and_behind(self) -> None:
        try:
            import moderngl  # noqa: F401
            from render import coverage, render
        except Exception as exc:  # pragma: no cover - machine without a GPU context
            self.skipTest(f"no GPU context: {exc}")
        vertices, faces = column()
        mesh = world._region_mesh("box", vertices, faces, None, __import__("body_parts").part_ids_for_vertices(vertices))
        person = world.Entity("ada", "character", mesh, (0.0, 0.0, 0.0), 0.0, BODY_PARTS)
        face = BODY_PARTS.index("face")
        for position, expected in (((0, 3, 1.6), True), ((0, -3, 1.6), False)):
            camera = {"position": list(position), "lookAt": [0, 0, 1.5], "verticalFovDegrees": 30}
            frame = render([person], camera)
            shown = int(np.count_nonzero((frame.entity == 1) & (frame.region == face + 1)))
            extent = coverage(person, camera)[0][face]
            self.assertEqual(passes(shown, extent, 250), expected, (position, shown, extent))

    def test_fire_in_front_hides_a_face_from_the_checks_but_not_from_the_shot(self) -> None:
        try:
            import moderngl  # noqa: F401
            from render import render
        except Exception as exc:  # pragma: no cover - machine without a GPU context
            self.skipTest(f"no GPU context: {exc}")
        from observe import _seen_through_effects

        vertices, faces = column()
        mesh = world._region_mesh("box", vertices, faces, None, __import__("body_parts").part_ids_for_vertices(vertices))
        person = world.Entity("ada", "character", mesh, (0.0, 0.0, 0.0), 0.0, BODY_PARTS)
        flame = {"appearance": "flames", "size": [1.0, 0.2, 2.0], "offset": [0, 0, 0]}
        pebble = world._region_mesh("pebble", vertices * 0.01, faces, None, np.zeros(len(vertices), dtype=np.int16))
        ring = world.Entity("ring", "landmark", pebble, (0.0, 1.0, 0.0), 0.0, ("whole",), (flame,))
        camera = {"position": [0, 3, 1.6], "lookAt": [0, 0, 1.5], "verticalFovDegrees": 30}
        frame = render([person, ring], camera)
        face = BODY_PARTS.index("face") + 1
        self.assertGreater(int(np.count_nonzero((frame.entity == 1) & (frame.region == face))), 0, "in the shot")
        seen = _seen_through_effects([person, ring], camera, frame)
        self.assertEqual(int(np.count_nonzero((seen.entity == 1) & (seen.region == face))), 0, "hidden behind the fire")




class UpscaleTests(unittest.TestCase):
    def test_the_upscale_setting_reaches_the_clip_stage(self) -> None:
        from depth_control import load_renderer_config

        self.assertEqual(load_renderer_config({"ltxSpatialUpscale": "x2"})["ltxSpatialUpscale"], "x2")
        self.assertIsNone(load_renderer_config({})["ltxSpatialUpscale"])
        with self.assertRaises(SystemExit):
            load_renderer_config({"ltxSpatialUpscale": "x3"})

    def test_stage_two_refines_the_upscaled_latent_and_decodes_it(self) -> None:
        import json as _json

        from PIL import Image

        import generate_batch as gb

        root = Path(__file__).resolve().parents[1]
        for workflow, depth in (("ltx_gemma_api.json", False), ("ltx_gemma_api_depth.json", True)):
            graph = gb.clone_workflow(_json.loads((root / "workflows" / workflow).read_text(encoding="utf-8")))
            with tempfile.TemporaryDirectory() as folder, patch.object(gb, "COMFY_INPUT_DIR", Path(folder)):
                still = Path(folder) / "still.png"
                Image.new("RGB", (768, 1360), (90, 90, 90)).save(still)
                size = gb.inject_spatial_upscale(graph, still, "x2", 0.7, 5, depth=depth)
            self.assertEqual(size, (896, 1536))
            self.assertEqual(graph["41"]["inputs"]["samples"], ["28", 2] if depth else ["25", 0], "guides cropped first")
            self.assertEqual(graph["47"]["inputs"]["guider"], ["50", 0] if depth else ["17", 0])
            if depth:
                self.assertEqual(graph["49"]["inputs"]["model"], ["11", 0], "the refine has no depth guide")
            self.assertEqual(graph["8"]["inputs"]["samples"], ["48", 0])
            self.assertEqual(graph["26"]["inputs"]["samples"], ["48", 1])


class VerifyTests(unittest.TestCase):
    def test_a_failed_output_is_remade_with_a_new_seed_and_reported(self) -> None:
        from generate_batch import verified

        with tempfile.TemporaryDirectory() as folder:
            dest = Path(folder) / "scene_01.mp4"
            seeds: list[int] = []
            answers = iter([["an invented person"], []])
            problems = verified(seeds.append, lambda: next(answers), 7, dest, 3)
            self.assertEqual(problems, [])
            self.assertEqual(len(seeds), 2)
            self.assertEqual(seeds[0], 7)
            self.assertNotEqual(seeds[1], 7, "a retry never repeats the failed seed")
            stuck = verified(seeds.append, lambda: ["an invented person"], 7, dest, 2)
            self.assertEqual(stuck, ["an invented person"])
            report = json.loads((Path(folder) / "inputs" / "scene_01" / "verify.json").read_text(encoding="utf-8"))
            self.assertEqual(report, {"tries": 2, "problems": ["an invented person"]})

    def test_a_person_where_the_scene_has_nobody_is_invented(self) -> None:
        from unittest import mock

        import verify

        frame = Image.new("RGB", (40, 40))
        expected = np.zeros((40, 40), dtype=bool)
        expected[:, :20] = True
        left, right = np.zeros_like(expected), np.zeros_like(expected)
        left[5:35, 5:15] = True
        right[5:35, 25:35] = True
        with mock.patch.object(verify, "people", return_value=[left]):
            self.assertEqual(verify.invented_people(frame, expected, 1), [])
        with mock.patch.object(verify, "people", return_value=[left, right]):
            found = verify.invented_people(frame, expected, 1)
            self.assertIn("stands where the 3D scene has nobody", found[0])
            self.assertIn("2 people in the picture, but the 3D scene has at most 1", found[1])
            fire = np.zeros_like(expected)
            fire[:, 22:] = True  # a tall flame the detector takes for a person
            self.assertEqual(verify.invented_people(frame, expected, 1, fire), [], "nobody is judged inside fire")
        leg = np.zeros_like(expected)
        leg[20:35, 5:10] = True  # a part of the same person, found again
        self.assertEqual(len(verify._distinct([left, leg])), 1, "one person is counted once")


class PasteTests(unittest.TestCase):
    def test_an_effect_is_added_as_light_only_where_the_shot_shows_it(self) -> None:
        from PIL import Image

        from generate_batch import add_light

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Image.new("RGB", (40, 80), (20, 30, 60)).save(root / "still.png")
            base = Image.new("RGB", (40, 80), (0, 0, 0))
            base.paste((90, 90, 100), (25, 10, 35, 30))  # its own stones, already pasted into the shot
            base.save(root / "base.png")
            fire = base.copy()
            fire.paste((150, 120, 100), (25, 10, 35, 30))  # the stones re-lit by the fire, not a flame
            fire.paste((255, 170, 40), (10, 10, 20, 30))
            fire.paste((255, 170, 40), (10, 50, 20, 70))
            fire.save(root / "fire.png")
            where = Image.new("L", (40, 80), 0)
            where.paste(255, (0, 0, 40, 40))  # only the upper half is the effect's box in the shot
            where.save(root / "where.png")
            solid = Image.new("L", (40, 80), 0)
            solid.paste(255, (25, 10, 35, 30))  # where the stones are
            solid.save(root / "solid.png")
            add_light(root / "still.png", [(root / "fire.png", root / "base.png", root / "solid.png", root / "where.png", [0.0, 0.0, 40.0, 80.0])])
            image = Image.open(root / "still.png")
            flame = image.getpixel((15, 20))
            self.assertGreater(flame[0], 240, "a flame brightens the shot")
            self.assertEqual(image.getpixel((30, 20)), (20, 30, 60), "what it redrew of itself adds nothing")
            self.assertEqual(image.getpixel((5, 20)), (20, 30, 60), "black adds nothing")
            self.assertEqual(image.getpixel((15, 60)), (20, 30, 60), "outside its box nothing is added")

    def test_drawings_are_laid_far_to_near(self) -> None:
        from generate_batch import paint_order

        seen = {
            "entities": {
                "ring": {"kind": "landmark", "depth": 6.0, "crop": [0, 0, 1, 1]},
                "man_behind": {"kind": "character", "depth": 7.5, "crop": [0, 0, 1, 1]},
                "flames_only": {"kind": "landmark", "depth": None, "crop": [0, 0, 1, 1]},
                "near": {"kind": "character", "depth": 2.0, "crop": [0, 0, 1, 1]},
                "sky": {"kind": "backdrop", "pixels": [10, 10]},
            }
        }
        self.assertEqual(paint_order(seen), ["flames_only", "man_behind", "ring", "near"])

    def test_a_drawing_lands_only_where_the_shot_shows_its_subject(self) -> None:
        from PIL import Image

        from generate_batch import paste_drawn

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Image.new("RGB", (40, 80), (0, 0, 255)).save(root / "color.png")
            Image.new("RGB", (40, 80), (255, 0, 0)).save(root / "drawn.png")
            Image.new("L", (40, 80), 255).save(root / "matte.png")  # the drawing fills its whole canvas
            shown = Image.new("L", (40, 80), 0)
            shown.paste(255, (2, 4, 18, 30))  # the subject is only here in the shot
            shown.save(root / "shown.png")
            out = paste_drawn(
                root / "color.png",
                [(root / "drawn.png", root / "matte.png", root / "shown.png", [0.0, 0.0, 20.0, 40.0])],
                root / "out.png",
                root / "draw.png",
            )
            image = Image.open(out)
            self.assertEqual(image.getpixel((10, 20)), (255, 0, 0), "where the shot shows it")
            self.assertEqual(image.getpixel((10, 35)), (0, 0, 255), "drawn there, but the subject is not there")
            self.assertEqual(image.getpixel((30, 60)), (0, 0, 255), "a window scales the drawing down only")
            draw = Image.open(root / "draw.png")
            self.assertEqual(draw.getpixel((10, 20)), 0, "a laid-in pixel is kept")
            self.assertEqual(draw.getpixel((10, 35)), 255, "everything else is drawn by the shot")
            self.assertEqual(json.loads((root / "coverage.json").read_text(encoding="utf-8")), {"drawn": 1.0})
            half = Image.new("L", (40, 80), 0)
            half.paste(255, (0, 0, 20, 80))  # the drawn subject stands beside where the mesh is
            half.save(root / "matte.png")
            paste_drawn(
                root / "color.png",
                [(root / "drawn.png", root / "matte.png", root / "shown.png", [0.0, 0.0, 20.0, 40.0])],
                root / "out.png",
                root / "draw.png",
            )
            self.assertLess(json.loads((root / "coverage.json").read_text(encoding="utf-8"))["drawn"], 0.85)

    def test_the_matte_looks_only_where_the_mesh_is(self) -> None:
        from generate_batch import REGION_GROW_PIXELS, mesh_region

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            depth = Image.new("RGB", (100, 100), (0, 0, 0))
            depth.paste((120, 120, 120), (0, 30, 10, 90))  # only an arm at the window's edge
            depth.save(root / "depth.png")
            region = Image.open(mesh_region(root / "depth.png", root / "region.png"))
            self.assertEqual(region.getpixel((5, 50)), 255)
            self.assertEqual(region.getpixel((10 + REGION_GROW_PIXELS - 2, 50)), 255, "hair and cloth may reach past it")
            self.assertEqual(region.getpixel((60, 50)), 0, "a whole person drawn beside it is not looked at")


class ShotTests(unittest.TestCase):
    """A shot's camera follows from the blocking: it frames its subject by construction."""

    def setUp(self) -> None:
        import math

        self.math = math
        tracks = {
            "ada": [{"timeSeconds": 0, "locationId": "room", "position": [0, 0, 0], "bodyYawDegrees": 0, "lookAtId": "bram"}],
            "bram": [{"timeSeconds": 0, "locationId": "room", "position": [0, 3, 0], "bodyYawDegrees": 180, "lookAtId": "ada"}],
        }
        self.show = show_with(tracks)
        self.show["characters"]["bram"] = dict(self.show["characters"]["ada"], heightMeters=1.8)

    def frame(self, **shot):
        from shots import frame_shot

        return frame_shot(self.show, self.show["episodes"][0], scene(shot=shot))

    def test_a_single_is_seen_along_its_gaze_with_the_head_in_frame(self) -> None:
        from shots import PEOPLE_FOV, SIZES

        pose = self.frame(type="single", subjects=["ada"], size="mcu")[0]
        self.assertGreater(pose["position"][1], 0.5, "in front of her, toward whom she looks")
        self.assertAlmostEqual(pose["position"][0], 0.0, places=3)
        distance = self.math.dist(pose["position"], pose["lookAt"])
        shown = 2 * distance * self.math.tan(self.math.radians(PEOPLE_FOV) / 2)
        self.assertAlmostEqual(shown, SIZES["mcu"], places=2, msg="the frame's height shows the medium close-up band")
        self.assertGreater(pose["lookAt"][2] + shown / 2, 1.7, "the top of the head is in frame")

    def test_angle_side_and_move(self) -> None:
        eye = self.frame(type="single", subjects=["ada"])[0]
        low = self.frame(type="single", subjects=["ada"], angle="low")[0]
        self.assertLess(low["position"][2], eye["position"][2], "a low angle looks up")
        left = self.frame(type="single", subjects=["ada"], side="left")[0]
        self.assertNotAlmostEqual(left["position"][0], 0.0, places=2, msg="a three-quarter view")
        push = self.frame(type="single", subjects=["ada"], move="pushIn")
        self.assertEqual(len(push), 2)
        self.assertLess(self.math.dist(push[1]["position"], push[1]["lookAt"]), self.math.dist(push[0]["position"], push[0]["lookAt"]))

    def test_over_the_shoulder_sits_behind_the_near_one(self) -> None:
        pose = self.frame(type="overShoulder", subjects=["bram"], over="ada", size="mcu")[0]
        self.assertLess(pose["position"][1], 0.0, "behind her, looking past her at him")
        self.assertAlmostEqual(pose["lookAt"][1], 3.0, places=3)

    def test_an_authored_camera_is_kept(self) -> None:
        from shots import resolve_cameras

        own = {"keyframes": [{"timeSeconds": 0, "position": [0, -5, 2], "lookAt": [0, 0, 1], "verticalFovDegrees": 50}]}
        single = scene(shot={"type": "single", "subjects": ["ada"]})
        del single["camera"]
        self.show["episodes"][0]["scenes"] = [scene(shot={"type": "establishing"}, camera=own), single]
        framed = resolve_cameras(self.show)["episodes"][0]["scenes"]
        self.assertIs(framed[0]["camera"], own)
        self.assertGreater(framed[1]["camera"]["keyframes"][0]["position"][1], 0.5, "framed from its subject")


class WindowTests(unittest.TestCase):
    def window(self, box):
        from previs import crop_window

        with patch("render.screen_box", lambda _entity, _camera: box):
            return crop_window(SimpleNamespace(effects=()), {})

    def test_a_person_cut_by_the_frame_edge_is_drawn_whole(self) -> None:
        from render import HEIGHT, WIDTH

        x0, y0, x1, y1 = self.window((-120.0, 300.0, 40.0, 900.0))
        self.assertLessEqual(x0, -120.0, "the window runs past the frame to hold the whole person")
        self.assertGreaterEqual(x1, 40.0)
        self.assertLessEqual(y0, 300.0)
        self.assertGreaterEqual(y1, 900.0)
        self.assertAlmostEqual((x1 - x0) / (y1 - y0), WIDTH / HEIGHT, places=2)

    def test_something_bigger_than_the_frame_is_never_scaled_up(self) -> None:
        from render import HEIGHT, WIDTH

        x0, y0, x1, y1 = self.window((100.0, 200.0, 700.0, 4000.0))
        self.assertGreaterEqual(x0, 0.0)
        self.assertGreaterEqual(y0, 0.0)
        self.assertLessEqual(x1, WIDTH + 0.01)
        self.assertLessEqual(y1, HEIGHT + 0.01)

    def test_out_of_frame_has_no_window(self) -> None:
        self.assertIsNone(self.window((-300.0, 100.0, -10.0, 500.0)))


if __name__ == "__main__":
    unittest.main()
