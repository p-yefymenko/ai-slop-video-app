"""The world, the visibility rule, the words, and the checks."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import world  # noqa: E402
from body_parts import BODY_PARTS  # noqa: E402
from checks import check_scene  # noqa: E402
from describe import drawn_prompt, ltx_prompt, people_places, still_prompt  # noqa: E402
from mesh_io import write_schema_glb  # noqa: E402
from observe import passes, visible  # noqa: E402


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
    def test_one_rule_for_every_region(self) -> None:
        self.assertTrue(passes(5000, 6000))
        self.assertFalse(passes(1800, 42000), "a face seen from behind is a sliver of its extent")
        self.assertFalse(passes(100, 100), "too small to read")
        seen = observation({"ada": BEHIND})
        self.assertTrue(visible(seen, "ada"))
        self.assertTrue(visible(seen, "ada", ["hair"]))
        self.assertFalse(visible(seen, "ada", ["face"]))
        self.assertFalse(visible(seen, "nobody"))


class DescribeTests(unittest.TestCase):
    def test_a_person_is_drawn_whole_from_their_own_words(self) -> None:
        show = show_with({})
        shot = scene(performances={"ada": {"action": "waits", "parts": ["torso"], "expression": "jaw set"}})
        prompt = drawn_prompt(show, shot, "character", "ada", ["Depth", "OwnColor", "Edges", "Appearance"])
        self.assertIn("Adult woman, brown skin, 52 years old, black hair, jaw set.", prompt)
        self.assertIn("Picture 4 is how it looks", prompt)
        self.assertIn("One person, alone", prompt)
        self.assertNotIn("ada", prompt)

    def test_the_shot_prompt_names_no_person(self) -> None:
        show = show_with({})
        prompt, _log = still_prompt(show, scene(), observation({"ada": FRONT}), ["Depth", "Composite", "Edges"])
        self.assertIn("One person, already painted.", prompt)
        self.assertIn("already painted are finished and stay exactly as they are", prompt)
        self.assertNotIn("52 years old", prompt)
        self.assertNotIn("black hair", prompt)

    def test_landmarks_props_and_backdrop_use_the_same_rule(self) -> None:
        show = show_with({})
        seen = observation(
            {},
            bench={"kind": "landmark", "pixels": [9000, 10000], "regions": {"whole": [9000, 10000]}},
            sky={"kind": "backdrop", "pixels": [20000, 20000], "regions": {"whole": [20000, 20000]}},
            ground={"kind": "backdrop", "pixels": [40, 40], "regions": {"whole": [40, 40]}},
        )
        prompt, _log = still_prompt(show, scene(), seen, ["Depth", "Composite", "Edges"])
        self.assertIn("One stone bench.", prompt)
        self.assertIn("Behind and around: A gray sky.", prompt)
        self.assertNotIn("stone paving", prompt)
        hidden = observation({}, bench={"kind": "landmark", "pixels": [900, 10000], "regions": {"whole": [900, 10000]}})
        self.assertNotIn("bench", still_prompt(show, scene(), hidden, ["Depth", "Composite", "Edges"])[0])

    def test_ltx_names_people_by_where_the_start_frame_shows_them(self) -> None:
        show = show_with({})
        shot = scene(performances={"ada": {"action": "waits", "parts": ["torso"]}})
        start = observation({"ada": FRONT})
        self.assertEqual(people_places(start), {"ada": "In the center"})
        prompt = ltx_prompt(show, shot, start)
        self.assertIn("In the center, adult woman, brown skin: waits.", prompt)
        self.assertIn("No music plays.", prompt)


class CheckTests(unittest.TestCase):
    def test_the_script_must_match_what_is_visible(self) -> None:
        show = show_with({})
        start = observation({"ada": BEHIND})
        unlisted = check_scene(show, show["episodes"][0], scene(), [(0.0, start, [])])
        self.assertTrue(any("ada is visible from 0s but has no performance" in e for e in unlisted))
        shot = scene(
            speakerId="ada",
            performances={"ada": {"action": "talks", "parts": ["face"], "expression": "jaw set"}},
            sound={"events": [{"text": "a breath", "source": {"landmarkId": "bench"}}], "bed": "faint", "music": {"kind": "none"}},
        )
        errors = "\n".join(check_scene(show, show["episodes"][0], shot, [(0.0, start, [])]))
        self.assertIn("parts names face, but their face is never visible in the shot (start: 1800 px visible of 42000 px)", errors)
        self.assertIn("expression is set, but their face is not visible", errors)
        self.assertIn("speaker ada's face is 1800 px visible of 42000 px in the start frame; lip-sync needs at least", errors)
        self.assertIn("comes from bench, which is not visible", errors)
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
            self.assertEqual(passes(shown, extent), expected, (position, shown, extent))


if __name__ == "__main__":
    unittest.main()


class PasteTests(unittest.TestCase):
    def test_a_drawn_person_lands_only_where_the_shot_shows_them(self) -> None:
        from PIL import Image

        from generate_batch import paste_drawn

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Image.new("RGB", (40, 80), (0, 0, 255)).save(root / "color.png")
            Image.new("RGB", (20, 40), (255, 0, 0)).save(root / "drawn.png")
            mask = Image.new("L", (40, 80), 0)
            mask.paste(255, (10, 10, 20, 30))
            mask.save(root / "mask.png")
            out = paste_drawn(
                root / "color.png", [(root / "drawn.png", root / "mask.png", [5.0, 5.0, 25.0, 45.0])], root / "out.png", root / "draw.png"
            )
            image = Image.open(out)
            self.assertEqual(image.getpixel((15, 20)), (255, 0, 0))
            self.assertEqual(image.getpixel((22, 20)), (0, 0, 255), "drawn but hidden in the shot")
            self.assertEqual(image.getpixel((2, 2)), (0, 0, 255))
            draw = Image.open(root / "draw.png")
            self.assertEqual(draw.getpixel((15, 20)), 0, "a pasted pixel is kept")
            self.assertEqual(draw.getpixel((10, 10)), 255, "its edge is redrawn to sit in the shot")
            self.assertEqual(draw.getpixel((30, 60)), 255, "everything else is drawn")
