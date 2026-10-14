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
from checks import check_scene  # noqa: E402
from mesh_io import write_schema_glb  # noqa: E402
from observe import in_shot  # noqa: E402


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


def observation(people: dict[str, int], **entries) -> dict:
    """A start frame: each person by their pixels in the frame, in the center, 4 m away."""
    found = {cid: {"kind": "character", "pixels": pixels, "screenX": 380.0, "depth": 4.0} for cid, pixels in people.items()}
    found.update(entries)
    return {"timeSeconds": 0.0, "camera": {"position": [0, -4, 1.5], "lookAt": [0, 0, 1.2], "verticalFovDegrees": 40}, "entities": found}



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


class InShotTests(unittest.TestCase):
    def test_in_shot_is_any_pixel(self) -> None:
        self.assertTrue(in_shot(observation({"ada": 1}), "ada"))
        self.assertFalse(in_shot(observation({"ada": 0}), "ada"))
        self.assertFalse(in_shot(observation({}), "nobody"))


class CheckTests(unittest.TestCase):
    def test_the_script_must_match_what_is_in_the_frame(self) -> None:
        show = show_with({})
        start = observation({"ada": 5000})
        self.assertEqual(check_scene(show, show["episodes"][0], scene(), [(0.0, start, [])]), [], "nobody without a performance needs one")
        aimed = scene(performances={"ada": {"action": "talks to {target}", "target": "bench"}})
        errors = check_scene(show, show["episodes"][0], aimed, [(0.0, start, [])])
        self.assertEqual(errors, ["scene 1: performances.ada.target is bench, which is not in the start frame; frame it, or drop the target and keep the action to their own body."])
        absent = check_scene(show, show["episodes"][0], aimed, [(0.0, observation({}), [])])
        self.assertIn("scene 1: performances.ada is set, but ada is never in the frame.", absent)


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
        tilt = self.frame(type="single", subjects=["ada"], move="tiltUp")
        self.assertEqual(tilt[0]["position"], tilt[1]["position"], "a tilt keeps its place")
        self.assertLess(tilt[0]["lookAt"][2], 0.3, "and starts on the feet")
        self.assertEqual(tilt[1]["lookAt"], eye["lookAt"], "and ends on the framing")

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


if __name__ == "__main__":
    unittest.main()
