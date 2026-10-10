"""Cameras from shot types: a scene says what it shows, and the camera follows from the blocking.

A scene's ``shot`` names its type and subjects ("single, adam, mcu, low"). For
every shot of people the camera is computed here from where the subjects stand
and whom they look at, so a shot frames its subject by construction; nobody
types camera coordinates. Only an ``establishing`` or ``action`` shot may give
its own ``camera`` instead (a landscape, a chase).

Every shot of people is framed the same way: a band of the subject's height
(``SIZES``: from the top of the head down) fills the frame's height, seen from
along the subject's gaze, turned by ``side``, raised or lowered by ``angle``.
"""

from __future__ import annotations

import math

from world import character_state

# Meters of the subject the frame's height shows, from the top of the head down.
SIZES = {"ecu": 0.24, "cu": 0.42, "mcu": 0.72, "medium": 1.1, "full": None, "wide": None}
# A full shot shows the whole body with this much room; a wide one this much more.
FULL_ROOM, WIDE_ROOM = 1.18, 2.2
# Camera elevation above the subject's center, in degrees.
ANGLES = {"eye": 0.0, "low": -18.0, "high": 22.0}
# Horizontal turn off the gaze line, in degrees: a three-quarter view.
SIDES = {"front": 0.0, "left": 32.0, "right": -32.0}
# A push-in ends this much closer, a pull-out this much farther.
MOVES = {"static": 1.0, "pushIn": 0.82, "pullOut": 1.22}
# Lenses: people are shot long (flattering, little distortion), wide shots wider.
PEOPLE_FOV, WIDE_FOV = 30.0, 45.0
# Over the shoulder: the camera sits this far behind and beside the near one, at this
# share of their height (just above the shoulder), with at least this lens so the
# shoulder is at the frame's edge.
OTS_BACK, OTS_SIDE, OTS_HEIGHT, OTS_MIN_FOV = 0.55, 0.3, 0.86, 22.0
# Vertical frame over horizontal frame.
ASPECT = 1360 / 768
# Height bands of a body part, as fractions of height (see body_parts.py).
PART_BANDS = {
    "hair": (0.9, 1.0), "face": (0.86, 0.97), "eyes": (0.87, 0.92), "neck": (0.8, 0.87),
    "torso": (0.5, 0.8), "arms": (0.45, 0.8), "hands": (0.4, 0.54), "legs": (0.05, 0.5), "feet": (0.0, 0.08),
}


def _people(show: dict, episode: dict, scene: dict) -> dict[str, dict]:
    """Where everyone in the scene's location stands at its start: position, height, and gaze direction."""
    start = float(scene["timeRangeSeconds"][0])
    tracks = episode["spatialTimeline"]["characterTracks"]
    states = {cid: character_state(track, start) for cid, track in tracks.items()}
    here = {cid: state for cid, state in states.items() if state.get("locationId") == scene["locationId"]}
    landmarks = show["locations"][scene["locationId"]]["spatial"]["landmarks"]
    people = {}
    for cid, state in here.items():
        x, y = float(state["position"][0]), float(state["position"][1])
        target = state.get("lookAtId")
        point = here[target]["position"] if target in here else landmarks[target]["position"] if target in landmarks else None
        yaw = math.radians(state["bodyYawDegrees"])
        gaze = (math.sin(yaw), math.cos(yaw))
        if point is not None and math.hypot(point[0] - x, point[1] - y) > 1e-3:
            length = math.hypot(point[0] - x, point[1] - y)
            gaze = ((point[0] - x) / length, (point[1] - y) / length)
        people[cid] = {"x": x, "y": y, "z": float(state["position"][2]), "height": float(show["characters"][cid]["heightMeters"]), "gaze": gaze}
    return people


def _turn(vector: tuple[float, float], degrees: float) -> tuple[float, float]:
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    return (vector[0] * c - vector[1] * s, vector[0] * s + vector[1] * c)


def _keyframes(scene: dict, look: list[float], toward: tuple[float, float, float], distance: float, fov: float) -> list[dict]:
    """The camera ``distance`` from ``look`` along ``toward`` (a unit vector from the subject to the camera)."""
    move = MOVES[scene["shot"].get("move", "static")]
    start, finish = (float(value) for value in scene["timeRangeSeconds"])

    def pose(time_seconds: float, scale: float) -> dict:
        return {
            "timeSeconds": time_seconds,
            "position": [round(look[i] + toward[i] * distance * scale, 3) for i in range(3)],
            "lookAt": [round(value, 3) for value in look],
            "verticalFovDegrees": fov,
        }

    return [pose(start, 1.0)] if move == 1.0 else [pose(start, 1.0), pose(finish, move)]


def _direction(gaze: tuple[float, float], side: str, angle: str) -> tuple[float, float, float]:
    horizontal = _turn(gaze, SIDES[side])
    elevation = math.radians(ANGLES[angle])
    return (horizontal[0] * math.cos(elevation), horizontal[1] * math.cos(elevation), math.sin(elevation))


def _span_distance(span: float, fov: float) -> float:
    return span / (2.0 * math.tan(math.radians(fov) / 2.0))


# A person, for the camera's line of sight: a column this wide around their feet.
BODY_RADIUS = 0.32
# Views tried after the requested one, turning around the subject until one is clear.
TURNS = (0.0, 20.0, -20.0, 40.0, -40.0, 60.0, -60.0, 90.0, -90.0)


def _solids(show: dict, scene: dict, people: dict[str, dict], allowed: set[str]) -> list[tuple]:
    """Everything a camera must not stand in or look through: (x, y, half x, half y, top)."""
    landmarks = show["locations"][scene["locationId"]]["spatial"]["landmarks"]
    found = [(p["x"], p["y"], BODY_RADIUS, BODY_RADIUS, p["z"] + p["height"]) for cid, p in people.items() if cid not in allowed]
    for landmark in landmarks.values():
        (x, y, _z), (width, depth, height) = landmark["position"], landmark["size"]
        found.append((x, y, width / 2.0, depth / 2.0, height))
    return found


def _clear(camera: list[float], look: list[float], solids: list[tuple]) -> bool:
    """Nothing in the way: the camera is in no solid, and its line of sight to ``look`` crosses none."""
    for step in range(0, 20):
        amount = step / 20.0
        point = [camera[i] + (look[i] - camera[i]) * amount for i in range(3)]
        for x, y, half_x, half_y, top in solids:
            if abs(point[0] - x) < half_x and abs(point[1] - y) < half_y and point[2] < top:
                return False
    return True


def _search(scene: dict, look: list[float], gaze, side: str, angle: str, distance: float, fov: float, solids) -> list[dict]:
    """The requested view, or the nearest turn of it around the subject whose line of sight is clear."""
    for turn in TURNS:
        toward = _direction(_turn(gaze, turn), side, angle)
        camera = [look[i] + toward[i] * distance for i in range(3)]
        if _clear(camera, look, solids):
            return _keyframes(scene, look, toward, distance, fov)
    return _keyframes(scene, look, _direction(gaze, side, angle), distance, fov)


def frame_shot(show: dict, episode: dict, scene: dict) -> list[dict]:
    """Camera keyframes for a scene's shot, from the blocking at its start.

    The camera never stands in anyone or anything, and nobody else stands between
    it and its subjects: when the requested view is blocked it turns around them
    to the nearest clear one.
    """
    shot = scene["shot"]
    people = _people(show, episode, scene)
    subjects = [people[cid] for cid in shot.get("subjects", [])]
    angle, side = shot.get("angle", "eye"), shot.get("side", "front")
    kind = shot["type"]
    solids = _solids(show, scene, people, set(shot.get("subjects", [])))
    if kind == "overShoulder":
        # Low behind the near shoulder, with a lens wide enough that the shoulder is in frame.
        seen, near = subjects[0], people[shot["over"]]
        to_seen = (seen["x"] - near["x"], seen["y"] - near["y"])
        length = math.hypot(*to_seen)
        to_seen = (to_seen[0] / length, to_seen[1] / length)
        across = _turn(to_seen, -90.0 if side == "right" else 90.0)
        camera = [
            near["x"] - to_seen[0] * OTS_BACK + across[0] * OTS_SIDE,
            near["y"] - to_seen[1] * OTS_BACK + across[1] * OTS_SIDE,
            near["z"] + near["height"] * OTS_HEIGHT,
        ]
        span = SIZES[shot.get("size", "mcu")]
        look = [seen["x"], seen["y"], seen["z"] + seen["height"] + 0.04 - span * 0.45]
        distance = math.dist(camera, look)
        fov = max(round(math.degrees(2.0 * math.atan(span / 2.0 / distance)), 2), OTS_MIN_FOV)
        toward = tuple((camera[i] - look[i]) / distance for i in range(3))
        return _keyframes(scene, look, toward, distance, fov)
    if kind == "insert":
        subject = subjects[0]
        low, high = PART_BANDS[shot["part"]]
        span = max((high - low) * subject["height"] * 1.6, 0.18)
        look = [subject["x"], subject["y"], subject["z"] + (low + high) / 2.0 * subject["height"]]
        return _search(scene, look, subject["gaze"], side, angle, _span_distance(span, PEOPLE_FOV), PEOPLE_FOV, solids)
    # single, group, action: the subjects' heads at the top, a band of their height filling the frame.
    size = shot.get("size", "mcu" if kind == "single" else "medium")
    gaze = (sum(s["gaze"][0] for s in subjects), sum(s["gaze"][1] for s in subjects))
    length = math.hypot(*gaze) or 1.0
    gaze = (gaze[0] / length, gaze[1] / length)
    tall = max(s["z"] + s["height"] for s in subjects)
    center = (sum(s["x"] for s in subjects) / len(subjects), sum(s["y"] for s in subjects) / len(subjects))
    across = _turn(gaze, 90.0)
    spread = max(s["x"] * across[0] + s["y"] * across[1] for s in subjects) - min(s["x"] * across[0] + s["y"] * across[1] for s in subjects)
    fov = WIDE_FOV if size == "wide" else PEOPLE_FOV
    if SIZES[size] is None:
        span = tall * (WIDE_ROOM if size == "wide" else FULL_ROOM)
        look = [center[0], center[1], tall * 0.5]
    else:
        span = SIZES[size]
        look = [center[0], center[1], tall + 0.04 - span * 0.45]
    # Several people must all fit across the frame; one person is cut by the frame like any close shot.
    width = (spread + 0.7) * ASPECT if len(subjects) > 1 else 0.0
    distance = _span_distance(max(span, width), fov)
    depth = max(abs((s["x"] - center[0]) * gaze[0] + (s["y"] - center[1]) * gaze[1]) for s in subjects)
    return _search(scene, look, gaze, side, angle, distance + depth, fov, solids)


def resolve_cameras(show: dict) -> dict:
    """Give every scene that has no ``camera`` the one its shot frames. Returns the show."""
    for episode in show["episodes"]:
        for scene in episode["scenes"]:
            if "camera" not in scene and "shot" in scene:
                scene["camera"] = {"keyframes": frame_shot(show, episode, scene)}
    return show
