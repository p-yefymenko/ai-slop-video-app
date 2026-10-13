"""Words for Qwen and LTX, taken from the script and an observation.

A still with at most one person is drawn whole in one pass (``whole_prompt``):
the shot's depth for the layout, the person's plate for who they are, and plain
words for everything else. Any other still never asks words to find their own
place in a picture: every person, prop, and landmark in it is drawn alone from
its own guides and plate, with only its own words, and pasted where the shot
shows it; the shot pass is told only the sky, ground, and surround in the shot.
Character names and ids never reach a model.
"""

from __future__ import annotations

import math
import re

from observe import in_shot, visible
from render import HEIGHT, WIDTH
from world import camera_moves, character_appearance_text, clause, definition, descriptions, settings

PLACEHOLDER = re.compile(r"\{([a-zA-Z][a-zA-Z0-9]*)\}")
PICTURE_LEGEND = {
    "Depth": "Picture {n} is depth: brighter is closer, black is empty space.",
    "Edges": "Picture {n} is outlines.",
    "Composite": (
        "Picture {n} is the colors of everything in the shot; the people and objects in it that are "
        "already painted are finished and stay exactly as they are."
    ),
    "OwnColor": "Picture {n} is its colors.",
    "Appearance": "Picture {n} is how it looks: the same face, hair, skin, clothes, colors, and materials.",
    "Look": "Picture {n} is how it looks with its effects.",
    "Base": "Picture {n} is the object on pure black.",
}


def template(key: str, values: dict[str, str]) -> str:
    """Fill a ``prompts.json`` template. Every placeholder must be given and used."""
    unused = set(values)

    def fill(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in values:
            raise SystemExit(f"prompts.{key} uses {{{name}}}, which is not provided")
        unused.discard(name)
        return values[name]

    text = PLACEHOLDER.sub(fill, settings()[key]).strip()
    if unused:
        raise SystemExit(f"prompts.{key} never uses {', '.join(sorted(unused))}")
    return text


def _sentence(text: str) -> str:
    text = clause(text)
    return text[:1].upper() + text[1:] + "." if text else ""


def _side(screen_x: float) -> str:
    if screen_x < WIDTH * 0.34:
        return "On the left"
    if screen_x > WIDTH * 0.66:
        return "On the right"
    return "In the center"


def people_places(observation: dict) -> dict[str, str]:
    """characterId -> where the person stands in this frame, in words.

    Everyone in the shot counts, named by screen side. When some stand well
    behind the nearest (deeper than ``behindDepthRatio`` times), each also says
    whether they are in the foreground or farther back. Never a "row": the
    word makes the video model cut to a lineup of people.
    """
    people = [
        (entity_id, entry)
        for entity_id, entry in observation["entities"].items()
        if entry["kind"] == "character" and in_shot(observation, entity_id)
    ]
    if not people:
        return {}
    nearest = min(entry["depth"] for _id, entry in people)
    behind = {entity_id for entity_id, entry in people if entry["depth"] > nearest * settings()["behindDepthRatio"]}
    places: dict[str, str] = {}
    for entity_id, entry in people:
        depth = (", farther back" if entity_id in behind else ", in the foreground") if behind else ""
        places[entity_id] = _side(entry["screenX"]) + depth
    return places


def setting_sentences(show: dict, scene: dict, observation: dict) -> str:
    """The sky, ground, and surround in the shot. Each fills its own stretch of the frame."""
    said: list[str] = []
    for entity_id, entry in observation["entities"].items():
        if entry["kind"] != "backdrop" or not in_shot(observation, entity_id):
            continue
        for text, _regions in descriptions(show, scene, "backdrop", entity_id):
            sentence = _sentence(text)
            if sentence and sentence not in said:
                said.append(sentence)
    return " ".join(said)


def _light(show: dict, scene: dict) -> str:
    """The location's one light, said in every pass of a still, so separately drawn things match."""
    return template("stillLight", {"light": clause(show["locations"][scene["locationId"]]["look"]["light"])})


def _legend(pictures: list[str]) -> str:
    return " ".join(PICTURE_LEGEND[title].format(n=number) for number, title in enumerate(pictures, start=1))


def drawn_prompt(show: dict, scene: dict, kind: str, entity_id: str, pictures: list[str]) -> str:
    """One person, prop, or landmark drawn whole and alone, solid parts only, from its own guides and only its own words.

    The shot keeps only the part of it the camera shows, so nothing here depends
    on what is visible.
    """
    if kind == "character":
        subject = "One person, alone"
        words = [character_appearance_text(show["characters"][entity_id])]
        expression = ((scene.get("performances") or {}).get(entity_id) or {}).get("expression")
        if expression:
            words.append(clause(expression))
    else:
        subject = "One object, alone"
        words = list(descriptions(show, scene, kind, entity_id)[0][:1])
    parts = [
        clause(settings()["stillOpening"]) + ".",
        _legend(pictures),
        f"Keep the shape, pose, and turn from the pictures. {subject}, on a plain light gray background.",
        _sentence(", ".join(words)),
        _light(show, scene),
    ]
    return " ".join(part for part in parts if part)


def effect_prompt(show: dict, scene: dict, kind: str, entity_id: str, pictures: list[str]) -> str:
    """Only an entity's effects, as light on pure black, in its own window: added to the shot as light."""
    effects = definition(show, kind, entity_id, scene["locationId"]).get("effects") or ()
    words = "; ".join(dict.fromkeys(clause(effect["appearance"]) for effect in effects))
    return " ".join([_legend(pictures), template("effectDrawn", {"effects": words})])


def still_prompt(show: dict, scene: dict, observation: dict, pictures: list[str], drawn: int) -> tuple[str, dict]:
    """The shot around what is already painted: legend, keep, and the setting. No object is named here."""
    setting = setting_sentences(show, scene, observation)
    parts = [
        clause(settings()["stillOpening"]) + ".",
        _legend(pictures),
        "Keep the shape, position, and occlusion from the pictures.",
        "Everything standing in the shot is already painted; draw only what is around it." if drawn else "",
        f"Behind and around: {setting}" if setting else "",
        _light(show, scene),
    ]
    return " ".join(part for part in parts if part), {"drawn": drawn, "setting": setting}


def facing_key(gaze: tuple[float, float], position: tuple[float, float], camera: dict) -> tuple[str, str]:
    """How a person is turned to the camera (camera, three, profile, away) and to which side of the frame they look."""
    to_camera = (camera["position"][0] - position[0], camera["position"][1] - position[1])
    angle = abs(math.degrees(math.atan2(gaze[0] * to_camera[1] - gaze[1] * to_camera[0], gaze[0] * to_camera[0] + gaze[1] * to_camera[1])))
    forward = (camera["lookAt"][0] - camera["position"][0], camera["lookAt"][1] - camera["position"][1])
    side = "right" if gaze[0] * forward[1] - gaze[1] * forward[0] > 0 else "left"
    return ("camera" if angle < 20 else "three" if angle < 65 else "profile" if angle < 115 else "away"), side


def facing_words(gaze: tuple[float, float], position: tuple[float, float], camera: dict) -> str:
    """How a person is turned to the camera, in words: their gaze against the line to the camera."""
    key, side = facing_key(gaze, position, camera)
    return settings()["wholeFacing"][key].format(side=side)


# Heights of the body (shares of standing height) and what shows when a shot's band covers them.
BODY_SPANS = (("feet", 0.0, 0.08), ("legs", 0.08, 0.5), ("waist", 0.45, 0.6), ("chest", 0.6, 0.82), ("head", 0.86, 1.0))


def band_words(band: list[float] | None) -> str:
    """Which parts of a person a shot's height band (from the feet, shares of height) shows, in words."""
    if not band:
        return ""
    low, high = band
    shown = [name for name, start, end in BODY_SPANS if min(high, end) - max(low, start) > 0.3 * (end - start)]
    return settings()["wholeBodyParts"].get("+".join(shown), "")


def view_words(camera: dict) -> str:
    """A camera looking steeply down or up, in words."""
    position, look = camera["position"], camera["lookAt"]
    pitch = math.degrees(math.atan2(look[2] - position[2], math.hypot(look[0] - position[0], look[1] - position[1])))
    view = settings()["wholeView"]
    return view["high"] if pitch < -25 else view["low"] if pitch > 12 else ""


def frame_place(observation: dict, cid: str) -> str:
    """Where a person is in the frame: left, center, or right."""
    return _side(observation["entities"][cid]["screenX"])


def whole_prompt(
    show: dict,
    scene: dict,
    observation: dict,
    person: str | None,
    part: bool = False,
    facing: str | None = None,
    behind: bool = False,
    near_pictures: dict[str, int] | None = None,
    far_pictures: dict[str, int] | None = None,
) -> str:
    """A still drawn in one pass: picture 1 is the depth of the shot's solids, picture 2 the plate of ``person``.

    ``facing`` is how they are turned to the camera (``facing_words``): picture 1
    shows them only as a flat silhouette. ``behind``: they face away, and picture
    2 is their back. Someone nearer than them (a shoulder in front) gets no
    description: their words bleed into the subject (a beard, a coat drawn on
    the wrong person). They are a blurred back of a head and shoulder, dressed
    like their back-view picture when ``near_pictures`` names one. Someone
    farther away is their own picture when ``far_pictures`` names one, else only
    the first words of their body. ``part`` is a shot showing only part of the person, with the plate cut to
    that part: their words would name clothes the shot cannot show (boots in a
    close-up), and the model draws the whole person small to fit them in, so the
    picture alone says how they look. Objects are said in words, never shown:
    given an object's picture the model pastes that picture in. Each stretch of
    the backdrop is said by where it is, so the ground stays ground.
    """
    parts = [settings()["wholeOpening" if person is not None else "wholeOpeningEmpty"], view_words(observation["camera"]), settings()["wholeLayout"]]
    places = {cid: frame_place(observation, cid) for cid, entry in observation["entities"].items() if entry["kind"] == "character" and in_shot(observation, cid)}
    if person is not None:
        expression = ((scene.get("performances") or {}).get(person) or {}).get("expression")
        values = {
            "place": places[person],
            "expression": f", {clause(expression)}" if expression else "",
            "facing": f", {facing}" if facing else "",
            "shows": band_words(observation["entities"][person].get("band")) if part else "",
        }
        key = "wholePersonBehind" if behind else "wholePersonPart" if part else "wholePerson"
        if key != "wholePersonPart":
            values["body"] = clause(character_appearance_text(show["characters"][person]))
        if key != "wholePersonPart":
            del values["shows"]
        if key == "wholePersonBehind":
            del values["expression"], values["facing"]  # a face turned away shows none
        parts.append(template(key, values))
    for cid, entry in observation["entities"].items():
        if entry["kind"] == "character" and cid != person and in_shot(observation, cid):
            near = person is not None and entry["depth"] < observation["entities"][person]["depth"]
            if near:
                number = (near_pictures or {}).get(cid)
                dressed = template("wholeOtherDressed", {"n": str(number)}) if number else ""
                parts.append(template("wholeOtherNear", {"place": places.get(cid, "In the frame"), "dressed": dressed}))
            else:
                # Farther away too: a full description bleeds into the subject (armor on the wrong person).
                number = (far_pictures or {}).get(cid)
                if number:
                    parts.append(template("wholeOtherFarSeen", {"place": places.get(cid, "In the frame"), "n": str(number)}))
                else:
                    parts.append(template("wholeOtherFar", {"place": places.get(cid, "In the frame"), "who": _voice(show, cid)}))
    for entity_id, entry in observation["entities"].items():
        if entry["kind"] in ("landmark", "prop") and in_shot(observation, entity_id):
            parts.append(template("wholeThing", {"appearance": clause(descriptions(show, scene, entry["kind"], entity_id)[0][0])}))
    backdrop = settings()["wholeBackdrop"]
    for entity_id, entry in observation["entities"].items():
        if entry["kind"] == "backdrop" and in_shot(observation, entity_id):
            parts.append(backdrop[entity_id].format(text=clause(descriptions(show, scene, "backdrop", entity_id)[0][0])))
    parts.append(_light(show, scene))
    parts.append(settings()["wholeNobodyElse" if person is not None else "wholeNobody"])
    return " ".join(part for part in parts if part)


def sound_sentence(show: dict, scene: dict) -> str:
    """Location bed and space, voices from off screen, the scene's events, and the music policy."""
    soundscape = show["locations"][scene["locationId"]]["soundscape"]
    sound = scene["sound"]
    events = "; ".join(clause(event["text"]) for event in sound["events"])
    music = sound["music"]
    text = template(
        "sceneSound",
        {
            "label": template("sceneSoundLabel", {}),
            "bedPhrase": template(
                "sceneSoundBedPresent" if sound["bed"] == "present" else "sceneSoundBedFaint",
                {"ambience": _sentence(soundscape["ambience"])[:-1]},
            ),
            "spacePhrase": template("sceneSoundSpace", {"space": _sentence(soundscape["space"])[:-1]}),
            "voicesPhrase": " ".join(line_sentences(show, scene, None, on_screen=False)),
            "eventsPhrase": template("sceneSoundEvents", {"events": _sentence(events)[:-1]}),
            "musicPhrase": template("sceneSoundMusicNone", {})
            if music["kind"] == "none"
            else template("sceneSoundMusicDescribed", {"description": _sentence(music["description"])[:-1]}),
        },
    )
    return " ".join(text.split())


_ARTICLE = re.compile(r"^(one|an?|the)\s+", re.IGNORECASE)


def target_words(show: dict, scene: dict, start: dict, target: str) -> str:
    """What an action is aimed at, as the start frame shows it: a person by body and
    place, a thing by its effects when the frame shows them, else by its first words.

    The script checks keep every target in the start frame, so these words always
    point at something LTX can find there.
    """
    if target in show["characters"]:
        place = people_places(start).get(target)
        body = clause(show["characters"][target]["body"])
        return f"the {body}, {place[:1].lower() + place[1:]}" if place else f"the {body}"
    effects = [clause(effect["appearance"]) for effect in start.get("effects") or [] if effect["id"] == target]
    if effects:
        return "the " + _ARTICLE.sub("", effects[0])
    kind = "prop" if target in (show.get("props") or {}) else "landmark"
    return "the " + _ARTICLE.sub("", re.split(r"[,.;:]", descriptions(show, scene, kind, target)[0][0])[0].strip())


def _voice(show: dict, cid: str) -> str:
    """A voice from off screen: the first words of the speaker's body ("a middle-aged man's voice")."""
    who = clause(show["characters"][cid]["body"]).split(",")[0].strip()
    return ("an " if who[:1].lower() in "aeiou" else "a ") + who + "'s voice"


# A bystander smaller than this share of the frame is left out of the clip prompt:
# named, a distant figure or a sliver at the edge makes the model reframe to show them.
STILL_MIN_SHARE = 0.03

# A speech starting later than this into the take says when.
LATE_LINE_SECONDS = 1.0


def line_sentences(show: dict, scene: dict, person, on_screen: bool = True) -> list[str]:
    """What is said, in order: by on-screen speakers, named by where the frame shows
    them, or (``on_screen`` false) by voices from off screen, which go with the sound:
    told among the picture's actions, the video model cuts to show the speaker.

    One speaker's lines in a row are one continuous speech: timed separately
    ("At 0.7 seconds ... At 3.7 seconds ..."), the video model cuts to a new shot
    at each. A speech says when it starts only when that is late in the take.
    """
    speeches: list[list[dict]] = []
    for line in scene.get("lines") or []:
        if speeches and speeches[-1][0]["speakerId"] == line["speakerId"]:
            speeches[-1].append(line)
        else:
            speeches.append([line])
    performers = scene.get("performances") or {}
    speeches = [speech for speech in speeches if (speech[0]["speakerId"] in performers) == on_screen]
    sentences = []
    for index, speech in enumerate(speeches):
        first = speech[0]
        words = " ".join(" ".join(line["line"].split()) for line in speech)
        values = {"delivery": clause(first["delivery"]), "line": words}
        if on_screen:
            values["person"] = person(first["speakerId"])
        else:
            values["voice"] = _voice(show, first["speakerId"])
        text = template("sceneLine" if on_screen else "sceneLineOffScreen", values)
        start = first.get("startSeconds")
        follows = text[:1].lower() + text[1:]  # "In the center, ..." mid-sentence
        if start is not None and float(start) >= LATE_LINE_SECONDS:
            text = template("sceneLineAt", {"seconds": f"{float(start):g}", "line": follows})
        elif index:
            text = template("sceneLineThen", {"line": follows})
        sentences.append(text[:1].upper() + text[1:])
    return sentences


def camera_sentence(scene: dict) -> str:
    """How the camera moves: an own camera's words, a shot's move, else held by hand."""
    if scene.get("cameraMotion"):
        return template("sceneCameraMotion", {"motion": clause(scene["cameraMotion"])})
    move = (scene.get("shot") or {}).get("move", "static")
    if move != "static" and camera_moves(scene):
        return settings()["sceneCameraMoves"][move]
    return template("sceneCameraHeld", {})


def ltx_prompt(show: dict, scene: dict, start: dict, end: dict | None = None) -> str:
    """Lines, performances, motion, the camera, then sound.

    People are named by where the start frame shows them, plus their body, so
    LTX can find them in the first frame. Script text never points at anyone; an
    action aimed at someone says ``{target}``, filled here from the start frame.
    """
    places = people_places(start)

    acting = set(scene.get("performances") or {}) | {line["speakerId"] for line in scene.get("lines") or []}
    described: set[str] = set()

    def person(cid: str) -> str:
        # Someone who acts or speaks and whom the shot reveals (entering, or their
        # face hidden in the start frame but shown in the ``end`` frame) gets their
        # whole look, once: from the body words alone the video model invents a
        # stranger when they turn or enter. Anyone else keeps what the frame shows:
        # the face words of someone seen from behind make the model turn them, and
        # the whole look of a sliver at the edge makes it reframe to show them.
        minimum = settings()["onScreenMinPixels"]
        revealed = not in_shot(start, cid) or (end is not None and visible(end, cid, ["face"], minimum))
        known = cid not in acting or cid in described or visible(start, cid, ["face"], minimum) or not revealed
        described.add(cid)
        body = clause(show["characters"][cid]["body"] if known else character_appearance_text(show["characters"][cid]))
        if cid not in places:
            return body
        place = places[cid][:1].upper() + places[cid][1:]
        return template("scenePerson", {"place": place, "body": body})

    def action(performance: dict) -> str:
        text = clause(performance["action"])
        if performance.get("target"):
            text = text.replace("{target}", target_words(show, scene, start, performance["target"]))
        return text

    parts: list[str] = line_sentences(show, scene, person)
    performances = scene.get("performances") or {}
    for cid, performance in performances.items():
        parts.append(template("scenePerformance", {"person": person(cid), "action": action(performance)}))
        if performance.get("expression"):
            parts.append(template("scenePerformanceExpression", {"expression": clause(performance["expression"])}))
    # Anyone else the start frame shows large enough holds still, so the video model invents no motion for them.
    for cid in places:
        if cid not in performances and start["entities"][cid]["pixels"][0] >= STILL_MIN_SHARE * WIDTH * HEIGHT:
            parts.append(template("scenePerformance", {"person": person(cid), "action": clause(settings()["holdStill"])}))
    if scene.get("motion"):
        parts.append(template("sceneMotion", {"motion": clause(scene["motion"])}))
    effects = list(dict.fromkeys(clause(effect["appearance"]) for effect in start.get("effects") or []))
    if effects:
        parts.append(template("sceneEffects", {"effects": "; ".join(effects)}))
    parts.append(camera_sentence(scene))
    # The prompt names only what the start frame has: telling the model to keep
    # "its people" in an empty landscape makes it put people there.
    props = any(entry["kind"] == "prop" and in_shot(start, entity_id) for entity_id, entry in start["entities"].items())
    keep = ("people, wardrobe, " if places else "") + ("props, " if props else "")
    visual = template("sceneVideo", {"keep": keep, "action": " ".join(parts)})
    return f"{visual} {sound_sentence(show, scene)}"
