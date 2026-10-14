"""The words for Wan, taken from the script and the start frame's observation.

Who is said, and how, comes from the script alone: the shot's ``over`` is the
shoulder in the foreground, and everyone with a performance is said with their
whole look and their action. Nobody else is named. The 3D scene adds only exact
geometry: where on screen a person is, how they are turned to the camera, and
which stretches of sky, ground, and surround the frame shows. Every sentence is
a template in ``prompts.json``. Nothing here knows any one scene.

Lines are never in the picture prompt: told among the actions, a line from off
screen makes the model cut to the speaker. A talking take (S2V) says only how its
speaker speaks. Character names and ids never reach a model.
"""

from __future__ import annotations

import math
import re

from observe import in_shot
from render import WIDTH
from world import camera_moves, character_appearance_text, clause, descriptions, settings

PLACEHOLDER = re.compile(r"\{([a-zA-Z][a-zA-Z0-9]*)\}")
_ARTICLE = re.compile(r"^(one|an?|the)\s+", re.IGNORECASE)


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


def _a(text: str) -> str:
    """``old man, tall`` -> ``an old man, tall``: a body's words as a noun phrase."""
    return text if _ARTICLE.match(text) else ("an " if text[:1].lower() in "aeiou" else "a ") + text


def _side(screen_x: float) -> str:
    """The third of the frame a person's pixels are centered in."""
    if screen_x < WIDTH / 3:
        return "on the left"
    if screen_x > WIDTH * 2 / 3:
        return "on the right"
    return "in the center"


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
    return settings()["facing"][key].format(side=side)


def view_words(camera: dict) -> str:
    """A camera looking steeply down or up, in words; empty when it looks roughly level."""
    position, look = camera["position"], camera["lookAt"]
    pitch = math.degrees(math.atan2(look[2] - position[2], math.hypot(look[0] - position[0], look[1] - position[1])))
    return settings()["view"]["high"] if pitch < -25 else settings()["view"]["low"] if pitch > 12 else ""


def shot_sentence(scene: dict, observation: dict) -> str:
    """The kind of shot: its size, angle, and an over-the-shoulder, from the script's shot."""
    shot = scene.get("shot") or {}
    words = settings()["shot"]
    angle = words["angles"].get(shot.get("angle", "eye"), "")
    size = words["sizes"].get(shot.get("size", ""), words["types"].get(shot.get("type", ""), words["sizes"]["wide"]))
    over = words["over"] if shot.get("type") == "overShoulder" else ""
    text = template("shotSentence", {"angle": f"{angle} " if angle else "", "size": size, "over": over})
    view = view_words(observation["camera"]) if shot.get("type") in ("establishing", "action") else ""
    return " ".join(part for part in (text[:1].upper() + text[1:], view) if part)


def camera_sentence(scene: dict) -> str:
    """How the camera moves: an own camera's words, a shot's move, else held by hand."""
    if scene.get("cameraMotion"):
        return template("cameraMotion", {"motion": clause(scene["cameraMotion"])})
    move = (scene.get("shot") or {}).get("move", "static")
    if move != "static" and camera_moves(scene):
        return settings()["cameraMoves"][move]
    return template("cameraHeld", {})


def target_words(show: dict, scene: dict, start: dict, target: str) -> str:
    """What an action is aimed at, as the start frame shows it: a person by body and place, a thing
    by its effects when the frame shows them, else by its first words. The script checks keep every
    target in the start frame, so these words always point at something the model can find there."""
    if target in show["characters"]:
        body = clause(show["characters"][target]["body"])
        entry = start["entities"].get(target)
        return f"the {body}, {_side(entry['screenX'])}" if entry and in_shot(start, target) else f"the {body}"
    effects = [clause(effect["appearance"]) for effect in start.get("effects") or [] if effect["id"] == target]
    if effects:
        return "the " + _ARTICLE.sub("", effects[0])
    kind = "prop" if target in (show.get("props") or {}) else "landmark"
    return "the " + _ARTICLE.sub("", re.split(r"[,.;:]", descriptions(show, scene, kind, target)[0][0])[0].strip())


def _action(show: dict, scene: dict, start: dict, performance: dict) -> str:
    text = clause(performance["action"])
    if performance.get("target"):
        text = text.replace("{target}", target_words(show, scene, start, performance["target"]))
    return text


def people_sentences(show: dict, episode: dict, scene: dict, start: dict) -> list[str]:
    """The shoulder in front (the shot's ``over``) and everyone with a performance, nearest first.

    Someone with a performance who is not in the start frame comes into it.
    """
    from shots import _people

    standing = _people(show, episode, scene)
    performances = scene.get("performances") or {}
    over = (scene.get("shot") or {}).get("over")
    framed = sorted(
        (cid for cid, entry in start["entities"].items() if entry["kind"] == "character" and in_shot(start, cid)),
        key=lambda cid: start["entities"][cid]["depth"],
    )
    sentences: list[str] = []
    for cid in framed + [cid for cid in performances if cid not in framed]:
        if cid != over and cid not in performances:
            continue
        look = _a(clause(character_appearance_text(show["characters"][cid])))
        if cid not in framed:
            sentences.append(template("personEnters", {"look": look}))
        elif cid == over:
            sentences.append(template("personNear", {"place": _side(start["entities"][cid]["screenX"]), "look": look}))
            continue
        else:
            here = standing.get(cid)
            facing = f", {facing_words(here['gaze'], (here['x'], here['y']), start['camera'])}" if here else ""
            sentences.append(template("personFace", {"place": _side(start["entities"][cid]["screenX"]), "look": look, "facing": facing}))
        performance = performances[cid]
        sentences.append(_sentence(_action(show, scene, start, performance)))
        if performance.get("expression"):
            sentences.append(template("expression", {"expression": clause(performance["expression"])}))
    return sentences


def setting_sentences(show: dict, scene: dict, start: dict) -> list[str]:
    """Landmarks and props in shot, effects, then each stretch of the backdrop in shot by where it is, then the light."""
    sentences: list[str] = []
    for entity_id, entry in start["entities"].items():
        if entry["kind"] in ("landmark", "prop") and in_shot(start, entity_id):
            sentences.append(_sentence(descriptions(show, scene, entry["kind"], entity_id)[0][0]))
    effects = list(dict.fromkeys(clause(effect["appearance"]) for effect in start.get("effects") or []))
    if effects:
        sentences.append(template("effects", {"effects": "; ".join(effects)}))
    backdrop = settings()["backdrop"]
    for entity_id, entry in start["entities"].items():
        if entry["kind"] == "backdrop" and in_shot(start, entity_id):
            sentences.append(backdrop[entity_id].format(text=clause(descriptions(show, scene, "backdrop", entity_id)[0][0])))
    if scene.get("motion"):
        sentences.append(_sentence(scene["motion"]))
    sentences.append(template("light", {"light": clause(show["locations"][scene["locationId"]]["look"]["light"])}))
    return sentences


def speech_sentence(scene: dict, speaker: str) -> str:
    """How the speaker of a talking take speaks: their lines' deliveries, never the words."""
    deliveries = list(dict.fromkeys(clause(line["delivery"]) for line in scene.get("lines") or [] if line["speakerId"] == speaker))
    return template("speaking", {"delivery": "; then ".join(deliveries)})


def video_prompt(show: dict, episode: dict, scene: dict, start: dict, speaker: str | None = None) -> str:
    """The take in words, for VACE (``speaker`` None) or for S2V (``speaker`` is the one whose voice drives it)."""
    parts = [shot_sentence(scene, start), camera_sentence(scene), *people_sentences(show, episode, scene, start)]
    if speaker is not None:
        parts.append(speech_sentence(scene, speaker))
    parts += setting_sentences(show, scene, start)
    parts = [" ".join(part.split()) for part in parts if part]
    return " ".join(part[:1].upper() + part[1:] for part in parts)


def talking_speaker(scene: dict) -> str | None:
    """Who a talking take (S2V) is about: the first speaker with a performance, i.e. on screen.

    A speaker without a performance is a voice from off screen, laid over the picture.
    """
    performances = scene.get("performances") or {}
    return next((line["speakerId"] for line in scene.get("lines") or [] if line["speakerId"] in performances), None)


def reference_people(show: dict, scene: dict) -> list[str]:
    """Whose reference photos the take gets: the shot's subjects, else everyone with a performance, never the shoulder in front.

    VACE paints a face for everyone on the sheet, so the shoulder in front (``over``)
    is left to the words.
    """
    shot = scene.get("shot") or {}
    chosen = shot.get("subjects") or list(scene.get("performances") or {})
    return [cid for cid in dict.fromkeys(chosen) if cid in show["characters"] and cid != shot.get("over")]


def reference_prompt(character: dict) -> str:
    """A character's reference photo: the whole person on pure white, from their words alone."""
    return template("referencePhoto", {"body": clause(character["body"]), "attributes": "; ".join(clause(item) for item in character["attributes"])})
