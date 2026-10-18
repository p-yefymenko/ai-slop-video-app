"""The words for Wan, taken from the script and the start frame's observation.

Only what holds whatever part of anyone or anything the frame shows, and nothing
the start still or the depth video already shows (no shot size or angle): the camera's move, everyone with a performance by their ``body``
with their action and expression, the effects, sky, ground, and surround that
are in the frame, the scene's motion, and the light. No clothes, no props or
landmarks, no screen side, no facing: words cannot know which part of a person
or thing the frame shows, and a described part the frame does not show is
drawn anyway. Looks come from the reference sheet; shapes from the depth video.
Every sentence is a template in ``prompts.json``. Nothing here knows any one scene.

Lines are never in the picture prompt: told among the actions, a line from off
screen makes the model cut to the speaker. A talking take says only how its
speaker speaks. Character names and ids never reach a model.

The start still (``layout_prompt``) is drawn by an image model from the shot's 3D
layout in the meshes' own colors: the looks come from the layout, and the words
add only the faces' expressions and the setting.
"""

from __future__ import annotations

import re

from observe import in_shot
from world import attribute_text, backdrop_text, camera_moves, clause, settings

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


def shot_sentence(scene: dict, framing: bool = True) -> str:
    """The kind of shot: its size and angle, from the script's shot, for a still drawn from words
    alone. With ``framing`` False, only ``footage``: no size, no angle, and no shot type, whose
    words are sizes too ("action" is a wide shot)."""
    if not framing:
        return template("footage", {})
    shot = scene.get("shot") or {}
    words = settings()["shot"]
    angle = words["angles"].get(shot.get("angle", "eye"), "")
    size = words["sizes"].get(shot.get("size", ""), words["types"].get(shot.get("type", ""), words["sizes"]["wide"]))
    text = template("shotSentence", {"angle": f"{angle} " if angle else "", "size": size})
    return text[:1].upper() + text[1:]


def camera_sentence(scene: dict) -> str:
    """How the camera moves, in the script's words; else held by hand."""
    if scene.get("cameraMotion") and camera_moves(scene):
        return template("cameraMotion", {"motion": clause(scene["cameraMotion"])})
    return template("cameraHeld", {})


def _action(show: dict, performance: dict) -> str:
    """The action, with ``{target}`` (always a person in the start frame) named by their body."""
    text = clause(performance["action"])
    if performance.get("target"):
        text = text.replace("{target}", "the " + _ARTICLE.sub("", clause(show["characters"][performance["target"]]["body"])))
    return text


def person_sentences(show: dict, scene: dict, start: dict, cid: str) -> list[str]:
    """One person: their body, action, and expression; their body alone when they have no performance.
    Someone with a performance who is not in the start frame comes into it."""
    body = _a(clause(show["characters"][cid]["body"]))
    performance = (scene.get("performances") or {}).get(cid)
    if performance is None:
        return [template("personPresent", {"body": body})]
    sentences = [template("person" if in_shot(start, cid) else "personEnters", {"body": body, "action": _action(show, performance)})]
    if performance.get("expression"):
        sentences.append(template("expression", {"expression": clause(performance["expression"])}))
    return sentences


def people_sentences(show: dict, scene: dict, start: dict) -> list[str]:
    """Everyone with a performance, in the script's order: their body, action, and expression.

    Nobody else is named, and nobody gets more than ``body`` (how the prompt points at a person).
    """
    return [sentence for cid in (scene.get("performances") or {}) for sentence in person_sentences(show, scene, start, cid)]


def setting_sentences(show: dict, scene: dict, start: dict) -> list[str]:
    """The effects in shot, each stretch of the backdrop in shot, the scene's motion, then the light."""
    sentences: list[str] = []
    effects = list(dict.fromkeys(clause(effect["appearance"]) for effect in start.get("effects") or []))
    if effects:
        sentences.append(template("effects", {"effects": "; ".join(effects)}))
    backdrop = settings()["backdrop"]
    for entity_id, entry in start["entities"].items():
        if entry["kind"] == "backdrop" and in_shot(start, entity_id):
            sentences.append(backdrop[entity_id].format(text=backdrop_text(show, scene, entity_id)))
    if scene.get("motion"):
        sentences.append(_sentence(scene["motion"]))
    sentences.append(template("light", {"light": clause(show["locations"][scene["locationId"]]["look"]["light"])}))
    return sentences


def _joined(parts: list[str]) -> str:
    parts = [" ".join(part.split()) for part in parts if part]
    return " ".join(part[:1].upper() + part[1:] for part in parts)


def speech_sentence(scene: dict, speaker: str) -> str:
    """How the speaker of a talking take speaks: their lines' deliveries, never the words."""
    deliveries = list(dict.fromkeys(clause(line["delivery"]) for line in scene.get("lines") or [] if line["speakerId"] == speaker))
    return template("speaking", {"delivery": "; then ".join(deliveries)})


def video_prompt(show: dict, episode: dict, scene: dict, start: dict, speaker: str | None = None) -> str:
    """The take in words, for VACE (``speaker`` None) or for a talking take (``speaker`` is the one whose voice drives it).

    No shot size, angle or type: the start still or the depth video shows the framing exactly, and
    the script's words for it can disagree with its camera (a shot of type ``action`` is called a
    wide shot whatever the camera frames)."""
    parts = [template("footage", {}), camera_sentence(scene), *people_sentences(show, scene, start)]
    if speaker is not None:
        parts.append(speech_sentence(scene, speaker))
    parts += setting_sentences(show, scene, start)
    return _joined(parts)


def still_prompt(show: dict, scene: dict, start: dict, framing: bool) -> str:
    """A still's shot and setting, naming no one: the layout's words say who is in it. The shot's
    size and angle only with ``framing``; never over whose shoulder (told it, a model drew a head to
    go with the shoulder)."""
    return _joined([shot_sentence(scene, framing=framing), *setting_sentences(show, scene, start)])


def layout_prompt(show: dict, scene: dict, start: dict, views: dict[str, float], size: tuple[int, int]) -> str:
    """The words for a still an image model draws from the shot's 3D layout in the meshes' own
    colors: how to read it (``layoutPrompt``), what the frame shows of each person only partly in it
    whose colors leave it a guess (``layoutPartly``: their attributes listed in ``shot.partlySeen``,
    authored, tied to no one), the face of each performer in it whose face is toward the camera
    (``views[cid]``, how far round from their front it stands, under 90°), named by their ``body``
    (``layoutFace``), then the still's setting (``still_prompt``), and ``frameOnly``.

    No looks in words beyond ``partlySeen``: any word for a part the frame does not show got that
    part drawn, and code cannot know which part of a person the frame shows. Given a
    near shoulder's owner's whole look, the model drew their head and reframed the shot to fit it;
    given everyone's wear as one list tied to no one, it drew the shoulder's owner whole, boots and
    all, and pulled back to full length. Without the words, the colors alone leave a material to
    guess (a fox-fur shoulder came out a sack). No photos: given
    full-length photos beside the layout, the model drew everyone full length, facing the camera.
    No size or angle: the layout shows the framing exactly, so words for it can only repeat or
    contradict it (told "wide shot" over a layout cut at the knees, the model drew everyone full
    length). No side of the frame is named."""
    performances = scene.get("performances") or {}
    partly = (scene.get("shot") or {}).get("partlySeen") or {}
    shown = [template("layoutPartly", {"wear": "; ".join(clause(item) for item in partly[cid])}) for cid in views if partly.get(cid)]
    faces = [template("layoutFace", {"body": _ARTICLE.sub("", clause(show["characters"][cid]["body"])),
                                     "expression": clause(performances[cid]["expression"])})
             for cid in views if (performances.get(cid) or {}).get("expression") and views[cid] < 90.0]
    return _joined([template("layoutPrompt", {"width": str(size[0]), "height": str(size[1])}),
                    *shown, *faces,
                    still_prompt(show, scene, start, framing=False), template("frameOnly", {})])


def plain_still_prompt(show: dict, scene: dict, start: dict, size: tuple[int, int]) -> str:
    """The words for a still with none of the 3D scene in its frame (only sky, ground and surround,
    which the layout has no shape for): the layout would be a blank, so the words go alone, with
    the shot's size and angle, unless the camera moves: then they are the whole shot's, not its
    first frame's (a tilt up from the boots is a medium close-up only at its end)."""
    return _joined([template("plainStill", {"width": str(size[0]), "height": str(size[1])}),
                    still_prompt(show, scene, start, framing=not camera_moves(scene))])


def talking_speaker(scene: dict) -> str | None:
    """Who a talking take (S2V) is about: the first speaker with a performance, i.e. on screen.

    A speaker without a performance is a voice from off screen, laid over the picture.
    """
    performances = scene.get("performances") or {}
    return next((line["speakerId"] for line in scene.get("lines") or [] if line["speakerId"] in performances), None)


def reference_people(show: dict, scene: dict) -> list[str]:
    """Whose reference photos the take gets: the shot's ``subjects``, exactly as authored.

    VACE paints a face for everyone on the sheet, so anyone whose face the shot
    does not show (a shoulder in front) is left to the depth video.
    """
    return [cid for cid in dict.fromkeys((scene.get("shot") or {}).get("subjects") or []) if cid in show["characters"]]


def reference_prompt(character: dict) -> str:
    """A character's reference photo: the whole person on pure white, from their words alone."""
    return template("referencePhoto", {"body": clause(character["body"]), "attributes": "; ".join(clause(attribute_text(item)) for item in character["attributes"])})
