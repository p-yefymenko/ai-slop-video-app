"""The words for Wan, taken from the script and the start frame's observation.

Only what holds whatever part of anyone or anything the frame shows: the shot's
size and angle, the camera's move, everyone with a performance by their ``body``
with their action and expression, the effects, sky, ground, and surround that
are in the frame, the scene's motion, and the light. No clothes, no props or
landmarks, no screen side, no facing: words cannot know which part of a person
or thing the frame shows, and a described part the frame does not show is
drawn anyway. Looks come from the reference sheet; shapes from the depth video.
Every sentence is a template in ``prompts.json``. Nothing here knows any one scene.

Lines are never in the picture prompt: told among the actions, a line from off
screen makes the model cut to the speaker. A talking take (S2V) says only how its
speaker speaks. Character names and ids never reach a model.
"""

from __future__ import annotations

import re

from observe import in_shot
from world import attribute_text, backdrop_text, camera_moves, clause, is_face, settings

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


def shot_sentence(scene: dict, over_shoulder: bool = True) -> str:
    """The kind of shot: its size and angle, from the script's shot; an over-the-shoulder shot says
    so (``overShoulder``, unless ``over_shoulder`` is False), naming nobody. Named by their body,
    the person it looks past was given the subject's action (a hand reached up from a blurred
    shoulder)."""
    shot = scene.get("shot") or {}
    words = settings()["shot"]
    angle = words["angles"].get(shot.get("angle", "eye"), "")
    size = words["sizes"].get(shot.get("size", ""), words["types"].get(shot.get("type", ""), words["sizes"]["wide"]))
    if over_shoulder and shot.get("type") == "overShoulder":
        size = template("overShoulder", {"size": size})
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


def _view_index(view_degrees: float) -> int:
    """The nearest of the ``views``, 45° apart, to ``view_degrees`` between where a person faces and
    where the camera is (0 their front, 180 their back)."""
    count = len(settings()["views"])
    return round(min(max(view_degrees, 0.0), 180.0) / (180.0 / (count - 1)))


def view_words(view_degrees: float) -> str:
    """Which side of a person the camera sees, in words: the depth alone does not tell a back from a
    front, and the model drew everyone from the front. The words name only what is seen: "the face
    not visible" put a face in the picture."""
    return settings()["views"][_view_index(view_degrees)]


def alone_negative(view_degrees: float) -> str:
    """The negative for a person drawn alone: ``aloneNegative``, and what the side they are seen
    from must not show (``viewNegatives``: a face, from behind)."""
    extra = settings()["viewNegatives"][_view_index(view_degrees)]
    return ", ".join(part for part in (settings()["aloneNegative"], extra) if part)


def alone_prompt(show: dict, scene: dict, start: dict, cid: str, view_degrees: float) -> str:
    """One person drawn alone and full height by Qwen-Image, for a still: which side of them the
    camera sees (``view_words``), every attribute, as their reference photo was drawn (the whole
    figure is drawn, so every described part is in the picture), save the face's when it is turned
    from the camera (named, the model turned the head to show a beard and eyes), their expression
    when their face is toward the camera, and the light. No setting: words for a backdrop made the model give it
    room and cut the person. No action: it is motion the take performs, the pose is the depth's,
    and an action's ``{target}`` is another person, whom the model then drew beside them."""
    character = show["characters"][cid]
    facing = view_degrees < 90.0
    light = template("light", {"light": clause(show["locations"][scene["locationId"]]["look"]["light"])})
    seen = [attribute_text(item) for item in character["attributes"] if facing or not is_face(item)]
    opening = template("aloneImage", {"body": clause(character["body"]), "view": view_words(view_degrees),
                                      "attributes": "; ".join(clause(item) for item in seen)})
    performance = (scene.get("performances") or {}).get(cid) or {}
    face = [template("expression", {"expression": clause(performance["expression"])})] if performance.get("expression") and facing else []
    return _joined([opening, *face, light])


def speech_sentence(scene: dict, speaker: str) -> str:
    """How the speaker of a talking take speaks: their lines' deliveries, never the words."""
    deliveries = list(dict.fromkeys(clause(line["delivery"]) for line in scene.get("lines") or [] if line["speakerId"] == speaker))
    return template("speaking", {"delivery": "; then ".join(deliveries)})


def video_prompt(show: dict, episode: dict, scene: dict, start: dict, speaker: str | None = None) -> str:
    """The take in words, for VACE (``speaker`` None) or for S2V (``speaker`` is the one whose voice drives it)."""
    parts = [shot_sentence(scene), camera_sentence(scene), *people_sentences(show, scene, start)]
    if speaker is not None:
        parts.append(speech_sentence(scene, speaker))
    parts += setting_sentences(show, scene, start)
    return _joined(parts)


def still_prompt(show: dict, scene: dict, start: dict) -> str:
    """The words for a still's final pass, which keeps the people already drawn and draws only what
    is around them: the shot's size and angle and the setting, and no one. Told the people (an
    action's ``{target}`` named by body), it drew a person into the free frame; told it was over
    someone's shoulder, a head to go with the shoulder."""
    return _joined([shot_sentence(scene, over_shoulder=False), *setting_sentences(show, scene, start)])


def sheet_prompt(show: dict, scene: dict, start: dict, colors: dict[str, str], facing: dict[str, bool], size: tuple[int, int]) -> str:
    """The words that go with a scene's sheet (``shot_inputs.prompt_sheet``) to an image model that
    takes pictures: how to read the sheet (``sheetPrompt``: the layout, and each person's tint
    ``colors[cid]`` matched to the photo framed in it), the face of each performer whose face is
    toward the camera (``facing``), by their color, then the still's own words, the shot and the
    setting (``still_prompt``). Looks are the photos'; no side of the frame is named."""
    performances = scene.get("performances") or {}
    faces = [template("sheetFace", {"color": colors[cid], "expression": clause(performances[cid]["expression"])})
             for cid in colors if (performances.get(cid) or {}).get("expression") and facing[cid]]
    return _joined([template("sheetPrompt", {"width": str(size[0]), "height": str(size[1])}), *faces,
                    still_prompt(show, scene, start), template("sheetOnly", {})])


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
