"""Words for Qwen and LTX, taken from the script and an observation.

Every entity is described the same way: each of its descriptions is sent when
any region it depends on is visible (``observe.visible``). People, landmarks,
props, and the sky/ground/surround differ only in where their sentence goes.
Character names and ids never reach a model.
"""

from __future__ import annotations

import re

from observe import visible
from render import WIDTH
from world import camera_moves, character_appearance_text, clause, descriptions, settings

PLACEHOLDER = re.compile(r"\{([a-zA-Z][a-zA-Z0-9]*)\}")
COUNT_WORDS = ("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten")
PICTURE_LEGEND = {
    "Depth": "Picture {n} is depth: brighter is closer, black is empty space.",
    "Edges": "Picture {n} is outlines.",
    "Composite": (
        "Picture {n} is the colors of everything in the shot; the people and objects in it that are "
        "already painted are finished and stay exactly as they are."
    ),
    "OwnColor": "Picture {n} is its colors.",
    "Appearance": "Picture {n} is how it looks: the same face, hair, skin, clothes, colors, and materials.",
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


def entity_text(show: dict, scene: dict, observation: dict, entity_id: str) -> tuple[list[str], list[dict]]:
    """Visible descriptions of one entity, and a log line for each description."""
    entry = observation["entities"][entity_id]
    sent: list[str] = []
    log: list[dict] = []
    for text, regions in descriptions(show, scene, entry["kind"], entity_id):
        shown = visible(observation, entity_id, regions)
        record = {"text": text, "sent": shown}
        if not shown:
            record["regions"] = {name: entry["regions"].get(name) for name in regions}
        log.append(record)
        if shown:
            sent.append(text)
    return sent, log


def _side(screen_x: float) -> str:
    if screen_x < WIDTH * 0.34:
        return "On the left"
    if screen_x > WIDTH * 0.66:
        return "On the right"
    return "In the center"


def _row_slot(index: int, count: int) -> str:
    if count == 1 or (count == 3 and index == 1):
        return "center"
    if index == 0:
        return "far left"
    if index == count - 1:
        return "far right"
    return "left of center" if index == 1 else "right of center" if index == count - 2 else "center"


def people_places(observation: dict) -> dict[str, str]:
    """characterId -> where the person stands in this frame, in words.

    Figures deeper than the nearest person times ``rowDepthRatio`` stand in a row
    behind when at least two qualify; everyone else is named by screen side.
    """
    people = [
        (entity_id, entry)
        for entity_id, entry in observation["entities"].items()
        if entry["kind"] == "character" and visible(observation, entity_id)
    ]
    if not people:
        return {}
    nearest = min(entry["depth"] for _id, entry in people)
    behind = sorted(
        [(entity_id, entry) for entity_id, entry in people if entry["depth"] > nearest * settings()["rowDepthRatio"]],
        key=lambda item: item[1]["screenX"],
    )
    places: dict[str, str] = {}
    if len(behind) >= 2:
        for index, (entity_id, _entry) in enumerate(behind):
            places[entity_id] = f"in the row behind, {_row_slot(index, len(behind))}"
    for entity_id, entry in people:
        if entity_id not in places:
            places[entity_id] = _side(entry["screenX"]) + (", in the foreground" if len(behind) >= 2 else "")
    return places


def thing_sentences(show: dict, scene: dict, observation: dict, kinds: tuple[str, ...]) -> tuple[str, list[dict]]:
    """Visible appearances of these kinds of entity, each said once."""
    said: list[str] = []
    log: list[dict] = []
    for entity_id, entry in observation["entities"].items():
        if entry["kind"] not in kinds:
            continue
        sent, records = entity_text(show, scene, observation, entity_id)
        log.append({"id": entity_id, "kind": entry["kind"], "pixels": entry["pixels"], "descriptions": records})
        for text in sent:
            sentence = _sentence(text)
            if sentence and sentence not in said:
                said.append(sentence)
    return " ".join(said), log


def _legend(pictures: list[str]) -> str:
    return " ".join(PICTURE_LEGEND[title].format(n=number) for number, title in enumerate(pictures, start=1))


def drawn_prompt(show: dict, scene: dict, kind: str, entity_id: str, pictures: list[str]) -> str:
    """One person or prop drawn whole and alone, from its own guides and only its own words.

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
        words = descriptions(show, scene, kind, entity_id)[0][:1]
    parts = [
        clause(settings()["stillOpening"]) + ".",
        _legend(pictures),
        f"Keep the shape, pose, and turn from the pictures. {subject}, on a plain light gray background.",
        _sentence(", ".join(words)),
    ]
    return " ".join(part for part in parts if part)


def still_prompt(show: dict, scene: dict, observation: dict, pictures: list[str]) -> tuple[str, dict]:
    """The shot around its already painted people and props: legend, keep, count, landmarks, setting."""
    people = len(people_places(observation))
    things, things_log = thing_sentences(show, scene, observation, ("landmark",))
    setting, setting_log = thing_sentences(show, scene, observation, ("backdrop",))
    count = COUNT_WORDS[people] if people < len(COUNT_WORDS) else str(people)
    parts = [
        clause(settings()["stillOpening"]) + ".",
        _legend(pictures),
        "Keep the shape, position, and occlusion from the pictures.",
        f"{count} {'person' if people == 1 else 'people'}, already painted." if people else "",
        things,
        f"Behind and around: {setting}" if setting else "",
    ]
    return " ".join(part for part in parts if part), {"people": people, "things": things_log, "setting": setting_log}


def sound_sentence(show: dict, scene: dict) -> str:
    """Location bed and space, the scene's events, and the music policy."""
    soundscape = show["locations"][scene["locationId"]]["soundscape"]
    sound = scene["sound"]
    events = "; ".join(clause(event["text"]) for event in sound["events"])
    music = sound["music"]
    return template(
        "sceneSound",
        {
            "label": template("sceneSoundLabel", {}),
            "bedPhrase": template(
                "sceneSoundBedPresent" if sound["bed"] == "present" else "sceneSoundBedFaint",
                {"ambience": _sentence(soundscape["ambience"])[:-1]},
            ),
            "spacePhrase": template("sceneSoundSpace", {"space": _sentence(soundscape["space"])[:-1]}),
            "eventsPhrase": template("sceneSoundEvents", {"events": _sentence(events)[:-1]}),
            "musicPhrase": template("sceneSoundMusicNone", {})
            if music["kind"] == "none"
            else template("sceneSoundMusicDescribed", {"description": _sentence(music["description"])[:-1]}),
        },
    )


def ltx_prompt(show: dict, scene: dict, start: dict) -> str:
    """Dialogue, performances, motion, a locked camera, then sound.

    People are named by where the start frame shows them, plus their body, so
    LTX can find them in the first frame.
    """
    places = people_places(start)

    def person(cid: str) -> str:
        body = clause(show["characters"][cid]["body"])
        if cid not in places:
            return body
        place = places[cid][:1].upper() + places[cid][1:]
        return template("scenePerson", {"place": place, "body": body})

    parts: list[str] = []
    if scene.get("dialogue"):
        parts.append(
            template(
                "sceneDialogue",
                {
                    "person": person(scene["speakerId"]),
                    "delivery": clause(scene["dialogue"]["delivery"]),
                    "line": " ".join(scene["dialogue"]["line"].split()),
                },
            )
        )
    for cid, performance in (scene.get("performances") or {}).items():
        parts.append(template("scenePerformance", {"person": person(cid), "action": clause(performance["action"])}))
        if performance.get("expression"):
            parts.append(template("scenePerformanceExpression", {"expression": clause(performance["expression"])}))
    if scene.get("motion"):
        parts.append(template("sceneMotion", {"motion": clause(scene["motion"])}))
    if not camera_moves(scene):
        parts.append(template("sceneCameraLocked", {}))
    visual = template("sceneVideo", {"action": " ".join(parts)})
    return f"{visual} {sound_sentence(show, scene)}"
