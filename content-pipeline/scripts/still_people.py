"""People sentence for a still. Words come from body, tagged attributes, and blocking."""

from __future__ import annotations

import json
from pathlib import Path

from body_parts import BODY_PARTS, _collapse, empty_part_stats

BODY_VISIBLE_MIN = 0.3
COUNT_WORDS = (
    "Zero",
    "One",
    "Two",
    "Three",
    "Four",
    "Five",
    "Six",
    "Seven",
    "Eight",
    "Nine",
    "Ten",
)
PROMPTS_PATH = Path(__file__).resolve().parents[1] / "prompts.json"


def _count_word(count: int) -> str:
    if 0 <= count < len(COUNT_WORDS) and COUNT_WORDS[count]:
        return COUNT_WORDS[count]
    return str(count)


def _height(person: dict) -> int:
    return int(person.get("pixel_height") or 0)


def _side(screen_x: float, width: float) -> str:
    if screen_x < width * 0.34:
        return "left"
    if screen_x > width * 0.66:
        return "right"
    return "center"


def _row_slot(index: int, count: int) -> str:
    if count <= 1:
        return "center"
    if index <= 0:
        return "far left"
    if index >= count - 1:
        return "far right"
    if count == 3 and index == 1:
        return "center"
    if index == 1:
        return "left of center"
    if index == count - 2:
        return "right of center"
    return "center"


def _join(bits: list[str]) -> str:
    cleaned = [bit.strip(" .") for bit in bits if bit and bit.strip(" .")]
    if not cleaned:
        return ""
    return ". ".join(cleaned) + "."


def character_appearance_text(character: dict) -> str:
    """Plate and portrait text: body, then every attribute."""
    items = [_collapse(character.get("body"))]
    for attribute in character.get("attributes") or []:
        if not isinstance(attribute, dict):
            continue
        items.append(_collapse(attribute.get("text")))
    return ", ".join(item for item in items if item)


def _load_attributes(character: dict) -> list[dict]:
    loaded: list[dict] = []
    for attribute in character.get("attributes") or []:
        if not isinstance(attribute, dict):
            continue
        text = _collapse(attribute.get("text"))
        parts = [str(part) for part in (attribute.get("parts") or []) if part in BODY_PARTS]
        if text and parts:
            loaded.append({"text": text, "parts": parts})
    return loaded


def _shot_label(scene: dict, time_seconds: float, shot_label: str | None) -> str:
    if shot_label in ("start", "end"):
        return shot_label
    times = scene.get("timeRangeSeconds") or [0, 0]
    start, end = float(times[0]), float(times[1])
    if start != end and float(time_seconds) == end:
        return "end"
    return "start"


def load_row_depth_ratio(prompts: dict | None = None) -> float:
    data = prompts if isinstance(prompts, dict) else json.loads(
        PROMPTS_PATH.read_text(encoding="utf-8")
    )
    value = data.get("rowDepthRatio") if isinstance(data, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError("prompts.rowDepthRatio must be a number")
    ratio = float(value)
    if ratio <= 1:
        raise RuntimeError("prompts.rowDepthRatio must be greater than 1")
    return ratio


def load_part_min_pixel_height(prompts: dict | None = None) -> int:
    data = prompts if isinstance(prompts, dict) else json.loads(
        PROMPTS_PATH.read_text(encoding="utf-8")
    )
    value = data.get("partMinPixelHeight") if isinstance(data, dict) else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RuntimeError("prompts.partMinPixelHeight must be a positive integer")
    return value


def _coerce_part_stat(raw: object) -> dict[str, int]:
    if isinstance(raw, dict):
        try:
            pixels = int(raw.get("pixels") or 0)
        except (TypeError, ValueError):
            pixels = 0
        try:
            width = int(raw.get("width") or 0)
        except (TypeError, ValueError):
            width = 0
        try:
            height = int(raw.get("height") or 0)
        except (TypeError, ValueError):
            height = 0
        return {"pixels": pixels, "width": width, "height": height}
    try:
        pixels = int(raw or 0)
    except (TypeError, ValueError):
        pixels = 0
    return {"pixels": pixels, "width": 0, "height": 0}


def _part_stats(person: dict) -> dict[str, dict[str, int]]:
    raw = person.get("part_stats")
    if not isinstance(raw, dict):
        raw = person.get("part_pixels")
    stats = empty_part_stats()
    if isinstance(raw, dict):
        for name in BODY_PARTS:
            stats[name] = _coerce_part_stat(raw.get(name))
    return stats


def _part_size_line(stats: dict[str, dict[str, int]]) -> str:
    return " ".join(f"{name} {stats[name]['height']}px" for name in BODY_PARTS)


def _attribute_clause(
    attribute: dict,
    stats: dict[str, dict[str, int]],
    min_height: int,
    character_id: str | None,
) -> dict:
    text = _collapse(attribute.get("text"))
    parts = [str(part) for part in (attribute.get("parts") or [])]
    tall = [part for part in parts if stats.get(part, {}).get("height", 0) >= min_height]
    record: dict = {"text": text, "parts": parts, "sent": bool(tall)}
    if tall:
        part = tall[0]
        record["part"] = part
        record["partHeight"] = int(stats[part]["height"])
        if character_id:
            print(
                f"  {character_id} attribute {text!r} sent ({part} {stats[part]['height']}px)",
                flush=True,
            )
        return record
    reasons: list[str] = []
    for part in parts:
        height = int(stats.get(part, {}).get("height") or 0)
        if height <= 0:
            reasons.append(f"{part} not visible")
        else:
            reasons.append(f"{part} {height}px < {min_height}")
    record["skipReason"] = ", ".join(reasons) if reasons else "parts not visible"
    if character_id:
        print(f"  {character_id} attribute {text!r} dropped: {record['skipReason']}", flush=True)
    return record


def _person_text(person: dict, min_height: int) -> tuple[str, dict]:
    character_id = person.get("character_id")
    body = _collapse(person.get("body"))
    attributes = person.get("attributes") if isinstance(person.get("attributes"), list) else []
    stats = _part_stats(person)
    extra: dict = {
        "bodySent": bool(body),
        "parts": stats,
        "partMinPixelHeight": min_height,
    }
    if character_id:
        print(f"  {character_id} parts: {_part_size_line(stats)}", flush=True)
    if not body:
        if character_id:
            print(
                f"  warning: {character_id} has no body; omitting their still text",
                flush=True,
            )
        extra["omittedText"] = True
        extra["omitReason"] = "missing body"
        extra["attributes"] = []
        return "", extra
    items = [body]
    attribute_log: list[dict] = []
    for attribute in attributes:
        if not isinstance(attribute, dict):
            continue
        record = _attribute_clause(
            attribute, stats, min_height, str(character_id) if character_id else None
        )
        attribute_log.append(record)
        if record["sent"]:
            items.append(record["text"])
    extra["attributes"] = attribute_log
    return ", ".join(items), extra


def describe_people(
    entries: list[dict] | None,
    min_part_height: int | None = None,
    row_depth_ratio: float | None = None,
) -> tuple[str, list[dict]]:
    """Count, a back row by depth, then each other person by side. No names."""
    floor = load_part_min_pixel_height() if min_part_height is None else int(min_part_height)
    ratio = load_row_depth_ratio() if row_depth_ratio is None else float(row_depth_ratio)
    people = [dict(entry) for entry in (entries or []) if isinstance(entry, dict)]
    if not people:
        return "", []
    width = float(people[0].get("frame_width") or 768)
    for person in people:
        person["side"] = _side(float(person.get("screen_x") or 0), width)
        person.setdefault("visible_fraction", 1.0)
    nearest_depth = min(float(person.get("depth") or 0) for person in people)
    behind = sorted(
        [
            person
            for person in people
            if float(person.get("depth") or 0) > nearest_depth * ratio
        ],
        key=lambda person: float(person.get("screen_x") or 0),
    )
    used: set[int] = set()
    chunks: list[str] = []
    records: list[dict] = []
    has_behind_row = len(behind) >= 2

    def hidden_phrase(person: dict) -> str:
        if float(person.get("visible_fraction") or 1) < BODY_VISIBLE_MIN:
            return "partly hidden"
        return ""

    def labeled(place: str, person: dict) -> str:
        hidden = hidden_phrase(person)
        head = f"{place}, {hidden}" if hidden else place
        text, extra = _person_text(person, floor)
        phrase = f"{head}: {text}" if text else head
        record = {
            "place": place,
            "depth": round(float(person.get("depth") or 0), 3),
            "pixelHeight": _height(person),
            "visibleFraction": round(float(person.get("visible_fraction") or 0), 3),
            "phrase": phrase,
            **extra,
        }
        character_id = person.get("character_id")
        if character_id:
            record["characterId"] = character_id
        records.append(record)
        return phrase

    if has_behind_row:
        for person in behind:
            used.add(id(person))
        diffs = [
            labeled(_row_slot(index, len(behind)), person)
            for index, person in enumerate(behind)
        ]
        chunks.append(
            f"{_count_word(len(behind))} standing in a row behind, left to right: "
            + "; ".join(diffs)
        )

    rest = [
        person
        for person in sorted(people, key=lambda item: float(item.get("screen_x") or 0))
        if id(person) not in used
    ]
    side_words = {"left": "On the left", "right": "On the right", "center": "In the center"}
    for person in rest:
        place = side_words[person["side"]]
        if has_behind_row:
            place += ", in the foreground"
        chunks.append(labeled(place, person))

    noun = "person" if len(people) == 1 else "people"
    return _join([f"{_count_word(len(people))} {noun}", *chunks]), records


def still_people_line(entries: list[dict] | None = None) -> str:
    sentence, _records = describe_people(entries)
    return sentence


def _read_part_stats(path: Path) -> dict[str, dict[str, dict[str, int]]]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}
    found: dict[str, dict[str, dict[str, int]]] = {}
    for character_id, parts in raw.items():
        stats = empty_part_stats()
        if isinstance(parts, dict):
            for name in BODY_PARTS:
                stats[name] = _coerce_part_stat(parts.get(name))
        found[str(character_id)] = stats
    return found


def gather_visible_people(
    show: dict,
    episode: dict,
    scene: dict,
    time_seconds: float,
    *,
    shot_label: str | None = None,
) -> list[dict]:
    """People with pixels in this camera. Words come later from body + attributes."""
    import numpy as np
    from clay_gpu import raster_clay
    from pipeline_paths import guide_path
    from spatial_previs import (
        NEAR_CLIP,
        PROXY_HEIGHT,
        PROXY_WIDTH,
        _anchor_point,
        _camera_basis,
        _character_mesh_batch,
        camera_at,
        episode_character_state,
        project,
    )

    camera = camera_at(scene, time_seconds)
    basis = _camera_basis(camera)
    characters = show.get("characters") or {}
    label = _shot_label(scene, time_seconds, shot_label)
    parts_path = guide_path(
        str(show["id"]),
        int(episode["episodeNumber"]),
        int(scene["sceneNumber"]),
        label,
        "parts",
    ).with_suffix(".json")
    if not parts_path.is_file():
        raise RuntimeError(
            f"Missing previs guide {parts_path}. Run `pnpm run content:previs` first."
        )
    part_stats = _read_part_stats(parts_path)
    layers = []
    for character_id in scene.get("characterIds") or []:
        character = characters.get(character_id)
        if not isinstance(character, dict):
            continue
        state = episode_character_state(episode, character_id, time_seconds)
        position = state.get("position")
        if not isinstance(position, list) or len(position) < 2:
            continue
        proxy = character.get("proxy") if isinstance(character.get("proxy"), dict) else {}
        if not proxy.get("heightMeters"):
            continue
        height_m = float(proxy["heightMeters"])
        batch = _character_mesh_batch(
            show,
            character_id,
            state,
            _anchor_point(show, episode, scene, state.get("lookAtId"), time_seconds),
            colored=False,
        )
        if batch is None:
            continue
        _image, depth = raster_clay(
            [batch],
            basis,
            width=PROXY_WIDTH,
            height=PROXY_HEIGHT,
            near=NEAR_CLIP,
            background=(0, 0, 0),
        )
        covered = np.isfinite(depth)
        if not bool(covered.any()):
            continue
        ys, xs = np.nonzero(covered)
        projected = project((float(position[0]), float(position[1]), height_m * 0.55), camera)
        layers.append(
            {
                "character_id": character_id,
                "screen_x": float(projected[0]) if projected else float(xs.mean()),
                "depth": float(projected[2]) if projected else float(np.nanmin(depth)),
                "x0": float(xs.min()),
                "x1": float(xs.max()),
                "pixel_height": int(ys.max() - ys.min() + 1),
                "body": _collapse(character.get("body")),
                "attributes": _load_attributes(character),
                "part_stats": part_stats.get(str(character_id), empty_part_stats()),
                "frame_width": float(PROXY_WIDTH),
                "_covered": covered,
                "_depth": depth,
            }
        )
    if not layers:
        return []
    owner = np.full(layers[0]["_covered"].shape, -1, dtype=np.int16)
    nearest = np.full(layers[0]["_covered"].shape, np.inf, dtype=np.float32)
    for index, layer in enumerate(layers):
        closer = layer["_covered"] & (layer["_depth"] < nearest)
        owner[closer] = index
        nearest[closer] = layer["_depth"][closer]
    found = []
    for index, layer in enumerate(layers):
        own = int(layer["_covered"].sum())
        visible = int((owner == index).sum())
        layer["visible_fraction"] = visible / own if own else 0.0
        found.append({key: value for key, value in layer.items() if not str(key).startswith("_")})
    return found
