"""People sentence for a still. Words come from generalDescription and blocking."""

from __future__ import annotations

import json
from pathlib import Path

ROW_DEPTH_GAP = 0.5
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


def _collapse(raw: object) -> str:
    if not isinstance(raw, str):
        return ""
    return " ".join(raw.split())


def character_appearance_text(character: dict) -> str:
    """Plate and portrait text: generalDescription, then frontalDescription if present."""
    general = _collapse(character.get("generalDescription"))
    frontal = _collapse(character.get("frontalDescription"))
    if not general:
        return ""
    if frontal:
        return f"{general} {frontal}"
    return general


def load_frontal_min_pixel_height(prompts: dict | None = None) -> int:
    data = prompts if isinstance(prompts, dict) else json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    value = data.get("frontalMinPixelHeight") if isinstance(data, dict) else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RuntimeError("prompts.frontalMinPixelHeight must be a positive integer")
    return value


def _facing_camera_ids(manifest: Path) -> list[str]:
    if not manifest.is_file():
        return []
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw]


def _head_mask_has_pixels(path: Path) -> bool:
    if not path.is_file():
        return False
    from PIL import Image

    with Image.open(path) as image:
        return bool(image.convert("L").getextrema()[1] > 0)


def _shot_label(scene: dict, time_seconds: float, shot_label: str | None) -> str:
    if shot_label in ("start", "end"):
        return shot_label
    times = scene.get("timeRangeSeconds") or [0, 0]
    start, end = float(times[0]), float(times[1])
    if start != end and float(time_seconds) == end:
        return "end"
    return "start"


def _frontal_skip_reason(person: dict, min_height: int) -> str | None:
    if not _collapse(person.get("frontal_description")):
        return "no frontalDescription"
    if not person.get("facing_camera"):
        return "not in facing-camera list"
    if not person.get("head_mask_pixels"):
        return "no head-mask pixels"
    height = _height(person)
    if height < min_height:
        return f"pixel height {height} < {min_height}"
    return None


def describe_people(
    entries: list[dict] | None,
    *,
    frontal_min_pixel_height: int | None = None,
) -> tuple[str, list[dict]]:
    """Count, a back row by depth, then each other person by side. No names."""
    people = [dict(entry) for entry in (entries or []) if isinstance(entry, dict)]
    if not people:
        return "", []
    min_height = (
        frontal_min_pixel_height
        if frontal_min_pixel_height is not None
        else load_frontal_min_pixel_height()
    )
    width = float(people[0].get("frame_width") or 768)
    for person in people:
        person["side"] = _side(float(person.get("screen_x") or 0), width)
        person.setdefault("visible_fraction", 1.0)
    nearest_depth = min(float(person.get("depth") or 0) for person in people)
    behind = sorted(
        [
            person
            for person in people
            if float(person.get("depth") or 0) > nearest_depth + ROW_DEPTH_GAP
        ],
        key=lambda person: float(person.get("screen_x") or 0),
    )
    used: set[int] = set()
    chunks: list[str] = []
    records: list[dict] = []

    def hidden_phrase(person: dict) -> str:
        if float(person.get("visible_fraction") or 1) < BODY_VISIBLE_MIN:
            return "partly hidden"
        return ""

    def labeled(place: str, person: dict) -> str:
        hidden = hidden_phrase(person)
        head = f"{place}, {hidden}" if hidden else place
        character_id = person.get("character_id")
        general = _collapse(person.get("general_description"))
        skip = _frontal_skip_reason(person, min_height)
        send_frontal = skip is None
        if not general:
            if character_id:
                print(
                    f"  warning: {character_id} has no generalDescription; omitting their still text",
                    flush=True,
                )
            text = ""
            extra = {
                "omittedText": True,
                "omitReason": "missing generalDescription",
                "frontalDescriptionSent": False,
                "frontalSkipReason": "missing generalDescription",
            }
        else:
            frontal = _collapse(person.get("frontal_description"))
            text = f"{general} {frontal}" if send_frontal else general
            extra = {"frontalDescriptionSent": send_frontal}
            if skip is not None:
                extra["frontalSkipReason"] = skip
            if character_id:
                if send_frontal:
                    print(f"  {character_id} frontalDescription sent", flush=True)
                else:
                    print(f"  {character_id} frontalDescription skipped: {skip}", flush=True)
        phrase = f"{head}: {text}" if text else head
        record = {
            "place": place,
            "depth": round(float(person.get("depth") or 0), 3),
            "pixelHeight": _height(person),
            "visibleFraction": round(float(person.get("visible_fraction") or 0), 3),
            "phrase": phrase,
            **extra,
        }
        if character_id:
            record["characterId"] = character_id
        records.append(record)
        return phrase

    if len(behind) >= 2:
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
        front = float(person.get("depth") or 0) <= nearest_depth + ROW_DEPTH_GAP
        place = side_words[person["side"]] + (", in the foreground" if front else "")
        chunks.append(labeled(place, person))

    noun = "person" if len(people) == 1 else "people"
    return _join([f"{_count_word(len(people))} {noun}", *chunks]), records


def still_people_line(
    entries: list[dict] | None = None,
    *,
    frontal_min_pixel_height: int | None = None,
) -> str:
    sentence, _records = describe_people(
        entries, frontal_min_pixel_height=frontal_min_pixel_height
    )
    return sentence


def gather_visible_people(
    show: dict,
    episode: dict,
    scene: dict,
    time_seconds: float,
    *,
    shot_label: str | None = None,
) -> list[dict]:
    """People with pixels in this camera. Words come later from generalDescription."""
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
    mask_path = guide_path(
        str(show["id"]),
        int(episode["episodeNumber"]),
        int(scene["sceneNumber"]),
        label,
        "faces",
    )
    facing_ids = set(_facing_camera_ids(mask_path.with_suffix(".json")))
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
        spot = mask_path.with_name(f"{mask_path.stem}_{character_id}.png")
        layers.append(
            {
                "character_id": character_id,
                "screen_x": float(projected[0]) if projected else float(xs.mean()),
                "depth": float(projected[2]) if projected else float(np.nanmin(depth)),
                "x0": float(xs.min()),
                "x1": float(xs.max()),
                "pixel_height": int(ys.max() - ys.min() + 1),
                "general_description": _collapse(character.get("generalDescription")),
                "frontal_description": _collapse(character.get("frontalDescription")),
                "facing_camera": character_id in facing_ids,
                "head_mask_pixels": _head_mask_has_pixels(spot),
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
