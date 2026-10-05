"""Body-part ids for character attributes. The enum lives in script.ts; keep this list identical."""

from __future__ import annotations

import re

import numpy as np

BODY_PARTS = (
    "hair",
    "face",
    "eyes",
    "neck",
    "torso",
    "arms",
    "hands",
    "legs",
    "feet",
)
BODY_PART_INDEX = {name: index for index, name in enumerate(BODY_PARTS)}

# Standing-height bands. Meshes are upright, fitted to height, feet at z=0, front +Y.
_FEET = 0.07
_LEGS = 0.48
_HANDS_LO = 0.42
_HANDS_HI = 0.52
_TORSO = 0.76
_NECK = 0.82
_EYES_LO = 0.86
_EYES_HI = 0.91
_FACE = 0.92
_ARM_X = 0.55
_HAND_X = 0.50

# Viewer debug colors, 0-1 RGB. Keep in sync with content-pipeline/viewer/src/stage.js.
PART_DEBUG_COLORS = {
    "hair": (0.77, 0.24, 0.24),
    "face": (0.94, 0.78, 0.47),
    "eyes": (0.24, 0.47, 0.94),
    "neck": (0.82, 0.47, 0.78),
    "torso": (0.24, 0.71, 0.31),
    "arms": (0.94, 0.63, 0.16),
    "hands": (0.63, 0.31, 0.16),
    "legs": (0.31, 0.63, 0.78),
    "feet": (0.78, 0.78, 0.31),
}

LTX_WIDTH = 448
LTX_MIN_PART_WIDTH = 5

# Anatomy words in scene text that need that part on screen in the start still.
# Wardrobe words (boots, gloves, collar) stay out; attributes gate those by part.
# "clench" stays out: it is as often teeth or jaw as fists.
_PART_WORDS = {
    "hair": r"hair",
    "face": r"(?:her|his|their|its|a|the)\s+face|mouth|lips?|teeth|jaw|cheeks?|chin|nose|forehead",
    "eyes": r"eyes?",
    "neck": r"neck|throat|nape",
    "torso": r"chest|torso|belly|stomach|waist",
    "arms": r"arms?|forearms?|elbows?",
    "hands": r"hands?|fists?|fingers?|fingertips?|palms?|knuckles?|wrists?"
    r"|grip(?:s|ped|ping)?|clutch(?:es|ed|ing)?",
    "legs": r"legs?|knees?|thighs?|shins?",
    "feet": r"foot|feet|barefoot|heels?|toes?",
}
_PART_PATTERNS = {
    part: re.compile(rf"\b(?:{words})\b", re.IGNORECASE) for part, words in _PART_WORDS.items()
}
_QUOTED = re.compile(r"\"[^\"]*\"|“[^”]*”")


def mentioned_parts(text: str) -> dict[str, list[str]]:
    """Body part -> the words that name it. Quoted dialogue is not a shot description."""
    unquoted = _QUOTED.sub(" ", text or "")
    found: dict[str, list[str]] = {}
    for part, pattern in _PART_PATTERNS.items():
        words = [match.group(0) for match in pattern.finditer(unquoted)]
        if words:
            found[part] = words
    return found


def _collapse(raw: object) -> str:
    if not isinstance(raw, str):
        return ""
    return " ".join(raw.split())


def validate_show_parts(show: dict) -> None:
    """Fail before GPU work. Messages name the character id and field."""
    characters = show.get("characters")
    if not isinstance(characters, dict):
        raise SystemExit("characters is required")
    for character_id, character in characters.items():
        if not isinstance(character, dict):
            raise SystemExit(f"characters.{character_id} must be an object")
        extra = set(character) - {"body", "attributes", "proxy", "id"}
        if extra:
            field = sorted(extra)[0]
            raise SystemExit(f"characters.{character_id}.{field}: unknown field")
        if not _collapse(character.get("body")):
            raise SystemExit(f"characters.{character_id}.body: missing body")
        attributes = character.get("attributes")
        if not isinstance(attributes, list):
            raise SystemExit(f"characters.{character_id}.attributes: attributes is required")
        for attribute_index, attribute in enumerate(attributes):
            if not isinstance(attribute, dict):
                raise SystemExit(
                    f"characters.{character_id}.attributes[{attribute_index}] must be an object"
                )
            extra_attr = set(attribute) - {"text", "parts"}
            if extra_attr:
                field = sorted(extra_attr)[0]
                raise SystemExit(
                    f"characters.{character_id}.attributes[{attribute_index}].{field}: unknown field"
                )
            if not _collapse(attribute.get("text")):
                raise SystemExit(
                    f"characters.{character_id}.attributes[{attribute_index}].text: text is required"
                )
            parts = attribute.get("parts")
            if not isinstance(parts, list) or not parts:
                raise SystemExit(
                    f"characters.{character_id}.attributes[{attribute_index}].parts: empty parts"
                )
            for part_index, part in enumerate(parts):
                if part not in BODY_PART_INDEX:
                    raise SystemExit(
                        f"characters.{character_id}.attributes[{attribute_index}].parts[{part_index}]: "
                        f"unknown part id {part!r}"
                    )
    for episode in show.get("episodes") or []:
        if not isinstance(episode, dict):
            continue
        for scene_index, scene in enumerate(episode.get("scenes") or []):
            if not isinstance(scene, dict):
                continue
            required = scene.get("requiresParts")
            if required is None:
                continue
            scene_path = (
                f"episodes[{episode.get('episodeNumber', '?')}].scenes[{scene_index}].requiresParts"
            )
            if not isinstance(required, list):
                raise SystemExit(f"{scene_path}: requiresParts must be an array")
            character_ids = [str(item) for item in (scene.get("characterIds") or [])]
            for requirement_index, requirement in enumerate(required):
                path = f"{scene_path}[{requirement_index}]"
                if not isinstance(requirement, dict):
                    raise SystemExit(f"{path} must be an object")
                character_id = requirement.get("characterId")
                if not isinstance(character_id, str) or not character_id:
                    raise SystemExit(f"{path}.characterId: characterId is required")
                if character_id not in characters:
                    raise SystemExit(f"{path}.characterId: unknown character {character_id!r}")
                if character_id not in character_ids:
                    raise SystemExit(
                        f"{path}.characterId: {character_id!r} is not in characterIds"
                    )
                part = requirement.get("part")
                if part not in BODY_PART_INDEX:
                    raise SystemExit(f"{path}.part: unknown part id {part!r}")


def empty_part_counts() -> dict[str, int]:
    return {name: 0 for name in BODY_PARTS}


def part_ids_for_vertices(vertices: np.ndarray, height_meters: float | None = None) -> np.ndarray:
    """Integer part index per vertex. Local schema space: Z up, +Y front, feet at z=0."""
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        return np.zeros((0,), dtype=np.int16)
    height = float(height_meters) if height_meters and height_meters > 0 else float(points[:, 2].max())
    height = max(height, 1e-6)
    t = points[:, 2] / height
    span = float(np.max(np.abs(points[:, 0]))) if len(points) else 0.0
    span = max(span, 1e-6)
    abs_x = np.abs(points[:, 0]) / span
    ids = np.full(len(points), BODY_PART_INDEX["torso"], dtype=np.int16)
    ids[t < _FEET] = BODY_PART_INDEX["feet"]
    legs = (t >= _FEET) & (t < _LEGS)
    ids[legs] = BODY_PART_INDEX["legs"]
    hands = legs & (t >= _HANDS_LO) & (t < _HANDS_HI) & (abs_x >= _HAND_X)
    ids[hands] = BODY_PART_INDEX["hands"]
    torso = (t >= _LEGS) & (t < _TORSO)
    ids[torso] = BODY_PART_INDEX["torso"]
    ids[torso & (abs_x >= _ARM_X)] = BODY_PART_INDEX["arms"]
    ids[(t >= _TORSO) & (t < _NECK)] = BODY_PART_INDEX["neck"]
    head = t >= _NECK
    ids[head] = BODY_PART_INDEX["hair"]
    if np.any(head):
        head_forward = points[head, 1]
        center = 0.5 * (float(head_forward.min()) + float(head_forward.max()))
        front = head & (points[:, 1] >= center)
        ids[front & (t < _FACE)] = BODY_PART_INDEX["face"]
        ids[front & (t >= _EYES_LO) & (t < _EYES_HI)] = BODY_PART_INDEX["eyes"]
    return ids


def part_id_colors(part_ids: np.ndarray, character_index: int) -> np.ndarray:
    """Linear RGB for the part-ID buffer. R = character+1, G = part+1. No gamma."""
    colors = np.zeros((len(part_ids), 3), dtype=np.float32)
    if len(part_ids) == 0:
        return colors
    colors[:, 0] = (int(character_index) + 1) / 255.0
    colors[:, 1] = (np.asarray(part_ids, dtype=np.float32) + 1.0) / 255.0
    colors[:, 2] = 128.0 / 255.0
    return colors


def debug_part_colors(part_ids: np.ndarray) -> np.ndarray:
    colors = np.zeros((len(part_ids), 3), dtype=np.float32)
    for index, name in enumerate(BODY_PARTS):
        colors[part_ids == index] = PART_DEBUG_COLORS[name]
    return colors


def decode_part_buffer(
    image: np.ndarray,
    character_ids: list[str],
) -> dict[str, dict[str, dict[str, int]]]:
    """Per characterId, visible pixels and bounding width/height per part."""
    pixels = np.asarray(image)
    found: dict[str, dict[str, dict[str, int]]] = {}
    if pixels.ndim != 3 or pixels.shape[2] < 2:
        return {
            character_id: {name: {"pixels": 0, "width": 0, "height": 0} for name in BODY_PARTS}
            for character_id in character_ids
        }
    red = pixels[:, :, 0]
    green = pixels[:, :, 1]
    for character_index, character_id in enumerate(character_ids):
        counts: dict[str, dict[str, int]] = {}
        owner = red == (character_index + 1)
        for part_index, name in enumerate(BODY_PARTS):
            mask = owner & (green == (part_index + 1))
            count = int(mask.sum())
            width = 0
            height = 0
            if count:
                columns = np.where(mask.any(axis=0))[0]
                rows = np.where(mask.any(axis=1))[0]
                if columns.size:
                    width = int(columns[-1] - columns[0] + 1)
                if rows.size:
                    height = int(rows[-1] - rows[0] + 1)
            counts[name] = {"pixels": count, "width": width, "height": height}
        found[character_id] = counts
    return found


def empty_part_stats() -> dict[str, dict[str, int]]:
    return {name: {"pixels": 0, "width": 0, "height": 0} for name in BODY_PARTS}


def part_stats_json(stats: dict[str, dict[str, dict[str, int]]]) -> dict[str, dict[str, dict[str, int]]]:
    """pixels, bounding width, and bounding height per part, for the still log and N."""
    written: dict[str, dict[str, dict[str, int]]] = {}
    for character_id, parts in stats.items():
        written[character_id] = {}
        for part, values in parts.items():
            written[character_id][part] = {
                "pixels": int(values.get("pixels") or 0),
                "width": int(values.get("width") or 0),
                "height": int(values.get("height") or 0),
            }
    return written


def part_counts_json(stats: dict[str, dict[str, dict[str, int]]]) -> dict[str, dict[str, dict[str, int]]]:
    return part_stats_json(stats)


def ltx_part_width(previs_width: int, previs_image_width: int) -> float:
    if previs_image_width <= 0:
        return 0.0
    return float(previs_width) * (LTX_WIDTH / float(previs_image_width))
