"""Filesystem layout for one show.

Authored files live under ``shows/<id>/``. Pictures and meshes built from
``script.json`` live under ``output/<id>/``. ``library/`` is the shared prefab
cache, not one show's render.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHOWS_DIR = ROOT / "shows"
LEGACY_SCRIPTS_DIR = ROOT / "scripts_input"
OUTPUT_DIR = ROOT / "output"
LIBRARY_DIR = ROOT / "library"

_LEGACY_BLOCKOUT = re.compile(r"^scene_(\d+)_blockout\.mp4$")
_LEGACY_CLAY = re.compile(r"^scene_(\d+)_(start|end)_blockout\.png$")
_LEGACY_GUIDE = re.compile(
    r"^scene_(\d+)_(start|end)_(faces|depth|pose|condition)\.png$"
)
_STILL = re.compile(r"^scene_(\d+)_(start|end)\.png$")
_CLIP = re.compile(r"^scene_(\d+)\.mp4$")


def show_id_for_script(path: Path) -> str:
    """Folder name for ``shows/<id>/script.json``; filename stem for a legacy file."""
    if path.name == "script.json":
        return path.parent.name
    return path.stem


def show_script_path(show_id: str) -> Path:
    return SHOWS_DIR / show_id / "script.json"


def show_assets_dir(show_id: str) -> Path:
    """Hand-placed prefabs. The pipeline does not write this folder."""
    return SHOWS_DIR / show_id / "assets"


def discover_show_scripts(show_id: str | None = None) -> list[Path]:
    """Prefer ``shows/<id>/script.json``. A legacy file is used only when that is absent."""
    found: dict[str, Path] = {}
    if SHOWS_DIR.is_dir():
        for script in sorted(SHOWS_DIR.glob("*/script.json")):
            found[script.parent.name] = script
    if LEGACY_SCRIPTS_DIR.is_dir():
        for script in sorted(LEGACY_SCRIPTS_DIR.glob("*.json")):
            if script.stem in found:
                print(
                    f"Ignoring legacy script {script}. Using {found[script.stem]}.",
                    flush=True,
                )
                continue
            print(
                f"Reading legacy script {script}. "
                f"Move it to {show_script_path(script.stem)}.",
                flush=True,
            )
            found[script.stem] = script
    if show_id is not None:
        script = found.get(show_id)
        return [script] if script is not None else []
    return [found[key] for key in sorted(found)]


def character_image_path(show_id: str, character_id: str) -> Path:
    return OUTPUT_DIR / show_id / "characters" / f"{character_id}.png"


def episode_dir(show_id: str, episode_number: int) -> Path:
    return OUTPUT_DIR / show_id / str(episode_number)


def manifest_path(show_id: str, episode_number: int) -> Path:
    return episode_dir(show_id, episode_number) / "manifest.json"


def previs_dir(show_id: str, episode_number: int) -> Path:
    return episode_dir(show_id, episode_number) / "01_previs"


def shot_dir(show_id: str, episode_number: int, scene_number: int) -> Path:
    return previs_dir(show_id, episode_number) / f"scene_{int(scene_number):02d}"


def clay_frame_path(show_id: str, episode_number: int, scene_number: int, label: str) -> Path:
    """Clay still. ``label`` is ``start`` or ``end``."""
    return shot_dir(show_id, episode_number, scene_number) / f"{label}.png"


def guide_path(
    show_id: str,
    episode_number: int,
    scene_number: int,
    label: str,
    kind: str,
) -> Path:
    return shot_dir(show_id, episode_number, scene_number) / "guides" / f"{label}_{kind}.png"


def blockout_video_path(show_id: str, episode_number: int, scene_number: int) -> Path:
    return shot_dir(show_id, episode_number, scene_number) / "blockout.mp4"


def shot_description_path(show_id: str, episode_number: int, scene_number: int) -> Path:
    return shot_dir(show_id, episode_number, scene_number) / "shot.json"


def contact_sheet_path(show_id: str, episode_number: int) -> Path:
    return previs_dir(show_id, episode_number) / "contact_sheet.png"


def stills_dir(show_id: str, episode_number: int) -> Path:
    return episode_dir(show_id, episode_number) / "02_postvis" / "stills"


def start_still_path(show_id: str, episode_number: int, scene_number: int) -> Path:
    return stills_dir(show_id, episode_number) / f"scene_{int(scene_number):02d}_start.png"


def end_still_path(show_id: str, episode_number: int, scene_number: int) -> Path:
    return stills_dir(show_id, episode_number) / f"scene_{int(scene_number):02d}_end.png"


def clips_dir(show_id: str, episode_number: int) -> Path:
    return episode_dir(show_id, episode_number) / "03_postvis" / "clips"


def clip_path(show_id: str, episode_number: int, scene_number: int) -> Path:
    return clips_dir(show_id, episode_number) / f"scene_{int(scene_number):02d}.mp4"


def edit_dir(show_id: str, episode_number: int) -> Path:
    return episode_dir(show_id, episode_number) / "04_edit"


def episode_video_path(show_id: str, episode_number: int) -> Path:
    return edit_dir(show_id, episode_number) / "episode.mp4"


def thumbnail_path(show_id: str, episode_number: int) -> Path:
    return edit_dir(show_id, episode_number) / "thumbnail.jpg"


def sets_dir(show_id: str) -> Path:
    return OUTPUT_DIR / show_id / "sets"


def set_dir(show_id: str, location_id: str) -> Path:
    return sets_dir(show_id) / location_id


def fallback_prefab_dir(show_id: str, prefab_id: str) -> Path:
    return OUTPUT_DIR / show_id / "assets" / "fallback" / prefab_id


def library_prefab_dir(category: str, prefab_id: str) -> Path:
    return LIBRARY_DIR / "prefabs" / category / prefab_id


def _scene_number(raw: str) -> str:
    return f"scene_{int(raw):02d}"


def legacy_output_moves(episode_directory: Path) -> list[tuple[Path, Path]]:
    """Map one episode's old flat files onto the stage folders. Never lists a delete."""
    moves: list[tuple[Path, Path]] = []
    previs = episode_directory / "previs"
    if previs.is_dir():
        for path in sorted(previs.iterdir()):
            if not path.is_file():
                continue
            name = path.name
            blockout = _LEGACY_BLOCKOUT.match(name)
            clay = _LEGACY_CLAY.match(name)
            guide = _LEGACY_GUIDE.match(name)
            if blockout:
                destination = (
                    episode_directory
                    / "01_previs"
                    / _scene_number(blockout.group(1))
                    / "blockout.mp4"
                )
            elif clay:
                destination = (
                    episode_directory
                    / "01_previs"
                    / _scene_number(clay.group(1))
                    / f"{clay.group(2)}.png"
                )
            elif guide:
                destination = (
                    episode_directory
                    / "01_previs"
                    / _scene_number(guide.group(1))
                    / "guides"
                    / f"{guide.group(2)}_{guide.group(3)}.png"
                )
            elif name == "contact_sheet.png":
                destination = episode_directory / "01_previs" / "contact_sheet.png"
            else:
                continue
            moves.append((path, destination))
    for path in sorted(episode_directory.glob("scene_*.png")):
        still = _STILL.match(path.name)
        if still:
            moves.append(
                (
                    path,
                    episode_directory
                    / "02_postvis"
                    / "stills"
                    / f"{_scene_number(still.group(1))}_{still.group(2)}.png",
                )
            )
    for path in sorted(episode_directory.glob("scene_*.mp4")):
        clip = _CLIP.match(path.name)
        if clip:
            moves.append(
                (
                    path,
                    episode_directory
                    / "03_postvis"
                    / "clips"
                    / f"{_scene_number(clip.group(1))}.mp4",
                )
            )
    episode_mp4 = episode_directory / "episode.mp4"
    if episode_mp4.is_file():
        moves.append((episode_mp4, episode_directory / "04_edit" / "episode.mp4"))
    thumbnail = episode_directory / "thumbnail.jpg"
    if thumbnail.is_file():
        moves.append((thumbnail, episode_directory / "04_edit" / "thumbnail.jpg"))
    return moves
