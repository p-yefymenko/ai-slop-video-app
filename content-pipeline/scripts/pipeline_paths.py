"""Filesystem layout for one show.

Authored files live under ``shows/<id>/``. Everything the pipeline creates
lives under ``output/<command>/<show>/``:

- ``plates`` — one picture per landmark, prop, and character
- ``assets`` — one mesh per plate
- ``previs`` — rendered scenes, guides, and observations
- ``frames`` — scene stills
- ``generate`` — scene clips and the episode cut
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_content_env() -> None:
    """Load ``content-pipeline/.env`` without overriding variables already set."""
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


SHOWS_DIR = ROOT / "shows"
LEGACY_SCRIPTS_DIR = ROOT / "scripts_input"
OUTPUT_DIR = ROOT / "output"
STAGES = ("plates", "assets", "landmarks", "previs", "frames", "generate")



def show_id_for_script(path: Path) -> str:
    """Folder name for ``shows/<id>/script.json``; filename stem for a legacy file."""
    if path.name == "script.json":
        return path.parent.name
    return path.stem


def show_script_path(show_id: str) -> Path:
    return SHOWS_DIR / show_id / "script.json"


def stage_dir(stage: str, show_id: str) -> Path:
    return OUTPUT_DIR / stage / show_id


def show_assets_dir(show_id: str) -> Path:
    """Generated location meshes for one show."""
    return stage_dir("assets", show_id)


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
    return stage_dir("frames", show_id) / "characters" / f"{character_id}.png"


def episode_dir(stage: str, show_id: str, episode_number: int) -> Path:
    return stage_dir(stage, show_id) / str(episode_number)


def manifest_path(show_id: str, episode_number: int) -> Path:
    return episode_dir("generate", show_id, episode_number) / "manifest.json"


def previs_dir(show_id: str, episode_number: int) -> Path:
    return episode_dir("previs", show_id, episode_number)


def scene_dir(show_id: str, episode_number: int, scene_number: int) -> Path:
    return previs_dir(show_id, episode_number) / f"scene_{int(scene_number):02d}"


def clay_frame_path(show_id: str, episode_number: int, scene_number: int, label: str) -> Path:
    """Clay still. ``label`` is ``start`` or ``end``."""
    return scene_dir(show_id, episode_number, scene_number) / f"{label}.png"


def guide_path(
    show_id: str,
    episode_number: int,
    scene_number: int,
    label: str,
    kind: str,
) -> Path:
    return scene_dir(show_id, episode_number, scene_number) / "guides" / f"{label}_{kind}.png"


def blockout_video_path(show_id: str, episode_number: int, scene_number: int) -> Path:
    return scene_dir(show_id, episode_number, scene_number) / "blockout.mp4"






def episode_blockout_path(show_id: str, episode_number: int) -> Path:
    """Every scene blockout, in script order, as one playable episode."""
    return previs_dir(show_id, episode_number) / "blockout.mp4"


def scene_description_path(show_id: str, episode_number: int, scene_number: int) -> Path:
    return scene_dir(show_id, episode_number, scene_number) / "scene.json"


def contact_sheet_path(show_id: str, episode_number: int) -> Path:
    return previs_dir(show_id, episode_number) / "contact_sheet.png"


def start_still_path(show_id: str, episode_number: int, scene_number: int) -> Path:
    return episode_dir("frames", show_id, episode_number) / f"scene_{int(scene_number):02d}_start.png"



def clip_path(show_id: str, episode_number: int, scene_number: int) -> Path:
    return episode_dir("generate", show_id, episode_number) / f"scene_{int(scene_number):02d}.mp4"


def voice_track_path(show_id: str, episode_number: int, scene_number: int) -> Path:
    """The scene's spoken lines laid out over its length: the sound of its clip."""
    return episode_dir("generate", show_id, episode_number) / "voices" / f"scene_{int(scene_number):02d}.wav"


def episode_video_path(show_id: str, episode_number: int) -> Path:
    return episode_dir("generate", show_id, episode_number) / "episode.mp4"


def thumbnail_path(show_id: str, episode_number: int) -> Path:
    return episode_dir("generate", show_id, episode_number) / "thumbnail.jpg"


def plate_path(show_id: str, location_id: str) -> Path:
    return stage_dir("plates", show_id) / location_id / "plate.png"


def location_dir(show_id: str, location_id: str) -> Path:
    return stage_dir("assets", show_id) / location_id
