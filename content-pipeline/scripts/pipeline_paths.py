"""Filesystem layout for one show.

Authored files live under ``shows/<id>/``. Everything the pipeline creates
lives under ``output/<command>/<show>/``:

- ``plates`` — the location picture
- ``assets`` — the location mesh
- ``previs`` — clay scenes
- ``frames`` — character portraits and scene stills
- ``generate`` — scene clips and the episode cut
"""

from __future__ import annotations

import os
import re
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
STAGES = ("plates", "assets", "previs", "frames", "generate")

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


def end_still_path(show_id: str, episode_number: int, scene_number: int) -> Path:
    return episode_dir("frames", show_id, episode_number) / f"scene_{int(scene_number):02d}_end.png"


def clip_path(show_id: str, episode_number: int, scene_number: int) -> Path:
    return episode_dir("generate", show_id, episode_number) / f"scene_{int(scene_number):02d}.mp4"


def episode_video_path(show_id: str, episode_number: int) -> Path:
    return episode_dir("generate", show_id, episode_number) / "episode.mp4"


def thumbnail_path(show_id: str, episode_number: int) -> Path:
    return episode_dir("generate", show_id, episode_number) / "thumbnail.jpg"


def plate_path(show_id: str, location_id: str) -> Path:
    return stage_dir("plates", show_id) / location_id / "plate.png"


def location_dir(show_id: str, location_id: str) -> Path:
    return stage_dir("assets", show_id) / location_id


def legacy_output_moves(episode_directory: Path) -> list[tuple[Path, Path]]:
    """Map one episode's old files onto the command folders. Never lists a delete.

    ``episode_directory`` is ``output/<show>/<episode>`` from the previous layout.
    """
    show_id = episode_directory.parent.name
    episode_number = int(episode_directory.name)
    moves: list[tuple[Path, Path]] = []
    previs = episode_directory / "previs"
    staged_previs = episode_directory / "01_previs"
    sources = [previs, staged_previs]
    for source_root in sources:
        if not source_root.is_dir():
            continue
        for path in sorted(source_root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source_root)
            if path.name == "contact_sheet.png" and len(relative.parts) == 1:
                moves.append((path, previs_dir(show_id, episode_number) / "contact_sheet.png"))
                continue
            if path.name == "blockout.mp4" and len(relative.parts) == 1:
                moves.append((path, episode_blockout_path(show_id, episode_number)))
                continue
            name = path.name
            blockout = _LEGACY_BLOCKOUT.match(name)
            clay = _LEGACY_CLAY.match(name)
            guide = _LEGACY_GUIDE.match(name)
            if blockout and len(relative.parts) == 1:
                moves.append(
                    (
                        path,
                        scene_dir(show_id, episode_number, int(blockout.group(1))) / "blockout.mp4",
                    )
                )
            elif clay and len(relative.parts) == 1:
                moves.append(
                    (
                        path,
                        clay_frame_path(show_id, episode_number, int(clay.group(1)), clay.group(2)),
                    )
                )
            elif guide and len(relative.parts) == 1:
                moves.append(
                    (
                        path,
                        guide_path(
                            show_id,
                            episode_number,
                            int(guide.group(1)),
                            guide.group(2),
                            guide.group(3),
                        ),
                    )
                )
            elif relative.parts[0].startswith("scene_"):
                destination = previs_dir(show_id, episode_number) / relative
                if path.name == "shot.json":
                    destination = destination.with_name("scene.json")
                if destination != path:
                    moves.append((path, destination))
    for path in sorted(episode_directory.glob("scene_*.png")):
        still = _STILL.match(path.name)
        if still:
            label = still.group(2)
            scene_number = int(still.group(1))
            destination = (
                start_still_path(show_id, episode_number, scene_number)
                if label == "start"
                else end_still_path(show_id, episode_number, scene_number)
            )
            moves.append((path, destination))
    stills = episode_directory / "02_postvis" / "stills"
    if stills.is_dir():
        for path in sorted(stills.glob("scene_*.png")):
            still = _STILL.match(path.name)
            if not still:
                continue
            label = still.group(2)
            scene_number = int(still.group(1))
            destination = (
                start_still_path(show_id, episode_number, scene_number)
                if label == "start"
                else end_still_path(show_id, episode_number, scene_number)
            )
            moves.append((path, destination))
    for path in sorted(episode_directory.glob("scene_*.mp4")):
        clip = _CLIP.match(path.name)
        if clip:
            moves.append((path, clip_path(show_id, episode_number, int(clip.group(1)))))
    clips = episode_directory / "03_postvis" / "clips"
    if clips.is_dir():
        for path in sorted(clips.glob("scene_*.mp4")):
            clip = _CLIP.match(path.name)
            if clip:
                moves.append((path, clip_path(show_id, episode_number, int(clip.group(1)))))
    episode_mp4 = episode_directory / "episode.mp4"
    if episode_mp4.is_file():
        moves.append((episode_mp4, episode_video_path(show_id, episode_number)))
    staged_episode = episode_directory / "04_edit" / "episode.mp4"
    if staged_episode.is_file():
        moves.append((staged_episode, episode_video_path(show_id, episode_number)))
    thumbnail = episode_directory / "thumbnail.jpg"
    if thumbnail.is_file():
        moves.append((thumbnail, thumbnail_path(show_id, episode_number)))
    staged_thumb = episode_directory / "04_edit" / "thumbnail.jpg"
    if staged_thumb.is_file():
        moves.append((staged_thumb, thumbnail_path(show_id, episode_number)))
    manifest = episode_directory / "manifest.json"
    if manifest.is_file():
        moves.append((manifest, manifest_path(show_id, episode_number)))
    return moves
