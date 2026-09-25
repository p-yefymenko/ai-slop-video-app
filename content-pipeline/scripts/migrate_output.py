#!/usr/bin/env python3
"""Move generated files into output/<command>/<show>/.

Dry-run is the default. Nothing is deleted. ``--apply`` renames files and
writes ``location.json`` / ``plate.json`` beside the meshes and plates.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

from asset_resolver import (
    GENERATED_SOURCE,
    appearance_source_id,
    asset_hash,
    location_scene_description,
)
from pipeline_paths import OUTPUT_DIR, ROOT, SHOWS_DIR, STAGES, legacy_output_moves

LIBRARY = ROOT / "library"
_STILL = re.compile(r"^scene_(\d+)_(start|end)\.png$")


def display(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def legacy_show_dirs(show_id: str | None) -> list[Path]:
    if not OUTPUT_DIR.is_dir():
        return []
    if show_id:
        directory = OUTPUT_DIR / show_id
        return [directory] if directory.is_dir() else []
    return sorted(
        path
        for path in OUTPUT_DIR.iterdir()
        if path.is_dir() and path.name not in STAGES and ".archive-" not in path.name
    )


def episode_directories(show_id: str | None) -> list[Path]:
    episodes: list[Path] = []
    for show in legacy_show_dirs(show_id):
        for episode in sorted(show.iterdir()):
            if episode.is_dir() and episode.name.isdigit():
                episodes.append(episode)
    return episodes


def leftover_files(episode_directory: Path, moved_sources: set[Path]) -> list[Path]:
    kept: list[Path] = []
    for folder_name in ("previs", "01_previs"):
        folder = episode_directory / folder_name
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path not in moved_sources:
                kept.append(path)
    for path in sorted(episode_directory.iterdir()):
        if path.is_file() and path not in moved_sources:
            kept.append(path)
    return kept


def _size_key(values) -> tuple[float, float, float] | None:
    if not isinstance(values, list) or len(values) != 3:
        return None
    return tuple(round(float(value), 4) for value in values)  # type: ignore[return-value]


def _matching_need(lock: dict, size) -> dict | None:
    key = _size_key(size)
    if key is None:
        return None
    matches = []
    for need in (lock.get("needs") or {}).values():
        if isinstance(need, dict) and _size_key(need.get("sizeMeters")) == key:
            matches.append(need)
    if len(matches) != 1:
        return None
    return matches[0]


def legacy_character_moves(show_dir: Path) -> list[tuple[Path, Path]]:
    folder = show_dir / "characters"
    if not folder.is_dir():
        return []
    return [
        (path, OUTPUT_DIR / "frames" / show_dir.name / "characters" / path.name)
        for path in sorted(folder.glob("*.png"))
    ]


def legacy_location_plan(
    show_id: str | None,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, str]], list[str]]:
    """Move set meshes and library plates, and record the current location text."""
    moves: list[tuple[Path, Path]] = []
    writes: list[tuple[Path, str]] = []
    notes: list[str] = []
    lock_path = LIBRARY / "lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.is_file() else {"needs": {}}
    for show_dir in legacy_show_dirs(show_id):
        script_path = SHOWS_DIR / show_dir.name / "script.json"
        if not script_path.is_file():
            notes.append(f"No script at {display(script_path)}; location meshes stay unrecorded.")
            continue
        show = json.loads(script_path.read_text(encoding="utf-8"))
        for location_id, location in (show.get("locations") or {}).items():
            if not isinstance(location, dict):
                continue
            spatial = location.get("spatial") or {}
            need = _matching_need(lock, spatial.get("sizeMeters"))
            glb = show_dir / "sets" / str(location_id) / "set.glb"
            glb_dest = OUTPUT_DIR / "assets" / show_dir.name / str(location_id) / "model.glb"
            plate_dest = OUTPUT_DIR / "plates" / show_dir.name / str(location_id) / "plate.png"
            plate_src: Path | None = None
            prefab: dict = {}
            if need is not None:
                source_id = str(need.get("sourceId") or "")
                if source_id:
                    candidate = LIBRARY / "raw" / "trellis2" / source_id[:16] / "plate.png"
                    if candidate.is_file():
                        plate_src = candidate
                prefab_id = str(need.get("prefabId") or "")
                prefab_path = LIBRARY / "prefabs" / prefab_id / "prefab.json" if prefab_id else None
                if prefab_path is not None and prefab_path.is_file():
                    prefab = json.loads(prefab_path.read_text(encoding="utf-8"))
            if glb.is_file():
                moves.append((glb, glb_dest))
            elif not glb_dest.is_file():
                notes.append(f"{show_dir.name}/{location_id}: no set.glb to move.")
            if plate_src is not None:
                moves.append((plate_src, plate_dest))
            elif need is not None and not plate_dest.is_file():
                notes.append(f"{show_dir.name}/{location_id}: no plate.png in the old library.")
            if need is None or not prefab:
                if glb.is_file() or (plate_src is not None):
                    notes.append(f"{show_dir.name}/{location_id}: mesh or plate moves without a location record.")
                continue
            appearance = location_scene_description(str(location_id), location)
            size = tuple(float(value) for value in spatial["sizeMeters"])
            digest = asset_hash(GENERATED_SOURCE, appearance_source_id(appearance), size)
            record_path = glb_dest.parent / "location.json"
            if glb.is_file() or (glb_dest.is_file() and not record_path.is_file()):
                record = {
                    "showId": show_dir.name,
                    "locationId": location_id,
                    "space": "gltf-y-up",
                    "sizeMeters": list(size),
                    "descriptionHash": digest,
                    "title": appearance,
                    "source": GENERATED_SOURCE,
                    "sourceId": need.get("sourceId") or prefab.get("sourceId") or "",
                    "author": prefab.get("author") or "Qwen-Image-Edit-2511, TRELLIS.2",
                    "license": prefab.get("license") or "MIT",
                    "pageUrl": prefab.get("pageUrl") or "https://github.com/microsoft/TRELLIS.2",
                    "retrieved": prefab.get("retrieved") or date.today().isoformat(),
                    "triangleCount": prefab.get("triangleCount"),
                    "triangleBudget": prefab.get("triangleBudget"),
                    "decimator": prefab.get("decimator"),
                    "fit": prefab.get("fit") or "uniform",
                    "model": "model.glb",
                }
                writes.append((record_path, json.dumps(record, indent=2) + "\n"))
            meta_path = plate_dest.with_name("plate.json")
            if plate_src is not None or (plate_dest.is_file() and not meta_path.is_file()):
                writes.append(
                    (
                        meta_path,
                        json.dumps({"locationId": location_id, "descriptionHash": digest}, indent=2) + "\n",
                    )
                )
    return moves, writes, notes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move generated files into output/plates, assets, previs, frames, and generate."
    )
    parser.add_argument("--show", help="Only this show id")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform the moves. Without this flag the command only prints the plan.",
    )
    args = parser.parse_args()
    episodes = episode_directories(args.show)
    location_moves, writes, notes = legacy_location_plan(args.show)
    character_moves: list[tuple[Path, Path]] = []
    for show_dir in legacy_show_dirs(args.show):
        character_moves.extend(legacy_character_moves(show_dir))
    if not episodes and not location_moves and not writes and not character_moves:
        scope = f" for {args.show}" if args.show else ""
        print(f"No legacy output{scope} under {OUTPUT_DIR}.")
        return

    planned = 0
    conflicts = 0
    seen_destinations: set[Path] = set()

    def plan_move(source: Path, destination: Path) -> None:
        nonlocal planned, conflicts
        planned += 1
        print(f"  MOVE {display(source)} -> {display(destination)}")
        if destination in seen_destinations or destination.exists():
            conflicts += 1
            print("    CONFLICT destination already exists; this file would be left in place.")
        seen_destinations.add(destination)

    for episode in episodes:
        moves = legacy_output_moves(episode)
        sources = {source for source, _destination in moves}
        print(f"\n{display(episode)}")
        if not moves:
            print("  No legacy episode files to move.")
        for source, destination in moves:
            plan_move(source, destination)
        for path in leftover_files(episode, sources):
            note = ""
            if _STILL.match(path.name) and path.parent.name == "previs":
                note = " (kept; the clay frame moves from scene_XX_start_blockout.png)"
            print(f"  LEAVE {display(path)}{note}")

    if location_moves or writes or character_moves or notes:
        print("\nlocations")
    for source, destination in character_moves + location_moves:
        plan_move(source, destination)
    for destination, _text in writes:
        planned += 1
        print(f"  WRITE {display(destination)}")
        if destination.exists():
            conflicts += 1
            print("    CONFLICT destination already exists; this file would be left in place.")
    for note in notes:
        print(f"  NOTE {note}")

    print(f"\n{planned} action(s), {conflicts} conflict(s). Nothing is deleted.")
    if not args.apply:
        print("Dry run. Re-run with --apply to move these files.")
        return
    if conflicts:
        raise SystemExit("Refusing to apply while a destination already exists.")
    for episode in episodes:
        for source, destination in legacy_output_moves(episode):
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
    for source, destination in character_moves + location_moves:
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
    for destination, text in writes:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
    print("Applied.")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
