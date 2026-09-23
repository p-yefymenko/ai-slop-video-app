#!/usr/bin/env python3
"""List or apply the move from flat episode output into stage folders.

Dry-run is the default. Nothing is deleted. ``--apply`` renames files and
leaves anything it does not recognize where it is.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from pipeline_paths import OUTPUT_DIR, legacy_output_moves

_STILL = re.compile(r"^scene_(\d+)_(start|end)\.png$")


def episode_directories(show_id: str | None) -> list[Path]:
    if not OUTPUT_DIR.is_dir():
        return []
    shows = [OUTPUT_DIR / show_id] if show_id else sorted(
        path for path in OUTPUT_DIR.iterdir() if path.is_dir() and ".archive-" not in path.name
    )
    episodes: list[Path] = []
    for show in shows:
        if not show.is_dir():
            continue
        for episode in sorted(show.iterdir()):
            if episode.is_dir() and episode.name.isdigit():
                episodes.append(episode)
    return episodes


def leftover_files(episode_directory: Path, moved_sources: set[Path]) -> list[Path]:
    kept: list[Path] = []
    previs = episode_directory / "previs"
    if previs.is_dir():
        for path in sorted(previs.rglob("*")):
            if path.is_file() and path not in moved_sources:
                kept.append(path)
    for path in sorted(episode_directory.iterdir()):
        if not path.is_file():
            continue
        if path.name == "manifest.json" or path in moved_sources:
            continue
        kept.append(path)
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move flat episode output into 01_previs, 02_postvis, 03_postvis, and 04_edit."
    )
    parser.add_argument("--show", help="Only this show id")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform the moves. Without this flag the command only prints the plan.",
    )
    args = parser.parse_args()
    episodes = episode_directories(args.show)
    if not episodes:
        scope = f" for {args.show}" if args.show else ""
        print(f"No episode output{scope} under {OUTPUT_DIR}.")
        return

    planned = 0
    conflicts = 0
    for episode in episodes:
        moves = legacy_output_moves(episode)
        sources = {source for source, _destination in moves}
        print(f"\n{episode.relative_to(OUTPUT_DIR.parent)}")
        if not moves:
            print("  No legacy files to move.")
        for source, destination in moves:
            planned += 1
            relation = destination.relative_to(episode)
            print(f"  MOVE {source.relative_to(episode)} -> {relation}")
            if destination.exists():
                conflicts += 1
                print("    CONFLICT destination already exists; this file would be left in place.")
        for path in leftover_files(episode, sources):
            note = ""
            if _STILL.match(path.name) and path.parent.name == "previs":
                note = " (kept; the clay frame moves from scene_XX_start_blockout.png)"
            print(f"  LEAVE {path.relative_to(episode)}{note}")

    print(
        f"\n{planned} move(s), {conflicts} conflict(s). "
        "Nothing is deleted. manifest.json and characters/ stay where they are."
    )
    if not args.apply:
        print("Dry run. Re-run with --apply to move these files.")
        return
    if conflicts:
        raise SystemExit("Refusing to apply while a destination already exists.")
    for episode in episodes:
        for source, destination in legacy_output_moves(episode):
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
        previs = episode / "previs"
        if previs.is_dir() and not any(previs.rglob("*")):
            previs.rmdir()
    print("Applied.")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
