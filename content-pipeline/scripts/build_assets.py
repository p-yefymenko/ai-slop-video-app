"""Draw every entity's one picture (an object's plate, a character's reference photo) and each character's voice, or mesh reviewed pictures.

``--episode``, ``--scene``, ``--seed``, and ``--size`` are accepted so ``content:render``
can forward the same arguments. They do not change what is built.
``content:render`` meshes only. It does not draw plates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from asset_generate import draw_picture  # noqa: E402
from mesh_io import TRIANGLE_BUDGET, require_triangle_budget  # noqa: E402
from asset_resolver import (  # noqa: E402
    AssetResolver,
    export_credits,
    write_pictures,
)
from pipeline_paths import (  # noqa: E402
    OUTPUT_DIR,
    discover_show_scripts,
    load_content_env,
    show_id_for_script,
)


def write_voices(show: dict, *, refresh: bool, offline: bool) -> None:
    """Every character's voice, made from their ``voice`` description; a character without one has no voice."""
    from comfy import free_comfy_models
    from voice import design_voices, voice_is_current
    from world import voice_path

    stale = [
        cid for cid, character in show["characters"].items()
        if character.get("voice") and (refresh or not voice_is_current(character, voice_path(show, cid)))
    ]
    if not stale:
        return
    if offline:
        raise SystemExit(f"Missing or stale voices for {', '.join(stale)}, and --offline does not generate.")
    print(f"Voices for {', '.join(stale)}...", flush=True)
    free_comfy_models()
    design_voices(show, stale)


def main() -> None:
    load_content_env()
    parser = argparse.ArgumentParser(description="Draw plates, or mesh reviewed plates.")
    parser.add_argument(
        "--stage",
        choices=("plates", "meshes"),
        default="meshes",
        help="plates draws images and stops. meshes builds GLBs from those images.",
    )
    parser.add_argument("--show", help="Build only this show id")
    parser.add_argument("--episode", type=int, help="Accepted and ignored")
    parser.add_argument("--scene", type=int, help="Accepted and ignored")
    parser.add_argument("--seed", type=int, help="Accepted and ignored")
    parser.add_argument("--size", help="Accepted and ignored")
    parser.add_argument("--force", action="store_true", help="Rebuild and ignore the saved mesh")
    parser.add_argument("--refresh", action="store_true", help="Same as --force")
    parser.add_argument("--offline", action="store_true", help="Do not generate. A missing file fails the build")
    parser.add_argument("--credits", action="store_true", help="Rewrite docs/CREDITS.md from generated locations")
    parser.add_argument(
        "--triangles",
        type=int,
        default=TRIANGLE_BUDGET,
        help=f"Polygon limit for the mesh. Default {TRIANGLE_BUDGET}.",
    )
    args = parser.parse_args()
    if args.credits:
        destination = Path(__file__).resolve().parents[2] / "docs" / "CREDITS.md"
        export_credits(OUTPUT_DIR, destination)
        print(destination)
        return
    scripts = discover_show_scripts(args.show)
    if not scripts:
        target = f" for show {args.show!r}" if args.show else ""
        raise SystemExit(f"No show JSON files found{target}")
    try:
        triangle_budget = require_triangle_budget(args.triangles)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.stage == "plates":
        for script in scripts:
            show = json.loads(script.read_text(encoding="utf-8"))
            write_voices(show, refresh=args.force or args.refresh, offline=args.offline)
            pictures = write_pictures(show, draw_picture, refresh=args.force or args.refresh, offline=args.offline)
            print(f"{show_id_for_script(script)}: {len(pictures)} pictures")
        print("Review the plates, references and voices, then run `pnpm run content:assets`.")
        return
    for script in scripts:
        show = json.loads(script.read_text(encoding="utf-8"))
        show_id = show_id_for_script(script)
        resolver = AssetResolver(
            show_id,
            offline=args.offline,
            refresh=args.force or args.refresh,
            triangle_budget=triangle_budget,
        )
        resolved = resolver.resolve_show(show)
        for warning in resolver.warnings:
            print(warning)
        print(f"{show_id}: {len(resolved)} meshes, polygon limit {triangle_budget}")


if __name__ == "__main__":
    main()
