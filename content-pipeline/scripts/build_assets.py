"""Draw location plates, or mesh plates that have already been reviewed.

``--episode``, ``--scene``, and ``--seed`` are accepted so ``content:render``
can forward the same arguments. They do not change which prefabs are built.
``content:render`` meshes only. It does not draw plates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from asset_generate import generate_asset_plate  # noqa: E402
from asset_resolver import (  # noqa: E402
    AssetResolver,
    apply_lock_override,
    export_credits,
    prune_unused_library,
    write_location_plates,
)
from pipeline_paths import (  # noqa: E402
    LIBRARY_DIR,
    discover_show_scripts,
    load_content_env,
    show_id_for_script,
)


def main() -> None:
    load_content_env()
    parser = argparse.ArgumentParser(description="Draw location plates, or mesh reviewed plates.")
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
    parser.add_argument("--force", action="store_true", help="Rebuild prefabs and ignore locks")
    parser.add_argument("--refresh", action="store_true", help="Same as --force")
    parser.add_argument("--offline", action="store_true", help="Use the library only; a missing model fails the build")
    parser.add_argument("--pick", nargs=2, metavar=("ASSET_HASH", "PREFAB_ID"))
    parser.add_argument("--credits", action="store_true", help="Rewrite docs/CREDITS.md from library/sources.json")
    args = parser.parse_args()
    if args.credits:
        destination = Path(__file__).resolve().parents[2] / "docs" / "CREDITS.md"
        export_credits(LIBRARY_DIR / "sources.json", destination)
        print(destination)
        return
    if args.pick:
        apply_lock_override(LIBRARY_DIR / "lock.json", args.pick[0], args.pick[1])
        print(f"Locked {args.pick[0]} to {args.pick[1]}")
        return
    scripts = discover_show_scripts(args.show)
    if not scripts:
        target = f" for show {args.show!r}" if args.show else ""
        raise SystemExit(f"No show JSON files found{target}")
    if args.stage == "plates":
        for script in scripts:
            show = json.loads(script.read_text(encoding="utf-8"))
            plates = write_location_plates(
                show,
                LIBRARY_DIR,
                generate_asset_plate,
                refresh=args.force or args.refresh,
                offline=args.offline,
            )
            print(f"{show_id_for_script(script)}: {len(plates)} plates")
        print("Review the plates, then run `pnpm run content:assets`.")
        return
    for script in scripts:
        show = json.loads(script.read_text(encoding="utf-8"))
        show_id = show_id_for_script(script)
        resolver = AssetResolver(
            show_id,
            offline=args.offline,
            refresh=args.force or args.refresh,
        )
        resolved = resolver.resolve_show(show)
        for warning in resolver.warnings:
            print(warning)
        print(f"{show_id}: {len(resolved)} prefabs")
    shows = [json.loads(path.read_text(encoding="utf-8")) for path in discover_show_scripts(None)]
    removed = prune_unused_library(LIBRARY_DIR, shows)
    if removed:
        print(f"Removed {len(removed)} library prefabs that no show uses")


if __name__ == "__main__":
    main()
