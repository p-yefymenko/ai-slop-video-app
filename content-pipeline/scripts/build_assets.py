"""Resolve show needs into prefabs and write one set per location.

``--episode``, ``--scene``, and ``--seed`` are accepted so ``content:render``
can forward the same arguments. They do not change which prefabs are built.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from asset_resolver import AssetResolver, apply_lock_override, export_credits  # noqa: E402
from pipeline_paths import (  # noqa: E402
    LIBRARY_DIR,
    discover_show_scripts,
    load_content_env,
    show_id_for_script,
)


def load_clip_score():
    """Score a clay thumbnail against the need text. Downloads CLIP weights once."""
    try:
        import open_clip
        import torch
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "CLIP ranking needs the optional open_clip package in the content Python. "
            "Keyword ranking is the default."
        ) from exc
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    def score(query: str, image_path: Path) -> float:
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0)
        text = tokenizer([query])
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return float((image_features @ text_features.T)[0, 0])

    return score


def main() -> None:
    load_content_env()
    parser = argparse.ArgumentParser(description="Build prefabs and sets for a show.")
    parser.add_argument("--show", help="Build only this show id")
    parser.add_argument("--episode", type=int, help="Accepted and ignored")
    parser.add_argument("--scene", type=int, help="Accepted and ignored")
    parser.add_argument("--seed", type=int, help="Accepted and ignored")
    parser.add_argument("--force", action="store_true", help="Rebuild prefabs and ignore locks")
    parser.add_argument("--refresh", action="store_true", help="Same as --force")
    parser.add_argument("--offline", action="store_true", help="Use locks and primitives, never search")
    parser.add_argument("--review", action="store_true", help="Keep the top candidates for review")
    parser.add_argument("--rank", choices=("keyword", "clip"), default="keyword")
    parser.add_argument("--pick", nargs=2, metavar=("NEED_HASH", "PREFAB_ID"))
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
    clip_fn = load_clip_score() if args.rank == "clip" else None
    for script in scripts:
        show = json.loads(script.read_text(encoding="utf-8"))
        show_id = show_id_for_script(script)
        resolver = AssetResolver(
            show_id,
            offline=args.offline,
            refresh=args.force or args.refresh,
            review=args.review,
            clip_fn=clip_fn,
        )
        resolved = resolver.resolve_show(show)
        for warning in resolver.warnings:
            print(warning)
        print(f"{show_id}: {len(resolved)} prefabs")


if __name__ == "__main__":
    main()
