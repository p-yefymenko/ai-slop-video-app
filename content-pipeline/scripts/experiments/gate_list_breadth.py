#!/usr/bin/env python3

"""Pick breadth-test scenes for gate 3b."""

from __future__ import annotations

import _scripts_path  # noqa: F401

import math

from clip_spec import clip_frame_count
from pipeline_paths import clay_24fps_path, discover_show_scripts, start_still_path
from spatial_previs import camera_travel_score, load_show, scene_has_spatial_change


def camera_travel(scene: dict) -> float:
    return camera_travel_score(scene)


def main() -> None:
    show = load_show(discover_show_scripts("the-iron-bride")[0])
    ep = show["episodes"][0]
    exclude = {5, 13, 19}
    rows = []
    for sc in ep["scenes"]:
        if not scene_has_spatial_change(ep, sc):
            continue
        n = int(sc["sceneNumber"])
        tr = sc["timeRangeSeconds"]
        dur = float(tr[1]) - float(tr[0])
        loc = sc["locationId"]
        chars = list(sc.get("characterIds") or [])
        rows.append(
            {
                "n": n,
                "dur": dur,
                "fc": clip_frame_count(tr),
                "travel": camera_travel(sc),
                "chars": len(chars),
                "char_ids": chars,
                "loc": loc,
                "speaker": sc.get("speakerId"),
                "kfs": len((sc.get("camera") or {}).get("keyframes") or []),
                "clay": clay_24fps_path(show["id"], 1, n).is_file(),
                "still": start_still_path(show["id"], 1, n).is_file(),
            }
        )
    print("all spatial scenes:")
    for r in sorted(rows, key=lambda x: x["n"]):
        print(
            f"{r['n']:02d} dur={r['dur']:.1f} fc={r['fc']} travel={r['travel']:.2f} "
            f"chars={r['chars']} loc={r['loc']} speaker={r['speaker']!r} "
            f"kfs={r['kfs']} clay={r['clay']} still={r['still']}"
        )

    cand = [r for r in rows if r["n"] not in exclude and r["dur"] <= 5.0 + 1e-9]
    # (i) largest camera travel
    i = max(cand, key=lambda r: r["travel"])
    # (ii) 1-2 chars, medium/close, camera move (kfs>=2 or travel>0)
    ii_pool = [r for r in cand if 1 <= r["chars"] <= 2 and r["travel"] > 0.05]
    ii = max(ii_pool, key=lambda r: r["travel"]) if ii_pool else None
    # (iii) most characters
    iii = max(cand, key=lambda r: (r["chars"], r["travel"]))
    # (iv) interior/enclosed if any; else most different location from 05/13/19.
    # Show only has sun_well_court (enclosed court) and sky_forge_exterior вЂ” no true
    # interior. Prefer an unused enclosed-court scene with camera travel (scene 04).
    used = {i["n"], ii["n"] if ii else -1, iii["n"]}
    known_locs = {"sun_well_court", "sky_forge_exterior"}  # S1/S3 and S2
    iv_pool = [r for r in cand if r["loc"] not in known_locs and r["n"] not in used]
    if not iv_pool:
        # No third location: pick enclosed court (sun_well_court) with travel, unused.
        iv_pool = [
            r
            for r in cand
            if r["loc"] == "sun_well_court" and r["travel"] > 0.05 and r["n"] not in used
        ]
    iv = max(iv_pool, key=lambda r: r["travel"]) if iv_pool else None

    picks = {"i_travel": i, "ii_medium": ii, "iii_crowd": iii, "iv_enclosed": iv}
    print("\nPICKS:")
    for key, r in picks.items():
        print(key, r)


if __name__ == "__main__":
    main()
