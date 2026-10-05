#!/usr/bin/env python3

from __future__ import annotations

import _scripts_path  # noqa: F401

from clip_spec import clip_frame_count
from pipeline_paths import discover_show_scripts, depth_video_mp4_path, start_still_path
from spatial_previs import load_show, scene_has_spatial_change


def main() -> None:
    show = load_show(discover_show_scripts("the-iron-bride")[0])
    ep = show["episodes"][0]
    rows = []
    for sc in ep["scenes"]:
        tr = sc.get("timeRangeSeconds")
        if not tr or not sc.get("camera"):
            continue
        dur = float(tr[1]) - float(tr[0])
        spatial = scene_has_spatial_change(ep, sc)
        speaker = sc.get("speakerId")
        fc = clip_frame_count(tr)
        depth = depth_video_mp4_path(show["id"], 1, sc["sceneNumber"])
        still = start_still_path(show["id"], 1, sc["sceneNumber"])
        rows.append(
            {
                "n": sc["sceneNumber"],
                "dur": dur,
                "fc": fc,
                "spatial": spatial,
                "speaker": speaker,
                "depth": depth.is_file(),
                "still": still.is_file(),
            }
        )
    print("num dur frames spatial speaker depth still")
    for r in rows:
        print(
            f"{r['n']:02d} {r['dur']:4.1f}s {r['fc']:3d}f spatial={r['spatial']} "
            f"speaker={r['speaker']!r} depth={r['depth']} still={r['still']}"
        )
    print("--- S2: spatial, no speaker, longest ---")
    c2 = [r for r in rows if r["spatial"] and not r["speaker"]]
    c2.sort(key=lambda x: (-x["dur"], -x["fc"]))
    for r in c2[:10]:
        print(r)
    print("--- S3: spatial + speaker, prefer >=4s ---")
    c3 = [r for r in rows if r["spatial"] and r["speaker"]]
    c3.sort(key=lambda x: (-x["dur"], -x["fc"]))
    for r in c3[:10]:
        print(r)


if __name__ == "__main__":
    main()
