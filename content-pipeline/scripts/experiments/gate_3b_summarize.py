#!/usr/bin/env python3

import _scripts_path  # noqa: F401

import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
for r in json.loads(path.read_text(encoding="utf-8")):
    f = r["camera_follow"]
    bd = r["brightness_drift"]
    bc = r.get("brightness_drift_castle") or {}
    print(
        f"{r['variant']}: follow={f['mean']:.3f}/{f['min']:.3f} "
        f"wall={r.get('wall_seconds')} vram={r.get('peak_vram_mib')} "
        f"Lmax={bd['L_max_abs_delta_from_0']:.1f} "
        f"Cmax={bd['chroma_max_abs_delta_from_0']:.2f} "
        f"castleLmax={bc.get('L_max_abs_delta_from_0')} "
        f"appear={r['appearance']['mean']} "
        f"mod3={[round(x, 2) for x in r['stutter_mod3']]} "
        f"ff={r['first_frame_delta']:.2f}"
    )
