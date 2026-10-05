#!/usr/bin/env python3

import _scripts_path  # noqa: F401

import json
import sys
from pathlib import Path

for path in sys.argv[1:]:
    p = Path(path)
    print(f"\n=== {p} ===")
    for r in json.loads(p.read_text(encoding="utf-8")):
        bd = r["brightness_drift"]
        sd = r["saturation_drift"]
        print(
            f"{r['variant']} L@0/25/50/75/100={[round(x,1) for x in bd['L_mean']]} "
            f"Lmaxd={bd['L_max_abs_delta_from_0']:.1f}"
        )
        print(
            f"{r['variant']} C@0/25/50/75/100={[round(x,1) for x in sd['chroma_mean']]} "
            f"Cmaxd={sd['chroma_max_abs_delta_from_0']:.1f}"
        )
        bc = r.get("brightness_drift_castle")
        if bc:
            print(
                f"{r['variant']} castleL={[round(x,1) for x in bc['L_mean']]} "
                f"Lmaxd={bc['L_max_abs_delta_from_0']:.1f}"
            )
            sc = r["saturation_drift_castle"]
            print(
                f"{r['variant']} castleC={[round(x,1) for x in sc['chroma_mean']]} "
                f"Cmaxd={sc['chroma_max_abs_delta_from_0']:.1f}"
            )
