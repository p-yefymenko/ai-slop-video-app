#!/usr/bin/env python3

"""Run one LTX depth-gate clip (no pipeline wiring)."""

from __future__ import annotations

import _scripts_path  # noqa: F401

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

from clip_spec import CLIP_HEIGHT, CLIP_WIDTH, clip_frame_count
from gate_depth_lib import (
    GATE_SEED,
    IC_STRENGTH,
    I2V_STRENGTH,
    scene_gate_dir,
    stage_control_video,
    stage_fit_still,
)
from generate_batch import (
    compile_ltx_prompt,
    execute_queued_graph,
    free_comfy_models,
    inject_prompt,
    inject_seed,
    load_show,
)
from pipeline_paths import discover_show_scripts, start_still_path


class NvidiaPeak:
    def __init__(self) -> None:
        self.peak = 0.0
        self._stop = threading.Event()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                ).strip()
                self.peak = max(self.peak, float(out.splitlines()[0].strip()))
            except Exception:
                pass
            self._stop.wait(1.0)

    def __enter__(self) -> "NvidiaPeak":
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def __exit__(self, *args: object) -> None:
        self._stop.set()

ROOT = Path(__file__).resolve().parents[2]  # content-pipeline/
DEPTH_WF = ROOT / "workflows" / "ltx_gemma_api_depth.json"
DIALOGUE_WF = ROOT / "workflows" / "ltx_gemma_api_depth_dialogue.json"

VARIANT_FILES = {
    "A": "depth_inverse.mp4",
    "A_gamma": "depth_inverse_gamma06.mp4",
    "A_eq": "depth_inverse_eq.mp4",
    "B": "depth_linear.mp4",
    "C": "depth_depthanything.mp4",
}
OUTPUT_NAMES = {
    "A": "A_inverse.mp4",
    "A_gamma": "A_gamma06.mp4",
    "A_eq": "A_eq.mp4",
    "B": "B_linear.mp4",
    "C": "C_depthanything.mp4",
}


def inject_depth_length(graph: dict, length: int) -> None:
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        inputs = node.setdefault("inputs", {})
        ctype = node.get("class_type")
        if ctype in {"EmptyLTXVLatentVideo", "LTXVImgToVideo"}:
            inputs["length"] = length
            if "width" in inputs:
                inputs["width"] = CLIP_WIDTH
            if "height" in inputs:
                inputs["height"] = CLIP_HEIGHT
        elif ctype == "LTXVEmptyLatentAudio":
            inputs["frames_number"] = length


def inject_depth_inputs(graph: dict, still_name: str, depth_name: str) -> None:
    graph["19"]["inputs"]["image"] = still_name
    graph["20"]["inputs"]["strength"] = I2V_STRENGTH
    graph["20"]["inputs"]["width"] = CLIP_WIDTH
    graph["20"]["inputs"]["height"] = CLIP_HEIGHT
    graph["30"]["inputs"]["strength_model"] = IC_STRENGTH
    graph["31"]["inputs"]["file"] = depth_name
    # Keep IC-LoRA wiring: concat/sampler/guider stay on node 32.


def load_api_key() -> str:
    env_path = ROOT / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("LTXV_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"')
    return os.environ.get("LTXV_API_KEY", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", default="the-iron-bride")
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--scene", type=int, required=True)
    parser.add_argument("--variant", required=True, choices=sorted(VARIANT_FILES))
    parser.add_argument("--dialogue", action="store_true")
    parser.add_argument("--tiled-vae", action="store_true")
    args = parser.parse_args()

    show = load_show(discover_show_scripts(args.show)[0])
    episode = next(ep for ep in show["episodes"] if ep["episodeNumber"] == args.episode)
    scene = next(sc for sc in episode["scenes"] if sc["sceneNumber"] == args.scene)
    length = clip_frame_count(scene["timeRangeSeconds"])
    still = start_still_path(show["id"], args.episode, args.scene)
    if not still.is_file():
        raise SystemExit(f"Missing start still {still}")
    control = scene_gate_dir(args.scene) / VARIANT_FILES[args.variant]
    if not control.is_file():
        raise SystemExit(f"Missing control video {control}; run gate_depth_prepare.py first")

    wf_path = DIALOGUE_WF if args.dialogue else DEPTH_WF
    workflow = json.loads(wf_path.read_text(encoding="utf-8"))
    location = show["locations"][scene["locationId"]]
    prompt = compile_ltx_prompt(show, scene, location)
    api_key = load_api_key()
    if not api_key:
        raise SystemExit("LTXV_API_KEY missing from content-pipeline/.env")
    graph = inject_prompt(workflow, prompt, api_key)
    inject_depth_length(graph, length)
    inject_seed(graph, GATE_SEED)
    still_name = stage_fit_still(still, f"gate_start_s{args.scene:02d}.png")
    depth_name = stage_control_video(control, f"gate_depth_s{args.scene:02d}_{args.variant}.mp4")
    inject_depth_inputs(graph, still_name, depth_name)
    if args.tiled_vae:
        # Core ComfyUI tiled decode if present.
        node = graph.get("8")
        if node and node.get("class_type") == "VAEDecode":
            node["class_type"] = "VAEDecodeTiled"
            node["inputs"]["tile_size"] = 512
            node["inputs"]["overlap"] = 64
            print("Using VAEDecodeTiled (tile_size=512)", flush=True)

    out_name = OUTPUT_NAMES[args.variant]
    if args.dialogue:
        out_name = out_name.replace(".mp4", "_dialogue.mp4")
    dest = scene_gate_dir(args.scene) / out_name
    prefix = f"gate_ltx_s{args.scene:02d}_{args.variant}"
    graph["10"]["inputs"]["filename_prefix"] = prefix

    print(
        f"LTX gate scene={args.scene} variant={args.variant} frames={length} "
        f"seed={GATE_SEED} dialogue={args.dialogue} prompt_chars={len(prompt)}",
        flush=True,
    )
    free_comfy_models()
    started = time.time()
    with NvidiaPeak() as peak:
        execute_queued_graph(
            graph,
            dest,
            prefer="video",
            mode=f"depth-gate {args.variant}",
        )
    wall = time.time() - started
    free_comfy_models()
    meta = {
        "scene": args.scene,
        "variant": args.variant,
        "dialogue": args.dialogue,
        "frames": length,
        "seed": GATE_SEED,
        "i2v": I2V_STRENGTH,
        "ic_lora": IC_STRENGTH,
        "wall_seconds": wall,
        "peak_vram_mib": peak.peak,
        "control": str(control),
        "still": str(still),
        "output": str(dest),
        "workflow": str(wf_path.name),
        "tiled_vae": args.tiled_vae,
    }
    (scene_gate_dir(args.scene) / f"{dest.stem}_run.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {dest} in {wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
