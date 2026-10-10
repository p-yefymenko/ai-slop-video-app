#!/usr/bin/env python3
"""Depth Anything control video + IC-LoRA depth graph helpers for content:generate."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

from clip_spec import CLIP_HEIGHT, CLIP_WIDTH, FPS, clip_frame_count, fit_to_clip
from ffmpeg_tools import find_ffmpeg
from PIL import Image
from pipeline_paths import clay_24fps_path, control_depth_mp4_path

ROOT = Path(__file__).resolve().parents[1]
PROMPTS_PATH = ROOT / "prompts.json"
COMFY_INPUT = ROOT / ".comfyui" / "input"
COMFY_OUTPUT = ROOT / ".comfyui" / "output"
LTX_DEPTH_WF = ROOT / "workflows" / "ltx_gemma_api_depth.json"
LTX_PLAIN_WF = ROOT / "workflows" / "ltx_gemma_api.json"


def load_renderer_config(prompts: dict | None = None) -> dict:
    data = prompts if prompts is not None else json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    da = data.get("depthAnything") or {}
    options = data.get("sceneRenderOptions") or {}
    if not isinstance(options, dict):
        options = {}
    loudness = data.get("episodeLoudnessTargetLufs")
    loudness_target = float(loudness) if isinstance(loudness, (int, float)) else None
    upscale = data.get("ltxSpatialUpscale")
    if upscale not in (None, "x2", "x1.5"):
        raise SystemExit(f'prompts.json ltxSpatialUpscale must be "x2", "x1.5", or null, not {upscale!r}')
    return {
        "ltxSpatialUpscale": upscale,
        "endStill": bool(data.get("endStill", False)),
        "verifyAttempts": int(data.get("verifyAttempts", 3)),
        "ltxStartStrength": float(data.get("ltxStartStrength", 0.7)),
        "ltxIcLoRAStrength": float(data.get("ltxIcLoRAStrength", 1.0)),
        "controlDepth": str(data.get("controlDepth") or "depthanything"),
        "cameraTravelWarnThreshold": float(data.get("cameraTravelWarnThreshold", 20)),
        "episodeLoudnessTargetLufs": loudness_target,
        "sceneRenderOptions": options,
        "depthAnything": {
            "weights": str(da.get("weights") or "video_depth_anything_vits.pth"),
            "inputSize": int(da.get("inputSize") or 518),
            "precision": str(da.get("precision") or "fp16"),
        },
    }


def scene_render_option_key(show_id: str, episode_number: int, scene_number: int) -> str:
    return f"{show_id}/{int(episode_number)}/{int(scene_number)}"


def scene_depth_control_enabled(
    show_id: str,
    episode_number: int,
    scene_number: int,
    config: dict | None = None,
) -> bool:
    """False when sceneRenderOptions[<show>/<ep>/<scene>].depthControl is false."""
    cfg = config if config is not None else load_renderer_config()
    key = scene_render_option_key(show_id, episode_number, scene_number)
    entry = (cfg.get("sceneRenderOptions") or {}).get(key)
    if not isinstance(entry, dict):
        return True
    if "depthControl" not in entry:
        return True
    return bool(entry.get("depthControl"))


def depth_anything_workflow(
    clay_name: str,
    frame_count: int,
    prefix: str,
    *,
    weights: str,
    input_size: int,
    precision: str,
) -> dict:
    return {
        "1": {
            "class_type": "VHS_LoadVideo",
            "inputs": {
                "video": clay_name,
                "force_rate": FPS,
                "custom_width": CLIP_WIDTH,
                "custom_height": CLIP_HEIGHT,
                "frame_load_cap": frame_count,
                "skip_first_frames": 0,
                "select_every_nth": 1,
            },
        },
        "2": {
            "class_type": "LoadVideoDepthAnythingModel",
            "inputs": {"model": weights},
        },
        "3": {
            "class_type": "VideoDepthAnythingProcess",
            "inputs": {
                "vda_model": ["2", 0],
                "images": ["1", 0],
                "input_size": input_size,
                "max_res": 1280,
                "precision": precision,
            },
        },
        "4": {
            "class_type": "VideoDepthAnythingOutput",
            "inputs": {"depths": ["3", 0], "colormap": "gray"},
        },
        "5": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["4", 0],
                "frame_rate": FPS,
                "loop_count": 0,
                "filename_prefix": prefix,
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 14,
                "save_metadata": True,
                "pingpong": False,
                "save_output": True,
            },
        },
    }


def mux_silent_aac(src: Path, dest: Path) -> None:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found")
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(src),
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-shortest",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "64k",
            str(dest),
        ],
        check=True,
        capture_output=True,
    )


def stage_fit_still(still_path: Path, name: str) -> str:
    COMFY_INPUT.mkdir(parents=True, exist_ok=True)
    fitted = fit_to_clip(Image.open(still_path).convert("RGB"))
    dest = COMFY_INPUT / name
    fitted.save(dest)
    return dest.name


def stage_control_video(src: Path, name: str) -> str:
    COMFY_INPUT.mkdir(parents=True, exist_ok=True)
    dest = COMFY_INPUT / name
    shutil.copy2(src, dest)
    return dest.name


def inject_depth_ltx_graph(
    graph: dict,
    *,
    length: int,
    still_name: str,
    depth_name: str,
    start_strength: float,
    ic_strength: float,
) -> None:
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
    graph["19"]["inputs"]["image"] = still_name
    graph["20"]["inputs"]["strength"] = float(start_strength)
    graph["20"]["inputs"]["width"] = CLIP_WIDTH
    graph["20"]["inputs"]["height"] = CLIP_HEIGHT
    graph["20"]["inputs"]["length"] = length
    graph["30"]["inputs"]["strength_model"] = float(ic_strength)
    graph["31"]["inputs"]["file"] = depth_name


def choose_ltx_workflow(scene: dict, episode: dict, *, spatial: bool) -> tuple[Path, str]:
    if spatial:
        return LTX_DEPTH_WF, "ltx_gemma_api_depth.json"
    return LTX_PLAIN_WF, "ltx_gemma_api.json"


def control_depth_is_fresh(control: Path, clay: Path) -> bool:
    return control.is_file() and control.stat().st_size > 1024 and control.stat().st_mtime >= clay.stat().st_mtime


def ensure_control_depth(
    show_id: str,
    episode_number: int,
    scene: dict,
    *,
    force: bool,
    config: dict,
    queue_prompt,
    free_comfy_models,
    comfy_url: str,
    sample_vram_mib,
) -> dict:
    """Run Depth Anything on clay_24fps → guides/control_depth.mp4. Returns timing stats."""
    if config["controlDepth"] != "depthanything":
        raise SystemExit(f"Unsupported controlDepth={config['controlDepth']!r}")
    clay = clay_24fps_path(show_id, episode_number, scene["sceneNumber"])
    dest = control_depth_mp4_path(show_id, episode_number, scene["sceneNumber"])
    if not clay.is_file():
        raise SystemExit(
            f"Missing clay guide {clay}. Run `pnpm run content:previs` for this scene "
            "before `pnpm run content:generate` (Depth Anything needs clay_24fps.mp4)."
        )
    if not force and control_depth_is_fresh(dest, clay):
        print(f"  Depth control skipped (fresh): {dest}", flush=True)
        return {"skipped": True, "path": dest, "seconds": 0.0, "peak_vram_mib": 0.0}

    frame_count = clip_frame_count(scene["timeRangeSeconds"])
    da = config["depthAnything"]
    COMFY_INPUT.mkdir(parents=True, exist_ok=True)
    clay_name = f"gen_clay_s{int(scene['sceneNumber']):02d}.mp4"
    shutil.copy2(clay, COMFY_INPUT / clay_name)
    prefix = f"gen_da_s{int(scene['sceneNumber']):02d}"
    graph = depth_anything_workflow(
        clay_name,
        frame_count,
        prefix,
        weights=da["weights"],
        input_size=da["inputSize"],
        precision=da["precision"],
    )
    free_comfy_models()
    started = time.time()
    peak = 0.0
    prompt_id = queue_prompt(graph)
    import urllib.request

    while time.time() - started < 1800:
        used = sample_vram_mib()
        if used is not None:
            peak = max(peak, used)
        req = urllib.request.Request(f"{comfy_url}/history/{prompt_id}")
        with urllib.request.urlopen(req) as res:
            history = json.loads(res.read().decode("utf-8"))
        item = history.get(prompt_id)
        if item:
            status = item.get("status") or {}
            for message in status.get("messages") or []:
                if isinstance(message, (list, tuple)) and message and message[0] == "execution_error":
                    raise RuntimeError(json.dumps(message[1])[:4000])
            if item.get("outputs") or status.get("completed"):
                break
        time.sleep(1.0)
    else:
        raise TimeoutError(f"Depth Anything timed out for scene {scene['sceneNumber']}")

    candidates = sorted(
        COMFY_OUTPUT.rglob(f"{prefix}*.mp4"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No Comfy output matching {prefix}*.mp4")
    raw = dest.with_name(dest.stem + "_raw.mp4")
    shutil.copy2(candidates[0], raw)
    mux_silent_aac(raw, dest)
    try:
        raw.unlink()
    except OSError:
        pass
    free_comfy_models()
    seconds = time.time() - started
    print(
        f"  Depth Anything wrote {dest} in {seconds:.1f}s (peak VRAM {peak:.0f} MiB)",
        flush=True,
    )
    return {"skipped": False, "path": dest, "seconds": seconds, "peak_vram_mib": peak}
