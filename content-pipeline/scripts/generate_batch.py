#!/usr/bin/env python3
"""content:frames and content:generate. Qwen-Image-Edit stills, then LTX clips, through ComfyUI.

The words come from ``describe``; the pictures and the facts behind the words
come from ``content:previs`` (guides plus the observation of each start frame).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

from clip_spec import CLIP_HEIGHT, CLIP_WIDTH, FPS, fit_to_clip, ltx_length_for_duration
from depth_control import (
    choose_ltx_workflow,
    ensure_control_depth,
    inject_depth_ltx_graph,
    load_renderer_config,
    scene_depth_control_enabled,
    stage_control_video,
    stage_fit_still,
)
from describe import drawn_prompt, ltx_prompt, still_prompt
from ffmpeg_tools import concat_videos
from pipeline_paths import (
    OUTPUT_DIR,
    clip_path,
    control_depth_mp4_path,
    discover_show_scripts,
    end_still_path,
    episode_video_path,
    guide_path,
    manifest_path,
    start_still_path,
)
from previs import current_previs, load_observation, previs_episode
from render import HEIGHT as PROXY_HEIGHT
from render import WIDTH as PROXY_WIDTH
from world import load_show, look_path, scene_has_spatial_change

ROOT = Path(__file__).resolve().parents[1]
LTX_WORKFLOW_PATH = ROOT / "workflows" / "ltx_gemma_api.json"
SPATIAL_QWEN_WORKFLOW_PATH = ROOT / "workflows" / "qwen_image_edit_spatial.json"
COMFY_INPUT_DIR = ROOT / ".comfyui" / "input"
COMFYUI_URL = "http://127.0.0.1:8188"
_nvidia_smi_ok: bool | None = None

NVIDIA_QUERY_FIELDS = [
    "name",
    "memory.used",
    "memory.total",
    "memory.free",
    "utilization.gpu",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "clocks.current.sm",
    "clocks.max.sm",
    "pstate",
    "clocks_throttle_reasons.gpu_idle",
    "clocks_throttle_reasons.sw_power_cap",
    "clocks_throttle_reasons.hw_slowdown",
    "clocks_throttle_reasons.hw_thermal_slowdown",
    "clocks_throttle_reasons.hw_power_brake_slowdown",
    "clocks_throttle_reasons.sw_thermal_slowdown",
]

THROTTLE_LABELS = {
    "clocks_throttle_reasons.sw_power_cap": "power cap (hit TGP)",
    "clocks_throttle_reasons.hw_slowdown": "hardware slowdown",
    "clocks_throttle_reasons.hw_thermal_slowdown": "thermal (hardware)",
    "clocks_throttle_reasons.hw_power_brake_slowdown": "power brake",
    "clocks_throttle_reasons.sw_thermal_slowdown": "thermal (software)",
}

def load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    whole = int(seconds)
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{seconds:.1f}s"

def format_bytes(n: int) -> str:
    value = float(max(0, n))
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"

def format_gb(n_bytes: int) -> str:
    return f"{n_bytes / (1024 ** 3):.1f} GB"

def _parse_number(raw: str) -> float | None:
    text = str(raw).strip()
    if not text or text.upper() in {"[N/A]", "N/A", "[NOT SUPPORTED]"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None

def fetch_comfy_stats() -> dict | None:
    try:
        with urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=3) as res:
            return json.loads(res.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None

def parse_comfy_gpu(stats: dict | None) -> dict | None:
    devices = (stats or {}).get("devices") or []
    if not devices:
        return None
    device = next((item for item in devices if str(item.get("type", "")).lower() == "cuda"), devices[0])
    total = int(device.get("vram_total") or 0)
    free = int(device.get("vram_free") or 0)
    return {
        "name": device.get("name") or "GPU",
        "vram_total": total,
        "vram_free": free,
        "vram_used": max(0, total - free),
        "torch_vram_total": int(device.get("torch_vram_total") or 0),
        "torch_vram_free": int(device.get("torch_vram_free") or 0),
    }

def nvidia_smi_snapshot() -> dict | None:
    global _nvidia_smi_ok
    if _nvidia_smi_ok is False:
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(NVIDIA_QUERY_FIELDS)}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except FileNotFoundError:
        _nvidia_smi_ok = False
        return None
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    _nvidia_smi_ok = True
    values = [part.strip() for part in result.stdout.strip().splitlines()[0].split(",")]
    if len(values) < len(NVIDIA_QUERY_FIELDS):
        return None
    raw = dict(zip(NVIDIA_QUERY_FIELDS, values))
    throttles = [
        label
        for field, label in THROTTLE_LABELS.items()
        if str(raw.get(field, "")).strip().lower() == "active"
    ]
    used_mb = _parse_number(raw["memory.used"])
    total_mb = _parse_number(raw["memory.total"])
    free_mb = _parse_number(raw["memory.free"])
    return {
        "name": raw["name"],
        "vram_used": int(used_mb * 1024 * 1024) if used_mb is not None else 0,
        "vram_total": int(total_mb * 1024 * 1024) if total_mb is not None else 0,
        "vram_free": int(free_mb * 1024 * 1024) if free_mb is not None else 0,
        "gpu_util": _parse_number(raw["utilization.gpu"]),
        "temp_c": _parse_number(raw["temperature.gpu"]),
        "power_w": _parse_number(raw["power.draw"]),
        "power_limit_w": _parse_number(raw["power.limit"]),
        "clock_mhz": _parse_number(raw["clocks.current.sm"]),
        "clock_max_mhz": _parse_number(raw["clocks.max.sm"]),
        "pstate": raw["pstate"],
        "throttles": throttles,
        "idle": str(raw.get("clocks_throttle_reasons.gpu_idle", "")).strip().lower() == "active",
    }

def windows_shared_gpu_bytes() -> int | None:
    """Peak Shared Usage across GPU adapters (Windows). nvidia-smi does not expose this."""
    if os.name != "nt":
        return None
    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-Counter '\\GPU Adapter Memory(*)\\Shared Usage').CounterSamples | "
                "Measure-Object -Property CookedValue -Maximum | "
                "Select-Object -ExpandProperty Maximum",
            ],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    text = (result.stdout or "").strip()
    if not text:
        return None
    try:
        return int(float(text.splitlines()[-1].strip()))
    except ValueError:
        return None

def nvidia_vram_mib() -> float | None:
    snap = nvidia_smi_snapshot()
    if not snap or not snap.get("vram_used"):
        return None
    return float(snap["vram_used"]) / (1024.0 * 1024.0)

class GpuMonitor:
    def __init__(self) -> None:
        self.samples: list[dict] = []

    def sample(self) -> dict:
        snapshot = {
            "comfy": parse_comfy_gpu(fetch_comfy_stats()),
            "nv": nvidia_smi_snapshot(),
            "shared_bytes": windows_shared_gpu_bytes(),
        }
        self.samples.append(snapshot)
        return snapshot

    def _vram_points(self) -> list[tuple[int, int, int]]:
        points: list[tuple[int, int, int]] = []
        for sample in self.samples:
            for source in (sample.get("nv"), sample.get("comfy")):
                if source and source.get("vram_total"):
                    points.append((source["vram_used"], source["vram_free"], source["vram_total"]))
                    break
        return points

    def summarize(self) -> dict:
        vram = self._vram_points()
        nv_samples = [sample["nv"] for sample in self.samples if sample.get("nv")]
        peak_used = max((used for used, _free, _total in vram), default=0)
        min_free = min((free for _used, free, _total in vram), default=0)
        total = max((total for _used, _free, total in vram), default=0)
        start = vram[0] if vram else (0, 0, 0)
        end = vram[-1] if vram else (0, 0, 0)
        throttles: set[str] = set()
        for sample in nv_samples:
            throttles.update(sample.get("throttles") or [])
        name = None
        if nv_samples:
            name = nv_samples[-1].get("name")
        elif self.samples and self.samples[-1].get("comfy"):
            name = self.samples[-1]["comfy"].get("name")
        utils = [s["gpu_util"] for s in nv_samples if s.get("gpu_util") is not None]
        temps = [s["temp_c"] for s in nv_samples if s.get("temp_c") is not None]
        powers = [s["power_w"] for s in nv_samples if s.get("power_w") is not None]
        shared_vals = [
            int(sample["shared_bytes"])
            for sample in self.samples
            if sample.get("shared_bytes") is not None
        ]
        return {
            "name": name or "GPU",
            "start_used": start[0],
            "peak_used": peak_used,
            "end_used": end[0],
            "min_free": min_free,
            "total": total,
            "peak_util": max(utils) if utils else None,
            "max_temp": max(temps) if temps else None,
            "max_power": max(powers) if powers else None,
            "power_limit": next((s.get("power_limit_w") for s in nv_samples if s.get("power_limit_w")), None),
            "clock": next((s.get("clock_mhz") for s in reversed(nv_samples) if s.get("clock_mhz")), None),
            "clock_max": next((s.get("clock_max_mhz") for s in reversed(nv_samples) if s.get("clock_max_mhz")), None),
            "pstate": next((s.get("pstate") for s in reversed(nv_samples) if s.get("pstate")), None),
            "throttles": sorted(throttles),
            "peak_shared_bytes": max(shared_vals) if shared_vals else None,
        }

def snapshot_line(snapshot: dict) -> str:
    parts: list[str] = []
    source = snapshot.get("nv") or snapshot.get("comfy") or {}
    if source.get("vram_total"):
        parts.append(f"VRAM {format_gb(source['vram_used'])}/{format_gb(source['vram_total'])}")
    nv = snapshot.get("nv") or {}
    if nv.get("gpu_util") is not None:
        parts.append(f"GPU {nv['gpu_util']:.0f}%")
    if nv.get("temp_c") is not None:
        parts.append(f"{nv['temp_c']:.0f}C")
    if nv.get("throttles"):
        parts.append("throttle: " + ", ".join(nv["throttles"]))
    return " | ".join(parts) if parts else "GPU stats unavailable"

def resolution_advice(summary: dict, width: int, height: int, frames: int) -> str:
    total = summary.get("total") or 0
    min_free = summary.get("min_free") or 0
    if not total:
        return "Could not read VRAM, so resolution headroom is unknown."
    voxels = width * height * frames
    free_gb = min_free / (1024 ** 3)
    if free_gb < 1.0:
        return (
            f"VRAM was tight ({format_gb(min_free)} free at peak). "
            "Increasing resolution or frame count is likely to OOM."
        )
    if free_gb < 2.5:
        return (
            f"Limited headroom ({format_gb(min_free)} free at peak of {format_gb(total)}). "
            f"A small bump from {width}x{height} may work; jumping toward 704x1280 or adding many frames "
            f"(currently {frames}) is risky because VRAM scales with width x height x frames ({voxels:,} now)."
        )
    return (
        f"Comfortable headroom ({format_gb(min_free)} free at peak of {format_gb(total)}). "
        f"A moderate resolution bump from {width}x{height} / {frames} frames may be feasible; "
        "raise one axis at a time and watch peak VRAM."
    )

def log_gpu_summary(summary: dict, width: int, height: int, frames: int) -> None:
    print(
        f"  GPU: {summary['name']} | VRAM peak {format_gb(summary['peak_used'])}/"
        f"{format_gb(summary['total'])} used (min free {format_gb(summary['min_free'])}"
        f", start {format_gb(summary['start_used'])})",
        flush=True,
    )
    shared = summary.get("peak_shared_bytes")
    if shared is not None:
        shared_gb = shared / (1024 ** 3)
        flag = " FLAG >1GB shared" if shared > 1024 ** 3 else ""
        print(f"  Shared GPU memory peak: {shared_gb:.2f} GB{flag}", flush=True)
    extras: list[str] = []
    if summary.get("peak_util") is not None:
        extras.append(f"load {summary['peak_util']:.0f}%")
    if summary.get("max_temp") is not None:
        extras.append(f"temp {summary['max_temp']:.0f}C")
    if summary.get("max_power") is not None:
        power = f"{summary['max_power']:.0f}W"
        if summary.get("power_limit"):
            power += f"/{summary['power_limit']:.0f}W"
        extras.append(power)
    if summary.get("clock") is not None:
        clock = f"{summary['clock']:.0f} MHz"
        if summary.get("clock_max"):
            clock += f"/{summary['clock_max']:.0f} MHz"
        extras.append(clock)
    if summary.get("pstate"):
        extras.append(summary["pstate"])
    if extras:
        print(f"  GPU load: {', '.join(extras)}", flush=True)
    if summary.get("throttles"):
        print(f"  Throttle: {', '.join(summary['throttles'])}", flush=True)
    else:
        print("  Throttle: none observed (sampled during this scene)", flush=True)
    print(f"  Resolution: {resolution_advice(summary, width, height, frames)}", flush=True)

def graph_meta(graph: dict) -> dict:
    width, height, length = latent_size(graph)
    steps = None
    frame_rate = None
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type")
        inputs = node.get("inputs") or {}
        if class_type == "LTXVScheduler":
            steps = inputs.get("steps")
        elif class_type == "KSampler":
            candidate = int(inputs.get("steps") or 0)
            steps = candidate if steps is None else max(int(steps), candidate)
        if class_type in {"LTXVConditioning", "VHS_VideoCombine"}:
            frame_rate = inputs.get("frame_rate") or frame_rate
    return {
        "width": width,
        "height": height,
        "length": length,
        "steps": steps,
        "frame_rate": frame_rate,
        "i2v": any(
            isinstance(node, dict) and node.get("class_type") == "LTXVImgToVideo"
            for node in graph.values()
        ),
    }

def execution_seconds_from_history(item: dict | None) -> float | None:
    messages = ((item or {}).get("status") or {}).get("messages") or []
    start = None
    end = None
    for message in messages:
        if not isinstance(message, (list, tuple)) or len(message) < 2:
            continue
        kind, payload = message[0], message[1]
        timestamp = payload.get("timestamp") if isinstance(payload, dict) else None
        if timestamp is None:
            continue
        if kind == "execution_start":
            start = timestamp
        elif kind in {"execution_success", "execution_error", "execution_interrupted"}:
            end = timestamp
    if start is None or end is None:
        return None
    delta = float(end) - float(start)
    if delta > 10_000:
        delta /= 1000.0
    return max(0.0, delta)

def free_comfy_models() -> None:
    payload = json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8")
    req = urllib.request.Request(
        f"{COMFYUI_URL}/free",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=30)
        print("Unloaded idle ComfyUI models so the next stage can fit in VRAM.", flush=True)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"Could not unload ComfyUI models ({exc}); continuing.", flush=True)

def queue_prompt(prompt_graph: dict) -> str:
    payload = json.dumps({"prompt": prompt_graph}).encode("utf-8")
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as res:
        body = json.loads(res.read().decode("utf-8"))
    node_errors = body.get("node_errors") or {}
    if node_errors:
        raise RuntimeError(f"ComfyUI rejected the workflow: {json.dumps(node_errors)}")
    return body["prompt_id"]

def wait_for_output(
    prompt_id: str,
    timeout_s: int = 7200,
    monitor: GpuMonitor | None = None,
    prefer: str = "video",
    require_prefixes: list[str] | None = None,
) -> tuple[list[dict], dict | None]:
    started = time.time()
    last_heartbeat = started
    while time.time() - started < timeout_s:
        snapshot = monitor.sample() if monitor else None
        now = time.time()
        if snapshot and now - last_heartbeat >= 30:
            print(f"  ... {format_duration(now - started)} elapsed | {snapshot_line(snapshot)}", flush=True)
            last_heartbeat = now
        req = urllib.request.Request(f"{COMFYUI_URL}/history/{prompt_id}")
        with urllib.request.urlopen(req) as res:
            history = json.loads(res.read().decode("utf-8"))
        item = history.get(prompt_id)
        if item:
            status = item.get("status") or {}
            for message in status.get("messages") or []:
                if isinstance(message, (list, tuple)) and message and message[0] == "execution_error":
                    payload = message[1] if len(message) > 1 else {}
                    if isinstance(payload, dict):
                        raise RuntimeError(
                            payload.get("exception_message")
                            or payload.get("exception_type")
                            or json.dumps(payload)
                        )
                    raise RuntimeError(str(payload))
            videos: list[dict] = []
            images: list[dict] = []
            meshes: list[dict] = []
            for node_output in (item.get("outputs") or {}).values():
                if not isinstance(node_output, dict):
                    continue
                videos.extend(node_output.get("gifs", []) + node_output.get("videos", []))
                images.extend(node_output.get("images", []))
                meshes.extend(node_output.get("3d", []))
            if prefer == "mesh":
                files = meshes
            elif prefer == "image":
                files = images or videos
            else:
                files = videos or images
            if files:
                if prefer == "video":
                    with_audio = [
                        info
                        for info in files
                        if "-audio." in str(info.get("filename") or "")
                    ]
                    files = with_audio or files
                if require_prefixes and not _has_output_prefixes(files, require_prefixes):
                    if not status.get("completed"):
                        time.sleep(2)
                        continue
                return files, item
            if status.get("completed"):
                raise RuntimeError(
                    f"ComfyUI prompt {prompt_id} finished without a {prefer} output"
                )
        time.sleep(2)
    raise TimeoutError(f"ComfyUI prompt {prompt_id} did not finish in time")

def _output_prefix(filename: str) -> str:
    return str(filename or "")

def _has_output_prefixes(files: list[dict], prefixes: list[str]) -> bool:
    names = [_output_prefix(info.get("filename", "")) for info in files]
    return all(any(name.startswith(f"{prefix}_") for name in names) for prefix in prefixes)

def output_named(files: list[dict], prefix: str) -> dict:
    """Pick the SaveImage whose filename starts with this prefix."""
    matches = [
        info
        for info in files
        if _output_prefix(info.get("filename", "")).startswith(f"{prefix}_")
    ]
    if len(matches) != 1:
        names = [_output_prefix(info.get("filename", "")) for info in files]
        raise RuntimeError(f"Expected one {prefix} image, got {names}")
    return matches[0]

def still_log_path(dest: Path) -> Path:
    return still_inputs_dir(dest) / "log.json"

def still_inputs_dir(dest: Path) -> Path:
    return dest.parent / "inputs" / dest.stem

def begin_generation_log(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    inputs = still_inputs_dir(dest)
    if inputs.is_dir():
        shutil.rmtree(inputs)
    inputs.mkdir(parents=True, exist_ok=True)
    still_log_path(dest).write_text(
        json.dumps({"still": dest.name, "passes": []}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"  Log {still_log_path(dest)}", flush=True)

def download_output(file_info: dict, dest: Path) -> None:
    filename = file_info["filename"]
    subfolder = file_info.get("subfolder", "")
    file_type = file_info.get("type", "output")
    query = urllib.parse.urlencode(
        {"filename": filename, "subfolder": subfolder, "type": file_type}
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"{COMFYUI_URL}/view?{query}") as res, dest.open("wb") as handle:
        shutil.copyfileobj(res, handle)

def stable_seed(*parts: object) -> int:
    """Return a reproducible unsigned 32-bit seed for one render identity."""
    key = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2s(key, digest_size=4).digest(), "big")

def write_clip_generation_log(
    dest: Path,
    *,
    prompt: str,
    graph_name: str,
    seed: int,
    still_path: Path,
    enhance_prompt: bool,
) -> None:
    """Record the final LTX prompt and graph beside the clip, as stills do."""
    begin_generation_log(dest)
    log_path = still_log_path(dest)
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    inputs = still_inputs_dir(dest)
    inputs.mkdir(parents=True, exist_ok=True)
    recorded: list[dict] = []
    if still_path.is_file():
        copied = inputs / f"ltx_start{still_path.suffix.lower() or '.png'}"
        shutil.copy2(still_path, copied)
        recorded.append({"role": "start still", "source": str(still_path), "file": copied.name})
    else:
        recorded.append({"role": "start still", "missing": True})
    payload["clip"] = dest.name
    payload["passes"] = [
        {
            "pass": "ltx",
            "prompt": prompt,
            "graph": graph_name,
            "seed": seed,
            "enhance_prompt": enhance_prompt,
            "gemma_api": (
                "GemmaAPITextEncode encodes the authored prompt as-is when "
                "enhance_prompt is false; it does not rewrite or enhance the text."
            ),
            "images": recorded,
        }
    ]
    log_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

def write_solid_png(path: Path, color: tuple[int, int, int], width: int = 768, height: int = 1360) -> None:
    pixel = bytes(color)
    raw = b"".join(b"\x00" + (pixel * width) for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    )

def present(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 1024

def clone_workflow(workflow_template: dict) -> dict:
    raw = workflow_template["prompt"] if "prompt" in workflow_template else workflow_template
    return json.loads(json.dumps(raw))

def inject_seed(graph: dict, seed: int) -> None:
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type")
        inputs = node.setdefault("inputs", {})
        if class_type == "RandomNoise":
            inputs["noise_seed"] = seed
        elif class_type == "KSampler":
            inputs["seed"] = seed

def stage_start_still(image_path: Path, *, fit_clip: bool = False) -> str:
    COMFY_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = COMFY_INPUT_DIR / image_path.name
    if fit_clip:
        fit_to_clip(Image.open(image_path).convert("RGB")).save(dest)
        return dest.name
    if dest.resolve() != image_path.resolve():
        shutil.copy2(image_path, dest)
    return dest.name

def stage_named_image(image_path: Path, prefix: str) -> str:
    COMFY_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = COMFY_INPUT_DIR / f"{prefix}_{image_path.name}"
    shutil.copy2(image_path, dest)
    return dest.name

def inject_start_frame(graph: dict, image_name: str, strength: float | None = None) -> None:
    width, height, length = latent_size(graph)
    width = CLIP_WIDTH
    height = CLIP_HEIGHT
    i2v = float(strength if strength is not None else load_renderer_config()["ltxStartStrength"])
    graph["19"] = {
        "inputs": {"image": image_name},
        "class_type": "LoadImage",
        "_meta": {"title": "Scene start frame"},
    }
    graph["20"] = {
        "inputs": {
            "positive": ["16", 0],
            "negative": ["16", 1],
            "vae": ["9", 0],
            "image": ["19", 0],
            "width": width,
            "height": height,
            "length": length,
            "batch_size": 1,
            "strength": i2v,
        },
        "class_type": "LTXVImgToVideo",
        "_meta": {"title": "Animate from start frame"},
    }
    concat = graph.get("24")
    if isinstance(concat, dict) and concat.get("class_type") == "LTXVConcatAVLatent":
        concat.setdefault("inputs", {})["video_latent"] = ["20", 2]
    else:
        graph["24"] = {
            "inputs": {
                "video_latent": ["20", 2],
                "audio_latent": ["23", 0],
            },
            "class_type": "LTXVConcatAVLatent",
            "_meta": {"title": "Joint AV latent"},
        }
    # Scheduler/shift math needs the 5D video latent. The sampler must get the
    # concatenated AV latent or VHS writes a silent MP4.
    for node_id, key in (("15", "latent"), ("18", "latent")):
        node = graph.get(node_id)
        if isinstance(node, dict):
            node.setdefault("inputs", {})[key] = ["20", 2]
    sampler = graph.get("6")
    if isinstance(sampler, dict):
        sampler.setdefault("inputs", {})["latent_image"] = ["24", 0]

def _qwen_encoder(graph: dict, title: str) -> dict | None:
    for node in graph.values():
        if not isinstance(node, dict) or node.get("class_type") not in {
            "TextEncodeQwenImageEditPlus",
            "TextEncodeQwenBackdrop",
        }:
            continue
        if str((node.get("_meta") or {}).get("title", "")) == title:
            return node
    return None

def inject_qwen_prompt(graph: dict, prompt: str, title: str = "Positive instruction") -> None:
    target = _qwen_encoder(graph, title)
    if target is None and title == "Positive instruction":
        for node in graph.values():
            if not isinstance(node, dict) or node.get("class_type") != "TextEncodeQwenImageEditPlus":
                continue
            node_title = str((node.get("_meta") or {}).get("title", ""))
            if node_title in {"Negative instruction", "Structure instruction"}:
                continue
            target = node
            break
    if target is None:
        raise RuntimeError(f"Could not find the Qwen {title} node")
    target.setdefault("inputs", {})["prompt"] = prompt

def latent_size(graph: dict) -> tuple[int, int, int]:
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type")
        inputs = node.setdefault("inputs", {})
        if class_type == "EmptyLTXVLatentVideo":
            return int(inputs["width"]), int(inputs["height"]), int(inputs["length"])
        if class_type == "EmptySD3LatentImage":
            return int(inputs["width"]), int(inputs["height"]), 1
        if class_type == "ImageCropToMask":
            return int(inputs["width"]), int(inputs["height"]), 1
    for node in graph.values():
        if isinstance(node, dict) and node.get("class_type") == "VAEEncode":
            return PROXY_WIDTH, PROXY_HEIGHT, 1
    raise RuntimeError("Could not find a latent size in the ComfyUI workflow")

def workflow_frame_rate(graph: dict) -> float:
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") in {"LTXVConditioning", "VHS_VideoCombine"}:
            rate = (node.get("inputs") or {}).get("frame_rate")
            if rate:
                return float(rate)
    return 24.0

def inject_scene_length(graph: dict, duration_seconds: float | None = None, length: int | None = None) -> int:
    if length is None:
        length = ltx_length_for_duration(float(duration_seconds or 1), workflow_frame_rate(graph) or FPS)
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type")
        inputs = node.setdefault("inputs", {})
        if class_type == "EmptyLTXVLatentVideo":
            inputs["length"] = length
            inputs["width"] = CLIP_WIDTH
            inputs["height"] = CLIP_HEIGHT
        elif class_type == "LTXVImgToVideo":
            inputs["length"] = length
            inputs["width"] = CLIP_WIDTH
            inputs["height"] = CLIP_HEIGHT
        elif class_type == "LTXVEmptyLatentAudio":
            inputs["frames_number"] = length
    return length

def inject_prompt(workflow: dict, prompt: str, api_key: str) -> dict:
    raw = workflow["prompt"] if "prompt" in workflow else workflow
    graph = json.loads(json.dumps(raw))
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type", "")
        title = str((node.get("_meta") or {}).get("title", ""))
        if class_type in {"GemmaAPITextEncode", "CLIPTextEncode", "CLIPTextEncodeLTXV"} or "Gemma" in title:
            node.setdefault("inputs", {})["prompt" if class_type == "GemmaAPITextEncode" else "text"] = prompt
            if class_type == "GemmaAPITextEncode" and api_key:
                node["inputs"]["api_key"] = api_key
                node["inputs"]["ckpt_name"] = "ltx-2.3-22b-distilled-api-id.safetensors"
                node["inputs"]["enhance_prompt"] = False
            return graph
    raise RuntimeError("Could not find a text-conditioning node in the ComfyUI workflow")

def inject_dialogue_multimodal_guider(graph: dict) -> None:
    """Strengthen joint audio/video coherence for speaking shots."""
    graph["29"] = {
        "inputs": {
            "modality": "VIDEO",
            "cfg": 1.0,
            "stg": 0.0,
            "perturb_attn": False,
            "rescale": 0.0,
            "modality_scale": 3.0,
            "skip_step": 0,
            "cross_attn": True,
        },
        "class_type": "GuiderParameters",
        "_meta": {"title": "Dialogue video guidance"},
    }
    graph["30"] = {
        "inputs": {
            "modality": "AUDIO",
            "cfg": 1.0,
            "stg": 0.0,
            "perturb_attn": False,
            "rescale": 0.0,
            "modality_scale": 3.0,
            "skip_step": 0,
            "cross_attn": True,
            "parameters": ["29", 0],
        },
        "class_type": "GuiderParameters",
        "_meta": {"title": "Dialogue audio guidance"},
    }
    graph["17"] = {
        "inputs": {
            "model": ["18", 0],
            "positive": ["16", 0],
            "negative": ["16", 1],
            "parameters": ["30", 0],
            "skip_blocks": "",
        },
        "class_type": "MultimodalGuider",
        "_meta": {"title": "Dialogue audio-video coherence"},
    }

def execute_queued_graph(
    graph: dict,
    dest: Path,
    prefer: str,
    mode: str,
    also: list[tuple[str, Path]] | None = None,
) -> None:
    meta = graph_meta(graph)
    clip_seconds = None
    if meta["frame_rate"]:
        clip_seconds = meta["length"] / float(meta["frame_rate"])
    size_bit = f"{meta['width']}x{meta['height']}"
    if prefer == "video" and clip_seconds is not None:
        size_bit += f", {meta['length']} frames (~{clip_seconds:.2f}s)"
    print(
        f"  Graph: {size_bit}"
        f"{f', {meta['steps']} steps' if meta['steps'] else ''}"
        f"{f' @ {meta['frame_rate']} fps' if meta['frame_rate'] else ''}"
        f", {mode}",
        flush=True,
    )
    monitor = GpuMonitor()
    monitor.sample()
    started = time.time()
    prompt_id = queue_prompt(graph)
    prefixes = [prefix for prefix, _path in also or []]
    if also:
        prefixes.append("reelshort_start")
    outputs, history_item = wait_for_output(
        prompt_id,
        monitor=monitor,
        prefer=prefer,
        require_prefixes=prefixes or None,
    )
    for prefix, path in also or []:
        download_output(output_named(outputs, prefix), path)
        print(f"  Wrote {path} ({format_bytes(path.stat().st_size)})", flush=True)
    final = output_named(outputs, "reelshort_start") if also else outputs[0]
    download_output(final, dest)
    monitor.sample()
    elapsed = time.time() - started
    timing = f"in {format_duration(elapsed)}"
    comfy_elapsed = execution_seconds_from_history(history_item)
    if comfy_elapsed is not None:
        timing += f" (ComfyUI execution {format_duration(comfy_elapsed)})"
    print(f"  Finished {timing}", flush=True)
    print(f"  Wrote {dest} ({format_bytes(dest.stat().st_size)})", flush=True)
    log_gpu_summary(monitor.summarize(), meta["width"], meta["height"], meta["length"])

def confirm_vram_released(label: str, *, max_mib: float = 4096.0) -> float | None:
    """After /free, sample nvidia-smi and warn if Depth Anything may still be resident."""
    time.sleep(1.0)
    used = nvidia_vram_mib()
    if used is None:
        print(f"  VRAM after {label}: unavailable", flush=True)
        return None
    print(f"  VRAM after {label}: {used:.0f} MiB", flush=True)
    if used > max_mib:
        print(
            f"  WARNING: VRAM still high after unload ({used:.0f} MiB > {max_mib:.0f} MiB); "
            "continuing but LTX may OOM.",
            flush=True,
        )
    return used


def append_generation_log(dest: Path, pass_name: str, prompt: str, images: list[tuple[str, Path]], details: dict) -> None:
    """Record a pass: its prompt, a copy of each picture, and why each description was sent or not."""
    log_path = still_log_path(dest)
    if not log_path.is_file():
        begin_generation_log(dest)
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    inputs = still_inputs_dir(dest)
    recorded = []
    for role, source in images:
        copied = inputs / f"{pass_name}_{role.lower()}{source.suffix.lower()}"
        shutil.copy2(source, copied)
        recorded.append({"role": role, "source": str(source), "file": copied.name})
    payload.setdefault("passes", []).append({"pass": pass_name, "prompt": prompt, "images": recorded, **details})
    log_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


# Qwen spatial graph: node 6 is picture 1, nodes 40-42 are pictures 2-4, all
# into one four-picture encoder. The identity face pass is not used.
_PICTURE_NODES = (("40", "image2"), ("41", "image3"), ("42", "image4"))
_FACE_PASS_NODES = ("20", "21", "22", "23", "24", "25", "30", "31", "70")


def qwen_pass(
    dest: Path,
    prompt: str,
    pictures: list[tuple[str, Path]],
    seed: int,
    mode: str,
    keep: tuple[Path, Path] | None = None,
    matte: Path | None = None,
) -> None:
    """One Qwen-Image-Edit draw from up to four pictures. The first sets the canvas size.

    ``keep`` is (image, mask): the image's pixels where the mask is black are
    kept exactly, and only the white area is drawn. ``matte`` also writes the
    drawn subject's soft outline (background removal on the result).
    """
    graph = clone_workflow(json.loads(SPATIAL_QWEN_WORKFLOW_PATH.read_text(encoding="utf-8")))
    inject_seed(graph, seed)
    encoder = graph["7"]
    encoder["class_type"] = "TextEncodeQwenBackdrop"
    inputs = encoder.setdefault("inputs", {})
    inputs["prompt"] = prompt
    staged = [(stage_named_image(path, f"{dest.stem}_{title.lower()}"), title) for title, path in pictures]
    graph["6"]["inputs"]["image"] = staged[0][0]
    graph["6"]["_meta"] = {"title": staged[0][1]}
    inputs["image1"] = ["6", 0]
    for node_id, key in _PICTURE_NODES:
        inputs.pop(key, None)
        graph.pop(node_id, None)
    for (node_id, key), (name, title) in zip(_PICTURE_NODES, staged[1:]):
        graph[node_id] = {"inputs": {"image": name}, "class_type": "LoadImage", "_meta": {"title": title}}
        inputs[key] = [node_id, 0]
    graph["12"]["inputs"]["images"] = ["11", 0]
    for node_id in _FACE_PASS_NODES:
        graph.pop(node_id, None)
    if keep is not None:
        image, mask = keep
        graph["51"] = {"inputs": {"image": stage_named_image(image, f"{dest.stem}_keep")}, "class_type": "LoadImage"}
        graph["52"] = {"inputs": {"pixels": ["51", 0], "vae": ["3", 0]}, "class_type": "VAEEncode"}
        graph["53"] = {"inputs": {"image": stage_named_image(mask, f"{dest.stem}_draw")}, "class_type": "LoadImage"}
        graph["54"] = {"inputs": {"image": ["53", 0], "channel": "red"}, "class_type": "ImageToMask"}
        graph["55"] = {"inputs": {"samples": ["52", 0], "mask": ["54", 0]}, "class_type": "SetLatentNoiseMask"}
        graph["10"]["inputs"]["latent_image"] = ["55", 0]
    also = None
    if matte is not None:
        graph["60"] = {"inputs": {"bg_removal_name": "birefnet.safetensors"}, "class_type": "LoadBackgroundRemovalModel"}
        graph["61"] = {"inputs": {"bg_removal_model": ["60", 0], "image": ["11", 0]}, "class_type": "RemoveBackground"}
        graph["62"] = {"inputs": {"mask": ["61", 0]}, "class_type": "MaskToImage"}
        graph["63"] = {"inputs": {"filename_prefix": "reelshort_matte", "images": ["62", 0]}, "class_type": "SaveImage"}
        also = [("reelshort_matte", matte)]
    print(f"  Prompt: {prompt}", flush=True)
    print(f"  Graph seed {seed}", flush=True)
    execute_queued_graph(graph, dest, prefer="image", mode=mode, also=also)
    if keep is not None:
        # The sampler works in latent space; put the kept pixels back exactly.
        drawn = Image.open(dest).convert("RGB")
        kept = Image.open(keep[0]).convert("RGB").resize(drawn.size)
        hold = Image.open(keep[1]).convert("L").resize(drawn.size).point(lambda value: 255 - value)
        Image.composite(kept, drawn, hold).save(dest)


# Pixels of a pasted entity this close to its edge are redrawn, so it sits in the shot.
_BLEND_PIXELS = 3


# Fire and smoke are cut out by how far they are from the drawing's plain background:
# this close counts as background, and the cover reaches full over this much more.
_EFFECT_BACKGROUND_DISTANCE = 12.0
_EFFECT_RAMP_DISTANCE = 60.0


def effect_alpha(picture: Image.Image) -> np.ndarray:
    """Cover of an effect drawn on a plain background: opaque where it differs, clear where it does not.

    An object matte cuts solid outlines and drops fire, smoke, and steam; on a
    known plain background their own difference from it is their cover.
    """
    rgb = np.asarray(picture.convert("RGB"), dtype=np.float32)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    distance = np.linalg.norm(rgb - np.median(border, axis=0), axis=2)
    return np.clip((distance - _EFFECT_BACKGROUND_DISTANCE) / _EFFECT_RAMP_DISTANCE, 0.0, 1.0)


def paste_drawn(
    color: Path, drawn: list[tuple[Path, Path, Path, list[float], Path | None]], dest: Path, draw_mask: Path
) -> Path:
    """The shot's color guide with each drawn person, prop, and landmark laid in.

    Each drawing is scaled down into its window (never up: the window lies inside
    the frame) and blended by its own soft outline times where the shot shows it,
    so nothing drawn can land where the 3D scene has no such thing. Inside its
    effects' area (the fifth item, or None) the outline is also taken from the
    drawing's difference from its background, which keeps the fire the matte drops.
    ``draw_mask`` is white wherever the shot still has to be drawn.
    """
    from scipy import ndimage

    shot = Image.open(color).convert("RGB")
    held = np.zeros((shot.height, shot.width), dtype=np.float32)
    for picture, matte, shown, crop, effects in drawn:
        x0, y0, x1, y1 = crop
        size = (max(1, round(x1 - x0)), max(1, round(y1 - y0)))
        corner = (round(x0), round(y0))

        def placed(image: Image.Image, mode: str) -> Image.Image:
            canvas = Image.new(mode, shot.size, 0)
            canvas.paste(image.convert(mode).resize(size, Image.Resampling.LANCZOS), corner)
            return canvas

        drawing = Image.open(picture)
        layer = placed(drawing, "RGB")
        cover = np.asarray(placed(Image.open(matte), "L"), dtype=np.float32) / 255.0
        if effects is not None:
            differs = placed(Image.fromarray(np.rint(effect_alpha(drawing) * 255).astype(np.uint8)), "L")
            area = np.asarray(Image.open(effects).convert("L"), dtype=np.float32) / 255.0
            cover = np.maximum(cover, np.asarray(differs, dtype=np.float32) / 255.0 * area)
        cover *= np.asarray(Image.open(shown).convert("L"), dtype=np.float32) / 255.0
        shot = Image.composite(layer, shot, Image.fromarray(np.rint(cover * 255).astype(np.uint8)))
        held = np.maximum(held, cover)
    shot.save(dest)
    kept = ndimage.binary_erosion(held > 0.5, iterations=_BLEND_PIXELS)
    Image.fromarray(np.where(kept, 0, 255).astype(np.uint8)).save(draw_mask)
    return dest


def paint_order(observation: dict) -> list[str]:
    """Drawn entities, farthest first, so a nearer one is laid over a farther one.

    Solid overlaps are already cut by each ``shown`` mask; the order decides what
    has no surface to cut by: fire in front of a person behind it. An entity seen
    only through its effects (no pixel of its own) counts as farthest.
    """
    drawn = [(entity_id, entry) for entity_id, entry in observation["entities"].items() if "crop" in entry]
    drawn.sort(key=lambda item: -(item[1].get("depth") or float("inf")))
    return [entity_id for entity_id, _entry in drawn]


def render_still(show: dict, episode: dict, scene: dict, label: str, dest: Path, seed: int) -> None:
    """One scene still: everything in the shot drawn alone with its effects and pasted in, the empty space drawn around it.

    A drawn entity sees only its own guides (the shot camera magnified onto it),
    its plate, and its own words, so it never borrows another's description and
    is drawn at full size however far away it is. The shot pass keeps the pasted
    pixels and draws only the space around them.
    """
    ids = (show["id"], episode["episodeNumber"], scene["sceneNumber"])
    observation = load_observation(*ids, label)
    guide = lambda kind: guide_path(*ids, label, kind)
    begin_generation_log(dest)
    inputs = still_inputs_dir(dest)
    drawn = []
    for entity_id in paint_order(observation):
        entry = observation["entities"][entity_id]
        files = [
            ("Depth", guide(f"drawn_{entity_id}_depth")),
            ("OwnColor", guide(f"drawn_{entity_id}_color")),
            ("Edges", guide(f"drawn_{entity_id}_edges")),
            ("Appearance", look_path(show, entry["kind"], entity_id, scene["locationId"])),
        ]
        missing = [str(path) for _title, path in files if not present(path)]
        if missing:
            raise SystemExit(f"Missing {missing[0]}. Run `pnpm run content:previs` (and content:plates) first.")
        prompt = drawn_prompt(show, scene, entry["kind"], entity_id, [title for title, _path in files])
        picture = inputs / f"drawn_{entity_id}.png"
        matte = inputs / f"drawn_{entity_id}_matte.png"
        print(f"  Drawn alone: {entity_id}", flush=True)
        qwen_pass(
            picture, prompt, files, stable_seed(seed, "drawn", entity_id), f"Qwen {entry['kind']} ({entity_id})", matte=matte
        )
        append_generation_log(dest, f"drawn_{entity_id}", prompt, files, {"id": entity_id, "kind": entry["kind"]})
        effects = guide(f"drawn_{entity_id}_effects")
        drawn.append((picture, matte, guide(f"drawn_{entity_id}_shown"), entry["crop"], effects if effects.is_file() else None))
    draw_mask = inputs / "draw_mask.png"
    composite = paste_drawn(guide("color"), drawn, inputs / "composite.png", draw_mask)
    files = [("Depth", guide("depth")), ("Composite", composite), ("Edges", guide("edges"))]
    prompt, details = still_prompt(show, scene, observation, [title for title, _path in files], len(drawn))
    keep = (composite, draw_mask) if drawn else None
    qwen_pass(dest, prompt, files, seed, f"Qwen scene still ({label})", keep)
    append_generation_log(dest, "still", prompt, files, details)


def scene_needs_end_still(episode: dict, scene: dict, config: dict) -> bool:
    """scene_XX_end.png exists only when prompts.endStill is true and the shot changes."""
    return bool(config.get("endStill")) and scene_has_spatial_change(episode, scene)


def generate_frames(show: dict, episode: dict, scene: dict, force: bool, seed: int, renderer: dict) -> bool:
    """Previs this scene, stop on a script/scene mismatch, then draw the still(s)."""
    ids = (show["id"], episode["episodeNumber"], scene["sceneNumber"])
    targets = [("start", start_still_path(*ids))]
    if scene_needs_end_still(episode, scene, renderer):
        targets.append(("end", end_still_path(*ids)))
    if all(present(path) for _label, path in targets) and not force:
        print(f"  Skipped (already present): {targets[0][1]}", flush=True)
        return False
    errors = current_previs(show, episode, scene)
    if errors is None:
        errors, _written = previs_episode(show, episode, scene["sceneNumber"])
    else:
        print("  Previs is current; reusing its guides.", flush=True)
    if errors:
        raise SystemExit("The script does not match its 3D scene:\n- " + "\n- ".join(errors))
    for label, path in targets:
        if force or not present(path):
            render_still(show, episode, scene, label, path, seed)
    return True


def generate_clip(show: dict, episode: dict, scene: dict, seed: int, renderer: dict, use_depth: bool) -> Path:
    ids = (show["id"], episode["episodeNumber"], scene["sceneNumber"])
    still = start_still_path(*ids)
    dest = clip_path(*ids)
    prompt = ltx_prompt(show, scene, load_observation(*ids, "start"))
    wf_path, graph_name = choose_ltx_workflow(scene, episode, spatial=use_depth)
    graph = inject_prompt(load_json(wf_path), prompt, os.environ.get("LTXV_API_KEY", ""))
    start, finish = (float(value) for value in scene["timeRangeSeconds"])
    length = inject_scene_length(graph, duration_seconds=finish - start)
    inject_seed(graph, seed)
    start_strength = float(renderer["ltxStartStrength"])
    ic_strength = float(renderer["ltxIcLoRAStrength"])
    number = int(scene["sceneNumber"])
    if use_depth:
        control = control_depth_mp4_path(*ids)
        if not present(control):
            raise SystemExit(f"Missing control depth {control}. Depth pass should have written it.")
        inject_depth_ltx_graph(
            graph,
            length=length,
            still_name=stage_fit_still(still, f"ltx_start_s{number:02d}.png"),
            depth_name=stage_control_video(control, f"ltx_depth_s{number:02d}.mp4"),
            start_strength=start_strength,
            ic_strength=ic_strength,
        )
    else:
        if scene.get("speakerId"):
            inject_dialogue_multimodal_guider(graph)
        inject_start_frame(graph, stage_start_still(still, fit_clip=True), strength=start_strength)
    if os.environ.get("LTX_TILED_VAE", "").strip() in {"1", "true", "yes"}:
        node = graph.get("8")
        if isinstance(node, dict) and node.get("class_type") == "VAEDecode":
            node["class_type"] = "VAEDecodeTiled"
            node.setdefault("inputs", {}).update({"tile_size": 512, "overlap": 64})
    graph["10"]["inputs"]["filename_prefix"] = f"reelshort_{show['id']}_e{episode['episodeNumber']}_s{number:02d}"
    write_clip_generation_log(
        dest, prompt=prompt, graph_name=graph_name, seed=seed, still_path=still, enhance_prompt=False
    )
    print(f"  Prompt: {prompt}", flush=True)
    print(
        f"  Graph={graph_name} depth={use_depth} seed={seed} i2v={start_strength} "
        f"ic={ic_strength if use_depth else 'n/a'} size={CLIP_WIDTH}x{CLIP_HEIGHT} frames={length}",
        flush=True,
    )
    execute_queued_graph(graph, dest, prefer="video", mode=f"LTX scene {number:02d}")
    free_comfy_models()
    return dest


def generate_show(show: dict, stage: str, partial: bool = False, force: bool = False, seed_override: int | None = None) -> None:
    show_id = show["id"]
    started = time.time()
    renderer = load_renderer_config()
    for episode in show["episodes"]:
        number = episode["episodeNumber"]
        manifest = manifest_path(show_id, number)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            json.dumps(
                {
                    "series": show_id,
                    "episodeNumber": number,
                    "title": episode["title"],
                    "isFree": episode["isFree"],
                    "coinCost": episode["coinCost"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        def seed(scene: dict, kind: str) -> int:
            return seed_override if seed_override is not None else stable_seed(show_id, number, scene["sceneNumber"], kind)

        if stage == "frames":
            for scene in episode["scenes"]:
                print(f"Scene {scene['sceneNumber']:02d} still...", flush=True)
                generate_frames(show, episode, scene, force, seed(scene, "frame"), renderer)
            continue

        pending = []
        for scene in episode["scenes"]:
            ids = (show_id, number, scene["sceneNumber"])
            if present(clip_path(*ids)) and not force:
                print(f"Scene {scene['sceneNumber']:02d}: skipped (already present)", flush=True)
                continue
            if not present(start_still_path(*ids)):
                raise SystemExit(
                    f"Missing start still {start_still_path(*ids)}. Run `pnpm run content:frames`, review it, then rerun."
                )
            spatial = scene_has_spatial_change(episode, scene)
            use_depth = spatial and scene_depth_control_enabled(*ids, renderer)
            if spatial and not use_depth:
                print(f"Scene {scene['sceneNumber']:02d}: depth control disabled by override", flush=True)
            pending.append((scene, use_depth))
        depth_scenes = [scene for scene, use_depth in pending if use_depth]
        for scene in depth_scenes:
            print(f"Depth pass for scene {scene['sceneNumber']:02d}...", flush=True)
            ensure_control_depth(
                show_id,
                number,
                scene,
                force=force,
                config=renderer,
                queue_prompt=queue_prompt,
                free_comfy_models=free_comfy_models,
                comfy_url=COMFYUI_URL,
                sample_vram_mib=nvidia_vram_mib,
            )
        if depth_scenes:
            free_comfy_models()
            confirm_vram_released("Depth Anything unload")
        for scene, use_depth in pending:
            print(f"Scene {scene['sceneNumber']:02d} clip...", flush=True)
            generate_clip(show, episode, scene, seed(scene, "video"), renderer, use_depth)
        ordered = [clip_path(show_id, number, scene["sceneNumber"]) for scene in episode["scenes"]]
        if not partial and all(present(path) for path in ordered):
            episode_mp4 = episode_video_path(show_id, number)
            concat_videos(ordered, episode_mp4, loudness_target_lufs=renderer.get("episodeLoudnessTargetLufs"))
            print(f"Wrote {episode_mp4} ({format_bytes(episode_mp4.stat().st_size)})", flush=True)
    if stage == "frames":
        print(
            f"Stills for {show_id} are in {OUTPUT_DIR / 'frames' / show_id}. Review each scene_*_start.png; "
            "delete one and rerun `pnpm run content:frames` to redraw it. Then run `pnpm run content:generate`.",
            flush=True,
        )
    print(f"Show {show_id} {stage} finished in {format_duration(time.time() - started)}", flush=True)


def stage_needs_comfy(shows: list[dict], stage: str) -> bool:
    renderer = load_renderer_config()
    for show in shows:
        for episode in show["episodes"]:
            for scene in episode["scenes"]:
                ids = (show["id"], episode["episodeNumber"], scene["sceneNumber"])
                if stage == "video":
                    needed = [clip_path(*ids)]
                else:
                    needed = [start_still_path(*ids)]
                    if scene_needs_end_still(episode, scene, renderer):
                        needed.append(end_still_path(*ids))
                if not all(present(path) for path in needed):
                    return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Render scene stills (Qwen-Image-Edit) or clips (LTX) via ComfyUI.")
    parser.add_argument("--stage", choices=("frames", "video"), default="video")
    parser.add_argument("--show", help="Render only this show id")
    parser.add_argument("--episode", type=int, help="Render only this episode number")
    parser.add_argument("--scene", type=int, help="Render only this scene number (requires --episode)")
    parser.add_argument("--force", action="store_true", help="Regenerate an existing selected scene (requires --scene)")
    parser.add_argument("--seed", type=int, help="Override the deterministic seed for one selected scene (requires --scene)")
    args = parser.parse_args()
    if args.scene is not None and args.episode is None:
        parser.error("--scene requires --episode")
    if args.force and args.scene is None:
        parser.error("--force requires --scene")
    if args.seed is not None and args.scene is None:
        parser.error("--seed requires --scene")
    if args.seed is not None and not 0 <= args.seed <= 2**32 - 1:
        parser.error("--seed must be between 0 and 4294967295")
    load_dotenv()
    scripts = discover_show_scripts(args.show)
    if not scripts:
        raise SystemExit(f"No show JSON files found{f' for show {args.show!r}' if args.show else ''}")
    shows = [load_show(path) for path in scripts]
    for show in shows:
        if args.episode is not None:
            show["episodes"] = [episode for episode in show["episodes"] if episode["episodeNumber"] == args.episode]
            if not show["episodes"]:
                raise SystemExit(f"Show {show['id']!r} has no episode {args.episode}")
        if args.scene is not None:
            scenes = [scene for scene in show["episodes"][0]["scenes"] if scene["sceneNumber"] == args.scene]
            if not scenes:
                raise SystemExit(f"Show {show['id']!r} episode {args.episode} has no scene {args.scene}")
            show["episodes"][0]["scenes"] = scenes
    if args.force or stage_needs_comfy(shows, args.stage):
        if args.stage == "video" and not os.environ.get("LTXV_API_KEY"):
            raise SystemExit("Missing LTXV_API_KEY in content-pipeline/.env")
        try:
            urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=3)
        except urllib.error.URLError as exc:
            raise SystemExit(f"ComfyUI is not reachable at {COMFYUI_URL}. Start it with `pnpm run content:comfy`.") from exc
        free_comfy_models()
        idle = GpuMonitor()
        idle.sample()
        print(f"ComfyUI ready at {COMFYUI_URL} | stage={args.stage} | idle {snapshot_line(idle.samples[0])}", flush=True)
    else:
        print(f"All scene files already present | stage={args.stage} | skipping ComfyUI", flush=True)
    for show in shows:
        generate_show(show, args.stage, partial=args.scene is not None, force=args.force, seed_override=args.seed)


if __name__ == "__main__":
    main()
