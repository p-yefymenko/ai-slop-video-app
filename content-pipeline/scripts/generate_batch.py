#!/usr/bin/env python3
"""Render episode scripts via ComfyUI: Qwen-Image-Edit stills, then LTX video."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts_input"
CHARACTERS_DIR = ROOT / "characters"
OUTPUT_DIR = ROOT / "output"
LTX_WORKFLOW_PATH = ROOT / "workflows" / "ltx_gemma_api.json"
QWEN_WORKFLOW_PATH = ROOT / "workflows" / "qwen_image_edit.json"
COMFY_INPUT_DIR = ROOT / ".comfyui" / "input"
COMFYUI_URL = "http://127.0.0.1:8188"
I2V_STRENGTH = 0.7  # official LTX image-to-video default
MAX_QWEN_REFS = 3
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
_nvidia_smi_ok: bool | None = None


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


class GpuMonitor:
    def __init__(self) -> None:
        self.samples: list[dict] = []

    def sample(self) -> dict:
        snapshot = {"comfy": parse_comfy_gpu(fetch_comfy_stats()), "nv": nvidia_smi_snapshot()}
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
        if class_type in {"LTXVScheduler", "KSampler"}:
            steps = inputs.get("steps")
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
            for node_output in (item.get("outputs") or {}).values():
                videos.extend(node_output.get("gifs", []) + node_output.get("videos", []))
                images.extend(node_output.get("images", []))
            files = (images or videos) if prefer == "image" else (videos or images)
            if files:
                return files, item
            if status.get("completed"):
                raise RuntimeError(
                    f"ComfyUI prompt {prompt_id} finished without a video or image output"
                )
        time.sleep(2)
    raise TimeoutError(f"ComfyUI prompt {prompt_id} did not finish in time")


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


def load_character_registry() -> dict[str, dict]:
    registry: dict[str, dict] = {}
    if not CHARACTERS_DIR.exists():
        return registry
    for path in sorted(CHARACTERS_DIR.glob("*.json")):
        data = load_json(path)
        character_id = data.get("id")
        if not character_id:
            raise SystemExit(f"Character file {path} is missing an id")
        if character_id in registry:
            raise SystemExit(f"Duplicate character id {character_id!r} in {path}")
        registry[character_id] = {**data, "_dir": path.parent}
    return registry


def resolve_scene_characters(scene: dict, registry: dict[str, dict]) -> list[dict]:
    resolved: list[dict] = []
    for character_id in scene.get("characterIds") or []:
        character = registry.get(character_id)
        if character is None:
            raise SystemExit(
                f"Unknown characterId {character_id!r}. Add {CHARACTERS_DIR / f'{character_id}.json'}"
            )
        image_name = character.get("referenceImage")
        if not image_name:
            raise SystemExit(f"Character {character_id} is missing referenceImage")
        image_path = character["_dir"] / image_name
        if not image_path.is_file():
            raise SystemExit(
                f"Missing reference image for {character_id}: {image_path}. "
                "Generate a still and save it next to the character JSON before content:frames."
            )
        if not character.get("promptBlock", "").strip():
            raise SystemExit(f"Character {character_id} is missing promptBlock")
        resolved.append({**character, "image_path": image_path})
    return resolved


def compose_scene_prompt(scene: dict, characters: list[dict], field: str) -> str:
    text = (scene.get(field) or scene.get("prompt") or "").strip()
    if not text:
        raise SystemExit(
            f"Scene {scene.get('sceneNumber')} is missing {field} "
            "(or a prompt fallback). Add imagePrompt and videoPrompt to the episode JSON."
        )
    blocks = [character["promptBlock"].strip() for character in characters]
    return "\n".join([*blocks, text])


def start_still_path(out_dir: Path, scene_number: int) -> Path:
    return out_dir / f"scene_{int(scene_number):02d}_start.png"


def present(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 1024


def inject_seed(graph: dict, seed: int) -> None:
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type")
        inputs = node.setdefault("inputs", {})
        if class_type == "RandomNoise":
            inputs["noise_seed"] = seed
            return
        if class_type == "KSampler":
            inputs["seed"] = seed
            return


def stage_start_still(image_path: Path) -> str:
    COMFY_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = COMFY_INPUT_DIR / image_path.name
    if dest.resolve() != image_path.resolve():
        shutil.copy2(image_path, dest)
    return dest.name


def inject_start_frame(graph: dict, image_name: str) -> None:
    width, height, length = latent_size(graph)
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
            "strength": I2V_STRENGTH,
        },
        "class_type": "LTXVImgToVideo",
        "_meta": {"title": "Animate from start frame"},
    }
    for node_id, key in (("6", "latent_image"), ("15", "latent"), ("18", "latent")):
        node = graph.get(node_id)
        if isinstance(node, dict):
            node.setdefault("inputs", {})[key] = ["20", 2]


def compose_edit_instruction(scene: dict, characters: list[dict]) -> str:
    scene_text = compose_scene_prompt(scene, characters, "imagePrompt")
    if len(characters) == 1:
        lock = (
            "Using Picture 1 as the identity reference, keep this exact person: "
            "same face, identity, hair, skin tone, and body. Do not replace them with someone else. "
            "Generate a new vertical 9:16 cinematic still of them in the scene below."
        )
    else:
        pictures = ", ".join(f"Picture {index}" for index in range(1, len(characters) + 1))
        lock = (
            f"Using {pictures} as identity references, keep those exact people "
            "(same faces, identities, hair, skin, bodies). Do not replace them. "
            "Generate a new vertical 9:16 cinematic still of them in the scene below."
        )
    return f"{lock}\n\n{scene_text}"


def inject_qwen_prompt(graph: dict, prompt: str) -> None:
    for node in graph.values():
        if not isinstance(node, dict) or node.get("class_type") != "TextEncodeQwenImageEditPlus":
            continue
        title = str((node.get("_meta") or {}).get("title", ""))
        if title == "Negative instruction":
            continue
        node.setdefault("inputs", {})["prompt"] = prompt
        return
    raise RuntimeError("Could not find the Qwen positive-instruction node in qwen_image_edit.json")


def inject_character_images(graph: dict, characters: list[dict]) -> None:
    if not characters:
        raise RuntimeError("Qwen-Image-Edit needs at least one character reference image")
    names = [stage_start_still(character["image_path"]) for character in characters[:MAX_QWEN_REFS]]
    extra_ids = ("13", "14")
    image_keys = ("image2", "image3")
    encoder = None
    for node in graph.values():
        if (
            isinstance(node, dict)
            and node.get("class_type") == "TextEncodeQwenImageEditPlus"
            and str((node.get("_meta") or {}).get("title", "")) != "Negative instruction"
        ):
            encoder = node
            break
    if encoder is None:
        raise RuntimeError("Could not find TextEncodeQwenImageEditPlus in qwen_image_edit.json")
    load_node = graph.get("6")
    if not isinstance(load_node, dict):
        raise RuntimeError("Could not find character LoadImage node 6 in qwen_image_edit.json")
    load_node.setdefault("inputs", {})["image"] = names[0]
    encoder_inputs = encoder.setdefault("inputs", {})
    encoder_inputs["image1"] = ["6", 0]
    for extra_id, image_key in zip(extra_ids, image_keys):
        encoder_inputs.pop(image_key, None)
        graph.pop(extra_id, None)
    for index, (extra_id, image_key) in enumerate(zip(extra_ids, image_keys), start=1):
        if index >= len(names):
            break
        graph[extra_id] = {
            "inputs": {"image": names[index]},
            "class_type": "LoadImage",
            "_meta": {"title": f"Character reference {index + 1}"},
        }
        encoder_inputs[image_key] = [extra_id, 0]


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
    raise RuntimeError("Could not find EmptyLTXVLatentVideo or EmptySD3LatentImage in the ComfyUI workflow")


def workflow_frame_rate(graph: dict) -> float:
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") in {"LTXVConditioning", "VHS_VideoCombine"}:
            rate = (node.get("inputs") or {}).get("frame_rate")
            if rate:
                return float(rate)
    return 24.0


def ltx_length_for_duration(duration_seconds: float, frame_rate: float) -> int:
    """LTX video length must be 8n+1 (9, 17, 25, ...)."""
    raw = max(1.0, float(duration_seconds) * float(frame_rate))
    n = max(1, round((raw - 1.0) / 8.0))
    return 8 * n + 1


def inject_scene_length(graph: dict, duration_seconds: float | None = None, length: int | None = None) -> int:
    if length is None:
        length = ltx_length_for_duration(float(duration_seconds or 1), workflow_frame_rate(graph))
    for node in graph.values():
        if isinstance(node, dict) and node.get("class_type") == "EmptyLTXVLatentVideo":
            node.setdefault("inputs", {})["length"] = length
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
                node["inputs"]["ckpt_name"] = node["inputs"].get("ckpt_name") or "ltx-2.3-22b-distilled-api-id.safetensors"
            return graph
    raise RuntimeError("Could not find a text-conditioning node in the ComfyUI workflow")


def concat_with_ffmpeg(scene_files: list[Path], dest: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not scene_files:
        return False
    list_file = dest.with_suffix(".txt")
    list_file.write_text("".join(f"file '{path.as_posix()}'\n" for path in scene_files), encoding="utf-8")
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(dest)],
        check=False,
        capture_output=True,
        text=True,
    )
    list_file.unlink(missing_ok=True)
    return result.returncode == 0


def generate_episode(script_path: Path, workflow_template: dict, registry: dict[str, dict], stage: str) -> None:
    script = load_json(script_path)
    series = script["series"]
    episode_number = script["episodeNumber"]
    out_dir = OUTPUT_DIR / series / str(episode_number)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(script, indent=2), encoding="utf-8")

    scene_files: list[Path] = []
    episode_started = time.time()
    generated = 0
    skipped = 0
    for scene in script["scenes"]:
        scene_number = scene["sceneNumber"]
        still_path = start_still_path(out_dir, scene_number)
        video_path = out_dir / f"scene_{scene_number:02d}.mp4"
        dest = still_path if stage == "frames" else video_path
        print(f"Queued scene {scene_number} ({stage})...", flush=True)
        if present(dest):
            print(f"  Skipped (already present): {dest} ({format_bytes(dest.stat().st_size)})", flush=True)
            if stage == "video":
                scene_files.append(dest)
            skipped += 1
            continue
        if stage == "video" and not present(still_path):
            raise SystemExit(
                f"Missing start still {still_path}. Run `pnpm run content:frames`, review the PNGs, "
                "then rerun `pnpm run content:generate`. Delete a PNG and rerun content:frames to retry it."
            )
        characters = resolve_scene_characters(scene, registry)
        seed = random.randint(0, 2**32 - 1)
        if stage == "frames":
            if not characters:
                raise SystemExit(
                    f"Scene {scene_number} has no characterIds. content:frames needs a character "
                    "reference PNG so Qwen-Image-Edit can lock identity."
                )
            if len(characters) > MAX_QWEN_REFS:
                print(
                    f"  Using the first {MAX_QWEN_REFS} of {len(characters)} character refs "
                    "(Qwen-Image-Edit accepts up to 3).",
                    flush=True,
                )
            raw = workflow_template["prompt"] if "prompt" in workflow_template else workflow_template
            graph = json.loads(json.dumps(raw))
            inject_qwen_prompt(graph, compose_edit_instruction(scene, characters))
            inject_seed(graph, seed)
            inject_character_images(graph, characters)
            prefer = "image"
            mode = "Qwen-Image-Edit"
        else:
            prompt = compose_scene_prompt(scene, characters, "videoPrompt")
            graph = inject_prompt(workflow_template, prompt, os.environ.get("LTXV_API_KEY", ""))
            inject_seed(graph, seed)
            duration_seconds = float(scene.get("durationSeconds") or 1)
            inject_scene_length(graph, duration_seconds=duration_seconds)
            inject_start_frame(graph, stage_start_still(still_path))
            prefer = "video"
            mode = "I2V"
        meta = graph_meta(graph)
        character_ids = ", ".join(scene.get("characterIds") or []) or "none"
        clip_seconds = None
        if meta["frame_rate"]:
            clip_seconds = meta["length"] / float(meta["frame_rate"])
        size_bit = f"{meta['width']}x{meta['height']}"
        if stage == "video":
            size_bit += f", {meta['length']} frames (~{clip_seconds:.2f}s)"
        print(
            f"  Graph: {size_bit}, seed {seed}"
            f"{f', {meta['steps']} steps' if meta['steps'] else ''}"
            f"{f' @ {meta['frame_rate']} fps' if meta['frame_rate'] else ''}"
            f", {mode} ({character_ids})",
            flush=True,
        )
        monitor = GpuMonitor()
        monitor.sample()
        scene_started = time.time()
        prompt_id = queue_prompt(graph)
        outputs, history_item = wait_for_output(prompt_id, monitor=monitor, prefer=prefer)
        download_output(outputs[0], dest)
        monitor.sample()
        elapsed = time.time() - scene_started
        comfy_elapsed = execution_seconds_from_history(history_item)
        timing = f"in {format_duration(elapsed)}"
        if comfy_elapsed is not None:
            timing += f" (ComfyUI execution {format_duration(comfy_elapsed)})"
        print(f"  Scene {scene_number} {stage} finished {timing}", flush=True)
        print(f"  Wrote {dest} ({format_bytes(dest.stat().st_size)})", flush=True)
        log_gpu_summary(monitor.summarize(), meta["width"], meta["height"], meta["length"])
        if stage == "video":
            scene_files.append(dest)
        generated += 1

    if stage == "frames":
        print(
            f"Start stills for {series}/{episode_number} are in {out_dir}. "
            "Review the scene_*_start.png files. Replace one by hand, or delete it and rerun "
            "`pnpm run content:frames`. When they look right, run `pnpm run content:generate`.",
            flush=True,
        )
    else:
        episode_mp4 = out_dir / "episode.mp4"
        if concat_with_ffmpeg(scene_files, episode_mp4):
            print(f"Concatenated {episode_mp4}")
        elif len(scene_files) == 1:
            shutil.copyfile(scene_files[0], episode_mp4)
            print(f"Copied single scene to {episode_mp4}")
        else:
            print("ffmpeg not found; left individual scene files. Install ffmpeg to concat.")
    print(
        f"Episode {series}/{episode_number} {stage} finished in {format_duration(time.time() - episode_started)}"
        f" ({generated} generated, {skipped} skipped)",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render episode start stills (Qwen-Image-Edit) or videos (LTX) via ComfyUI."
    )
    parser.add_argument("--stage", choices=("frames", "video"), default="video")
    args = parser.parse_args()
    load_dotenv()
    if args.stage == "video" and not os.environ.get("LTXV_API_KEY"):
        raise SystemExit("Missing LTXV_API_KEY in content-pipeline/.env")
    workflow_path = QWEN_WORKFLOW_PATH if args.stage == "frames" else LTX_WORKFLOW_PATH
    if not workflow_path.is_file():
        raise SystemExit(f"Missing ComfyUI workflow: {workflow_path}")
    workflow_template = load_json(workflow_path)
    registry = load_character_registry()
    scripts = sorted(SCRIPTS_DIR.glob("*/*.json"))
    if not scripts:
        raise SystemExit(f"No JSON scripts found in {SCRIPTS_DIR}")
    try:
        urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=3)
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"ComfyUI is not reachable at {COMFYUI_URL}. Start it with `pnpm run content:comfy`."
        ) from exc
    free_comfy_models()
    idle = GpuMonitor()
    idle.sample()
    print(
        f"ComfyUI ready at {COMFYUI_URL} | stage={args.stage} | idle {snapshot_line(idle.samples[0])}",
        flush=True,
    )
    for script_path in scripts:
        print(f"{'Start stills' if args.stage == 'frames' else 'Videos'} from {script_path}", flush=True)
        generate_episode(script_path, workflow_template, registry, args.stage)


if __name__ == "__main__":
    main()
