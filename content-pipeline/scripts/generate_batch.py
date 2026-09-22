#!/usr/bin/env python3
"""Render show JSON via ComfyUI: Qwen-Image-Edit stills, then LTX video."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from pathlib import Path

import cv2
from ffmpeg_tools import concat_videos, render_camera_move
from PIL import Image, ImageFilter
from spatial_previs import (
    character_facing_direction,
    character_screen_position,
    compile_spatial_image_summary,
    compile_spatial_video_prompt,
    generate_episode_previs,
    proxy_frame_path,
    scene_has_spatial_change,
    spatial_target_screen_position,
    validate_spatial_episode,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts_input"
OUTPUT_DIR = ROOT / "output"
PROMPT_KEYS = ("characterImage", "sceneStill", "sceneVideo")
OPTIONAL_PROMPT_KEYS = (
    "characterProfile",
    "setPlate",
    "environmentStill",
    "coverageStill",
    "spatialStill",
    "spatialCoverageStill",
    "spatialEnvironmentStill",
    "spatialGroupEndStill",
    "spatialInsertStill",
    "spatialEndStill",
)
PLACEHOLDER = re.compile(r"\{([a-zA-Z][a-zA-Z0-9]*)\}")
MAX_QWEN_REFS = 2
LTX_WORKFLOW_PATH = ROOT / "workflows" / "ltx_gemma_api.json"
QWEN_WORKFLOW_PATH = ROOT / "workflows" / "qwen_image_edit.json"
COMFY_INPUT_DIR = ROOT / ".comfyui" / "input"
FACE_DETECTOR_PATH = (
    ROOT / ".comfyui" / "models" / "quality" / "face_detection_yunet_2023mar.onnx"
)
COMFYUI_URL = "http://127.0.0.1:8188"
I2V_STRENGTH = 0.7  # official LTX image-to-video default
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
                if prefer == "video":
                    with_audio = [
                        info
                        for info in files
                        if "-audio." in str(info.get("filename") or "")
                    ]
                    files = with_audio or files
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


def require_text(obj: dict, key: str, where: str) -> str:
    value = obj.get(key)
    text = value.strip() if isinstance(value, str) else ""
    if not text:
        raise SystemExit(f"{where} is missing {key}")
    return text


def stable_seed(*parts: object) -> int:
    """Return a reproducible unsigned 32-bit seed for one render identity."""
    key = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2s(key, digest_size=4).digest(), "big")


def render_prompt(template: str, values: dict[str, str]) -> str:
    unused = set(values)

    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            known = ", ".join(sorted(values)) or "(none)"
            raise SystemExit(f"Unknown prompt placeholder {{{key}}}. Known: {known}")
        unused.discard(key)
        return values[key]

    rendered = PLACEHOLDER.sub(repl, template).strip()
    if unused:
        missing = ", ".join(f"{{{key}}}" for key in sorted(unused))
        raise SystemExit(
            f"prompts template never uses {missing}. Those fields are dropped, so Qwen/LTX "
            "never see the actual scene. Add the placeholders to the template."
        )
    if not rendered:
        raise SystemExit("Rendered prompt is empty")
    return rendered


def show_prompt(show: dict, key: str, values: dict[str, str]) -> str:
    return render_prompt(show["prompts"][key], values)


def write_black_png(path: Path, width: int = 768, height: int = 1360) -> None:
    raw = b"".join(b"\x00" + (b"\x00\x00\x00" * width) for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    )


def ensure_blank_png() -> str:
    COMFY_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = COMFY_INPUT_DIR / "blank-768x1360.png"
    if not dest.is_file():
        write_black_png(dest)
    return dest.name


def load_show(path: Path) -> dict:
    show = load_json(path)
    show_id = require_text(show, "id", path.name)
    if path.stem != show_id:
        raise SystemExit(f"{path.name}: id {show_id!r} must match the filename stem")
    require_text(show, "title", path.name)
    characters = show.get("characters")
    if not isinstance(characters, dict) or not characters:
        raise SystemExit(f"{path.name} is missing characters")
    cleaned_chars: dict[str, dict] = {}
    for cid, character in characters.items():
        if not isinstance(character, dict):
            raise SystemExit(f"characters[{cid!r}] must be an object")
        cleaned_chars[str(cid)] = {
            "id": str(cid),
            "promptBlock": require_text(character, "promptBlock", f"characters[{cid!r}]"),
        }
    show["characters"] = cleaned_chars
    if "locationCharacters" in show:
        raise SystemExit(
            f"{path.name} still has locationCharacters. Replace that object with locations "
            "(set text only) and put characterIds on each scene."
        )
    locations = show.get("locations")
    if not isinstance(locations, dict) or not locations:
        raise SystemExit(f"{path.name} is missing locations")
    cleaned_locs: dict[str, dict] = {}
    for loc_id, loc in locations.items():
        if not isinstance(loc, dict):
            raise SystemExit(f"locations[{loc_id!r}] must be an object")
        cleaned_locs[str(loc_id)] = {
            "id": str(loc_id),
            "promptBlock": require_text(loc, "promptBlock", f"locations[{loc_id!r}]"),
            "spatial": loc.get("spatial"),
        }
    show["locations"] = cleaned_locs
    prompts = show.get("prompts")
    if not isinstance(prompts, dict):
        raise SystemExit(f"{path.name} is missing prompts")
    show["prompts"] = {key: require_text(prompts, key, "prompts") for key in PROMPT_KEYS}
    for key in OPTIONAL_PROMPT_KEYS:
        if key in prompts:
            show["prompts"][key] = require_text(prompts, key, "prompts")
    episodes = show.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise SystemExit(f"{path.name} is missing episodes")
    cleaned_eps: list[dict] = []
    for index, episode in enumerate(episodes):
        if not isinstance(episode, dict):
            raise SystemExit(f"episodes[{index}] must be an object")
        if "episodeNumber" not in episode:
            raise SystemExit(f"episodes[{index}] is missing episodeNumber")
        ep_num = int(episode["episodeNumber"])
        scenes = episode.get("scenes")
        if not isinstance(scenes, list) or not scenes:
            raise SystemExit(f"episode {ep_num} is missing scenes")
        cleaned_scenes: list[dict] = []
        for scene_index, scene in enumerate(scenes, start=1):
            if not isinstance(scene, dict):
                raise SystemExit(f"episode {ep_num} has a non-object scene")
            scene_label = f"episode {ep_num} scene {scene.get('sceneNumber')}"
            scene_number = int(scene["sceneNumber"])
            if scene_number != scene_index:
                raise SystemExit(
                    f"{scene_label} is out of sequence; expected sceneNumber {scene_index}"
                )
            if scene.get("locationCharacterId"):
                raise SystemExit(f"{scene_label} uses locationCharacterId; rename it to locationId")
            loc_id = require_text(scene, "locationId", scene_label)
            if loc_id not in cleaned_locs:
                known = ", ".join(sorted(cleaned_locs))
                raise SystemExit(f"Unknown locationId {loc_id!r}. Known: {known}")
            character_ids = [str(cid) for cid in (scene.get("characterIds") or []) if cid]
            if len(character_ids) > 2:
                raise SystemExit(
                    f"{scene_label} has {len(character_ids)} characterIds; "
                    "keep at most two people on camera."
                )
            for cid in character_ids:
                if cid not in cleaned_chars:
                    raise SystemExit(f"{scene_label} names unknown character {cid!r}")
            shot_type = require_text(scene, "shotType", scene_label)
            speaker_id = scene.get("speakerId")
            if speaker_id is not None:
                speaker_id = str(speaker_id)
            addressee_id = scene.get("addresseeId")
            if addressee_id is not None:
                addressee_id = str(addressee_id)
            video_prompt = require_text(scene, "videoPrompt", scene_label)
            image_prompt = require_text(scene, "imagePrompt", scene_label)
            coverage_reference_scene_number = scene.get("coverageReferenceSceneNumber")
            if coverage_reference_scene_number is not None:
                coverage_reference_scene_number = int(coverage_reference_scene_number)
                if coverage_reference_scene_number >= scene_number:
                    raise SystemExit(
                        f"{scene_label} coverageReferenceSceneNumber must name an earlier scene"
                    )
            duration_seconds = float(scene.get("durationSeconds") or 0)
            if duration_seconds <= 0:
                raise SystemExit(f"{scene_label} durationSeconds must be positive")
            cleaned_scenes.append(
                {
                    "sceneNumber": scene_number,
                    "beatType": str(scene.get("beatType") or ""),
                    "coverageRole": str(scene.get("coverageRole") or ""),
                    "locationId": loc_id,
                    "storyBeat": require_text(scene, "storyBeat", scene_label),
                    "continuityIn": require_text(scene, "continuityIn", scene_label),
                    "continuityOut": require_text(scene, "continuityOut", scene_label),
                    "shotType": shot_type,
                    "characterIds": character_ids,
                    "speakerId": speaker_id,
                    "addresseeId": addressee_id,
                    "motionMode": str(scene.get("motionMode") or ""),
                    "coverageReferenceSceneNumber": coverage_reference_scene_number,
                    "focusTargetId": str(scene.get("focusTargetId") or ""),
                    "timeRangeSeconds": scene.get("timeRangeSeconds"),
                    "camera": scene.get("camera"),
                    "guideKeyframesSeconds": scene.get("guideKeyframesSeconds") or [],
                    "endGuideFrame": bool(scene.get("endGuideFrame")),
                    "imagePrompt": image_prompt,
                    "videoPrompt": video_prompt,
                    "durationSeconds": duration_seconds,
                }
            )
        cleaned_eps.append(
            {
                "episodeNumber": ep_num,
                "title": require_text(episode, "title", f"episode {ep_num}"),
                "isFree": bool(episode.get("isFree")),
                "coinCost": int(episode.get("coinCost") or 0),
                "logline": str(episode.get("logline") or ""),
                "dramaticQuestion": str(episode.get("dramaticQuestion") or ""),
                "hook": str(episode.get("hook") or ""),
                "reversal": str(episode.get("reversal") or ""),
                "cliffhanger": str(episode.get("cliffhanger") or ""),
                "nextEpisodeOpening": str(episode.get("nextEpisodeOpening") or ""),
                "screenDirection": str(episode.get("screenDirection") or ""),
                "spatialTimeline": episode.get("spatialTimeline"),
                "scenes": cleaned_scenes,
                "_allScenes": cleaned_scenes,
            }
        )
    show["episodes"] = cleaned_eps
    return show


def character_image_path(show_id: str, character_id: str) -> Path:
    return OUTPUT_DIR / show_id / "characters" / f"{character_id}.png"


def character_profile_path(show_id: str, character_id: str, direction: str) -> Path:
    return OUTPUT_DIR / show_id / "characters" / f"{character_id}_look_{direction}.png"


def episode_dir(show_id: str, episode_number: int) -> Path:
    return OUTPUT_DIR / show_id / str(episode_number)


def start_still_path(out_dir: Path, scene_number: int) -> Path:
    return out_dir / f"scene_{int(scene_number):02d}_start.png"


def end_still_path(out_dir: Path, scene_number: int) -> Path:
    return out_dir / f"scene_{int(scene_number):02d}_end.png"


def set_plate_path(out_dir: Path, scene_number: int) -> Path:
    return out_dir / f"scene_{int(scene_number):02d}_set_plate.png"


def resolve_location(show: dict, scene: dict) -> dict:
    return show["locations"][scene["locationId"]]


def resolve_scene_characters(
    show: dict,
    scene: dict,
    episode: dict | None = None,
) -> list[dict]:
    resolved: list[dict] = []
    for character_id in scene["characterIds"]:
        image_path = character_image_path(show["id"], character_id)
        if not present(image_path):
            raise SystemExit(
                f"Missing character still {image_path}. Run `pnpm run content:frames` "
                "so identity portraits exist before scene stills."
            )
        if episode is not None:
            direction = character_facing_direction(episode, scene, character_id)
            if direction:
                profile_path = character_profile_path(
                    show["id"], character_id, direction
                )
                if present(profile_path):
                    image_path = profile_path
        resolved.append({"id": character_id, "image_path": image_path})
    return resolved


def present(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 1024


def scene_motion_mode(scene: dict) -> str:
    explicit = scene.get("motionMode")
    if explicit in {"generative", "cameraOnly"}:
        return explicit
    if scene.get("speakerId"):
        return "generative"
    if scene.get("shotType") in {"establishing", "insert", "twoShot", "reaction"}:
        return "cameraOnly"
    return "generative"


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


def stage_named_image(image_path: Path, prefix: str) -> str:
    COMFY_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = COMFY_INPUT_DIR / f"{prefix}_{image_path.name}"
    shutil.copy2(image_path, dest)
    return dest.name


def stage_coverage_crop(
    image_path: Path,
    prefix: str,
    screen_position: tuple[float, float],
    canvas_size: tuple[float, float],
) -> str:
    COMFY_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = COMFY_INPUT_DIR / f"{prefix}_{image_path.name}"
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        # A recognizable character crop plus a separate identity reference causes
        # Qwen to render the same person twice. Keep only coarse set/light/color.
        tiny_width = 48
        tiny_height = max(1, round(tiny_width * rgb.height / rgb.width))
        continuity = rgb.resize(
            (tiny_width, tiny_height), Image.Resampling.BILINEAR
        ).resize(rgb.size, Image.Resampling.BILINEAR)
        continuity.filter(ImageFilter.GaussianBlur(radius=14.0)).save(dest)
    return dest.name


def stage_insert_crop(
    image_path: Path,
    prefix: str,
    screen_position: tuple[float, float],
    canvas_size: tuple[float, float],
) -> str:
    COMFY_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = COMFY_INPUT_DIR / f"{prefix}_{image_path.name}"
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        center_x = screen_position[0] * rgb.width / canvas_size[0]
        center_y = screen_position[1] * rgb.height / canvas_size[1]
        crop_width = rgb.width * 0.32
        crop_height = rgb.height * 0.36
        left = max(0.0, min(rgb.width - crop_width, center_x - crop_width / 2.0))
        top = max(0.0, min(rgb.height - crop_height, center_y - crop_height / 2.0))
        crop = (
            round(left),
            round(top),
            round(left + crop_width),
            round(top + crop_height),
        )
        rgb.crop(crop).resize(rgb.size, Image.Resampling.LANCZOS).save(dest)
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


def inject_end_frame(graph: dict, image_name: str) -> None:
    _, _, length = latent_size(graph)
    graph["21"] = {
        "inputs": {"image": image_name},
        "class_type": "LoadImage",
        "_meta": {"title": "Scene end frame"},
    }
    graph["27"] = {
        "inputs": {
            "positive": ["20", 0],
            "negative": ["20", 1],
            "vae": ["9", 0],
            "latent": ["20", 2],
            "image": ["21", 0],
            "frame_idx": -1,
            "strength": 0.85,
        },
        "class_type": "LTXVAddGuide",
        "_meta": {"title": "Pin spatial end frame"},
    }
    graph["24"]["inputs"]["video_latent"] = ["27", 2]
    graph["23"]["inputs"]["frames_number"] = length + 8
    for node_id, key in (("15", "latent"), ("18", "latent")):
        graph[node_id]["inputs"][key] = ["27", 2]
    graph["17"]["inputs"]["conditioning"] = ["27", 0]
    graph["28"] = {
        "inputs": {
            "positive": ["27", 0],
            "negative": ["27", 1],
            "latent": ["25", 0],
        },
        "class_type": "LTXVCropGuides",
        "_meta": {"title": "Remove guide tokens"},
    }
    graph["25"]["inputs"]["av_latent"] = ["6", 0]
    graph["8"]["inputs"]["samples"] = ["28", 2]


def inject_qwen_character_canvas(graph: dict) -> None:
    inject_qwen_image_slots(graph, [(ensure_blank_png(), "Blank canvas")])


def inject_qwen_character_refs(graph: dict, characters: list[dict]) -> None:
    if not characters:
        raise RuntimeError("Scene stills need at least one character reference PNG")
    refs: list[tuple[str, str]] = []
    for index, character in enumerate(characters[:MAX_QWEN_REFS], start=1):
        refs.append(
            (
                stage_start_still(character["image_path"]),
                f"Character reference {index} ({character['id']})",
            )
        )
    inject_qwen_image_slots(graph, refs)


def inject_qwen_coverage_refs(
    graph: dict, characters: list[dict], coverage_reference_path: Path
) -> None:
    if not characters:
        raise RuntimeError("Coverage stills need at least one character reference PNG")
    refs = [
        (
            stage_start_still(character["image_path"]),
            f"Character reference {index} ({character['id']})",
        )
        for index, character in enumerate(characters[:MAX_QWEN_REFS], start=1)
    ]
    refs.append(
        (
            stage_start_still(coverage_reference_path),
            f"Coverage master ({coverage_reference_path.stem})",
        )
    )
    inject_qwen_image_slots(graph, refs)


def inject_qwen_spatial_refs(
    graph: dict,
    characters: list[dict],
    proxy_path: Path,
    coverage_reference_path: Path | None = None,
    coverage_screen_position: tuple[float, float] | None = None,
) -> None:
    refs = [
        (
            stage_start_still(character["image_path"]),
            f"Character reference {index} ({character['id']})",
        )
        for index, character in enumerate(characters[:MAX_QWEN_REFS], start=1)
    ]
    if coverage_reference_path is not None:
        refs.append(
            (
                stage_named_image(coverage_reference_path, "set_plate"),
                f"Person-free set continuity plate ({coverage_reference_path.stem})",
            )
        )
    refs.append((stage_named_image(proxy_path, "proxy"), "3D spatial projection"))
    if len(refs) > 3:
        raise RuntimeError("Qwen supports at most three spatial references")
    inject_qwen_image_slots(graph, refs)


def inject_qwen_environment_proxy(graph: dict, proxy_path: Path) -> None:
    inject_qwen_image_slots(
        graph,
        [(stage_named_image(proxy_path, "proxy"), "3D spatial projection")],
    )


def inject_qwen_set_plate_ref(graph: dict, master_path: Path) -> None:
    inject_qwen_image_slots(
        graph,
        [(stage_named_image(master_path, "set_master"), "Photorealistic coverage master")],
    )


def inject_qwen_spatial_insert_refs(
    graph: dict,
    coverage_path: Path,
    proxy_path: Path,
    target_screen_position: tuple[float, float],
) -> None:
    inject_qwen_image_slots(
        graph,
        [
            (
                stage_insert_crop(
                    coverage_path,
                    "insert_crop",
                    target_screen_position,
                    (768.0, 1360.0),
                ),
                "Photorealistic prop crop from coverage master",
            ),
            (stage_named_image(proxy_path, "proxy"), "3D spatial projection"),
        ],
    )


def inject_qwen_end_refs(
    graph: dict,
    character: dict,
    start_frame_path: Path,
    proxy_path: Path,
) -> None:
    inject_qwen_image_slots(
        graph,
        [
            (stage_named_image(start_frame_path, "shot_start"), "Photorealistic shot start"),
            (
                stage_start_still(character["image_path"]),
                f"Character reference ({character['id']})",
            ),
            (stage_named_image(proxy_path, "proxy_end"), "3D spatial end projection"),
        ],
    )


def inject_qwen_group_end_refs(
    graph: dict,
    start_frame_path: Path,
    proxy_path: Path,
) -> None:
    inject_qwen_image_slots(
        graph,
        [
            (stage_named_image(start_frame_path, "shot_start"), "Photorealistic shot start"),
            (stage_named_image(proxy_path, "proxy_end"), "3D spatial end projection"),
        ],
    )


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


def inject_qwen_image_slots(graph: dict, refs: list[tuple[str, str]]) -> None:
    if not refs:
        raise RuntimeError("Qwen-Image-Edit needs at least one reference image")
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
    load_node.setdefault("inputs", {})["image"] = refs[0][0]
    load_node["_meta"] = {"title": refs[0][1]}
    encoder_inputs = encoder.setdefault("inputs", {})
    encoder_inputs["image1"] = ["6", 0]
    for extra_id, image_key in zip(extra_ids, image_keys):
        encoder_inputs.pop(image_key, None)
        graph.pop(extra_id, None)
    for extra_id, image_key, (image_name, title) in zip(extra_ids, image_keys, refs[1:]):
        graph[extra_id] = {
            "inputs": {"image": image_name},
            "class_type": "LoadImage",
            "_meta": {"title": title},
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
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type")
        inputs = node.setdefault("inputs", {})
        if class_type == "EmptyLTXVLatentVideo":
            inputs["length"] = length
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


def execute_queued_graph(graph: dict, dest: Path, prefer: str, mode: str) -> None:
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
    outputs, history_item = wait_for_output(prompt_id, monitor=monitor, prefer=prefer)
    download_output(outputs[0], dest)
    monitor.sample()
    elapsed = time.time() - started
    timing = f"in {format_duration(elapsed)}"
    comfy_elapsed = execution_seconds_from_history(history_item)
    if comfy_elapsed is not None:
        timing += f" (ComfyUI execution {format_duration(comfy_elapsed)})"
    print(f"  Finished {timing}", flush=True)
    print(f"  Wrote {dest} ({format_bytes(dest.stat().st_size)})", flush=True)
    log_gpu_summary(monitor.summarize(), meta["width"], meta["height"], meta["length"])


def run_qwen_image(
    workflow_template: dict,
    dest: Path,
    prompt: str,
    mode: str,
    inject_images,
    seed: int,
    expected_max_faces: int | None = None,
) -> None:
    attempts = 3 if expected_max_faces is not None else 1
    for attempt in range(attempts):
        attempt_seed = (seed + attempt * 104729) & 0xFFFFFFFF
        graph = clone_workflow(workflow_template)
        inject_qwen_prompt(graph, prompt)
        inject_seed(graph, attempt_seed)
        inject_images(graph)
        print(f"  Graph seed {attempt_seed}", flush=True)
        execute_queued_graph(graph, dest, prefer="image", mode=mode)
        if expected_max_faces is None:
            return
        observed_faces = detect_face_count(dest)
        if observed_faces <= expected_max_faces:
            return
        print(
            f"  Rejected: detected {observed_faces} faces, expected at most "
            f"{expected_max_faces}. Rerolling...",
            flush=True,
        )
    raise RuntimeError(
        f"{dest} repeatedly contained more than {expected_max_faces} visible faces"
    )


def detect_face_count(image_path: Path) -> int:
    if not present(FACE_DETECTOR_PATH):
        raise RuntimeError(
            f"Missing duplicate-face detector {FACE_DETECTOR_PATH}. "
            "Run `pnpm run content:models`."
        )
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read generated image {image_path}")
    height, width = image.shape[:2]
    detector = cv2.FaceDetectorYN_create(
        str(FACE_DETECTOR_PATH),
        "",
        (width, height),
        0.55,
        0.3,
        5000,
    )
    _, faces = detector.detect(image)
    return 0 if faces is None else len(faces)


def generate_show(
    show: dict,
    workflow_template: dict,
    stage: str,
    partial: bool = False,
    force: bool = False,
    seed_override: int | None = None,
) -> None:
    show_id = show["id"]
    started = time.time()
    generated = 0
    skipped = 0

    if stage == "frames":
        for character_id, character in show["characters"].items():
            dest = character_image_path(show_id, character_id)
            print(f"Queued character {character_id}...", flush=True)
            if present(dest):
                print(f"  Skipped (already present): {dest} ({format_bytes(dest.stat().st_size)})", flush=True)
                skipped += 1
                continue
            prompt = show_prompt(
                show,
                "characterImage",
                {
                    "characterPromptBlock": character["promptBlock"],
                },
            )
            run_qwen_image(
                workflow_template,
                dest,
                prompt,
                f"Qwen character ({character_id})",
                inject_qwen_character_canvas,
                stable_seed(show_id, "character", character_id),
                expected_max_faces=1,
            )
            generated += 1
        if "characterProfile" in show["prompts"]:
            profile_needs: set[tuple[str, str]] = set()
            for episode in show["episodes"]:
                for scene in episode["scenes"]:
                    for character_id in scene["characterIds"]:
                        direction = character_facing_direction(
                            episode, scene, character_id
                        )
                        if direction:
                            profile_needs.add((character_id, direction))
            for character_id, direction in sorted(profile_needs):
                source = character_image_path(show_id, character_id)
                dest = character_profile_path(show_id, character_id, direction)
                print(
                    f"Queued character profile {character_id} looking {direction}...",
                    flush=True,
                )
                if present(dest):
                    print(
                        f"  Skipped (already present): {dest} "
                        f"({format_bytes(dest.stat().st_size)})",
                        flush=True,
                    )
                    skipped += 1
                    continue
                prompt = show_prompt(
                    show,
                    "characterProfile",
                    {
                        "characterId": character_id,
                        "direction": direction,
                    },
                )
                run_qwen_image(
                    workflow_template,
                    dest,
                    prompt,
                    f"Qwen profile reference ({character_id} looking {direction})",
                    lambda graph,
                    source=source,
                    character_id=character_id: inject_qwen_image_slots(
                        graph,
                        [
                            (
                                stage_start_still(source),
                                f"Frontal identity of {character_id}",
                            )
                        ],
                    ),
                    stable_seed(show_id, "profile", character_id, direction),
                    expected_max_faces=1,
                )
                generated += 1

    for episode in show["episodes"]:
        episode_number = episode["episodeNumber"]
        spatial_errors = validate_spatial_episode(show, episode)
        if spatial_errors:
            raise SystemExit("Spatial validation failed:\n- " + "\n- ".join(spatial_errors))
        out_dir = episode_dir(show_id, episode_number)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "series": show_id,
                    "episodeNumber": episode_number,
                    "title": episode["title"],
                    "isFree": episode["isFree"],
                    "coinCost": episode["coinCost"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        episode_started = time.time()
        scene_files: list[Path] = []
        episode_generated = 0
        episode_skipped = 0
        for scene in episode["scenes"]:
            scene_number = scene["sceneNumber"]
            scene_label = f"episode {episode_number} scene {scene_number}"
            still_path = start_still_path(out_dir, scene_number)
            end_path = end_still_path(out_dir, scene_number)
            video_path = out_dir / f"scene_{scene_number:02d}.mp4"
            dest = still_path if stage == "frames" else video_path
            location = resolve_location(show, scene)
            names = ", ".join(scene["characterIds"])
            print(f"Queued {show_id}/{episode_number} scene {scene_number} ({stage})...", flush=True)
            needs_end_guide = bool(scene["endGuideFrame"])
            frame_outputs_present = present(still_path) and (
                not needs_end_guide or present(end_path)
            )
            output_present = frame_outputs_present if stage == "frames" else present(dest)
            if output_present and not force:
                print(f"  Skipped (already present): {dest} ({format_bytes(dest.stat().st_size)})", flush=True)
                if stage == "video":
                    scene_files.append(dest)
                episode_skipped += 1
                skipped += 1
                continue
            if stage == "video" and not present(still_path):
                raise SystemExit(
                    f"Missing start still {still_path}. Run `pnpm run content:frames`, review the PNGs, "
                    "then rerun `pnpm run content:generate`. Delete a PNG and rerun content:frames to retry it."
                )
            if stage == "video" and needs_end_guide and not present(end_path):
                raise SystemExit(
                    f"Missing spatial end guide {end_path}. Regenerate this scene with "
                    "`pnpm run content:frame -- --show <id> --episode <n> --scene <n> --force`."
                )
            if stage == "frames":
                if scene.get("camera") and scene.get("timeRangeSeconds"):
                    generate_episode_previs(show, episode, scene_number)
                seed = (
                    seed_override
                    if seed_override is not None
                    else stable_seed(show_id, episode_number, scene_number, "frame")
                )
                if not present(still_path) or force:
                    if scene["characterIds"]:
                        characters = resolve_scene_characters(show, scene, episode)
                        coverage_reference_scene_number = scene["coverageReferenceSceneNumber"]
                        spatial_proxy_path = proxy_frame_path(
                            show_id, episode_number, scene_number, "start_condition"
                        )
                        if scene.get("camera") and scene.get("timeRangeSeconds"):
                            if not present(spatial_proxy_path):
                                raise SystemExit(f"Missing spatial proxy {spatial_proxy_path}")
                            coverage_screen_position = None
                            if coverage_reference_scene_number is not None:
                                template_key = "spatialCoverageStill"
                                coverage_master_path = start_still_path(
                                    out_dir, coverage_reference_scene_number
                                )
                                if not present(coverage_master_path):
                                    raise SystemExit(
                                        f"{scene_label} needs coverage master "
                                        f"{coverage_master_path}"
                                    )
                                coverage_reference_path = set_plate_path(
                                    out_dir, coverage_reference_scene_number
                                )
                                if not present(coverage_reference_path):
                                    if "setPlate" not in show["prompts"]:
                                        raise SystemExit(
                                            f"{scene_label} requires prompts.setPlate"
                                        )
                                    plate_prompt = show_prompt(
                                        show,
                                        "setPlate",
                                        {
                                            "locationPromptBlock": location[
                                                "promptBlock"
                                            ]
                                        },
                                    )
                                    run_qwen_image(
                                        workflow_template,
                                        coverage_reference_path,
                                        plate_prompt,
                                        f"Qwen person-free set plate "
                                        f"(scene {coverage_reference_scene_number})",
                                        lambda graph,
                                        master=coverage_master_path: inject_qwen_set_plate_ref(
                                            graph, master
                                        ),
                                        stable_seed(
                                            show_id,
                                            episode_number,
                                            coverage_reference_scene_number,
                                            "set-plate",
                                        ),
                                        expected_max_faces=0,
                                    )
                                spatial_proxy_path = proxy_frame_path(
                                    show_id,
                                    episode_number,
                                    scene_number,
                                    "start_pose_condition",
                                )
                                if len(scene["characterIds"]) != 1:
                                    raise SystemExit(
                                        f"{scene_label} spatial coverage requires one character"
                                    )
                                coverage_scene = next(
                                    candidate
                                    for candidate in episode["_allScenes"]
                                    if candidate["sceneNumber"]
                                    == coverage_reference_scene_number
                                )
                                coverage_screen_position = character_screen_position(
                                    episode, coverage_scene, scene["characterIds"][0]
                                )
                                prompt_values = {
                                    "referenceMap": "; ".join(
                                        f"Picture {index} = identity of {character_id}"
                                        for index, character_id in enumerate(
                                            scene["characterIds"], start=1
                                        )
                                    ),
                                    "coverageReferencePictureNumber": str(
                                        len(scene["characterIds"]) + 1
                                    ),
                                    "proxyPictureNumber": str(
                                        len(scene["characterIds"]) + 2
                                    ),
                                    "characterCount": str(len(scene["characterIds"])),
                                    "characterIds": ", ".join(scene["characterIds"]),
                                    "locationPromptBlock": location["promptBlock"],
                                    "blockingSummary": compile_spatial_image_summary(
                                        episode, scene
                                    ),
                                    "imagePrompt": scene["imagePrompt"],
                                }
                            else:
                                template_key = "spatialStill"
                                coverage_reference_path = None
                                prompt_values = {
                                    "referenceMap": "; ".join(
                                        f"Picture {index} = identity of {character_id}"
                                        for index, character_id in enumerate(
                                            scene["characterIds"], start=1
                                        )
                                    ),
                                    "proxyPictureNumber": str(
                                        len(scene["characterIds"]) + 1
                                    ),
                                    "characterCount": str(len(scene["characterIds"])),
                                    "characterIds": ", ".join(scene["characterIds"]),
                                    "locationPromptBlock": location["promptBlock"],
                                    "blockingSummary": compile_spatial_image_summary(
                                        episode, scene
                                    ),
                                    "imagePrompt": scene["imagePrompt"],
                                }
                            if template_key not in show["prompts"]:
                                raise SystemExit(
                                    f"{scene_label} requires prompts.{template_key}"
                                )
                            prompt = show_prompt(show, template_key, prompt_values)
                            inject_images = (
                                lambda graph,
                                chars=characters,
                                proxy=spatial_proxy_path,
                                ref=coverage_reference_path,
                                screen_position=coverage_screen_position: inject_qwen_spatial_refs(
                                    graph, chars, proxy, ref, screen_position
                                )
                            )
                        elif coverage_reference_scene_number is not None:
                            if "coverageStill" not in show["prompts"]:
                                raise SystemExit(
                                    f"{scene_label} uses coverageReferenceSceneNumber and "
                                    "requires prompts.coverageStill"
                                )
                            coverage_reference_path = start_still_path(
                                out_dir, coverage_reference_scene_number
                            )
                            if not present(coverage_reference_path):
                                raise SystemExit(
                                    f"{scene_label} needs coverage master "
                                    f"{coverage_reference_path}"
                                )
                            prompt = show_prompt(
                                show,
                                "coverageStill",
                                {
                                    "referenceMap": "; ".join(
                                        f"Picture {index} = identity of {character_id}"
                                        for index, character_id in enumerate(
                                            scene["characterIds"], start=1
                                        )
                                    ),
                                    "coverageReferencePictureNumber": str(
                                        len(scene["characterIds"]) + 1
                                    ),
                                    "characterCount": str(len(scene["characterIds"])),
                                    "characterIds": ", ".join(scene["characterIds"]),
                                    "imagePrompt": scene["imagePrompt"],
                                },
                            )
                            inject_images = (
                                lambda graph,
                                chars=characters,
                                ref=coverage_reference_path: inject_qwen_coverage_refs(
                                    graph, chars, ref
                                )
                            )
                        else:
                            prompt = show_prompt(
                                show,
                                "sceneStill",
                                {
                                    "referenceMap": "; ".join(
                                        f"Picture {index} = {character_id}"
                                        for index, character_id in enumerate(
                                            scene["characterIds"], start=1
                                        )
                                    ),
                                    "characterCount": str(len(scene["characterIds"])),
                                    "characterIds": ", ".join(scene["characterIds"]),
                                    "locationPromptBlock": location["promptBlock"],
                                    "imagePrompt": scene["imagePrompt"],
                                },
                            )
                            inject_images = (
                                lambda graph,
                                chars=characters: inject_qwen_character_refs(graph, chars)
                            )
                    else:
                        if scene.get("camera") and scene.get("timeRangeSeconds"):
                            coverage_reference_scene_number = scene[
                                "coverageReferenceSceneNumber"
                            ]
                            focus_target_id = scene.get("focusTargetId")
                            spatial_proxy_path = proxy_frame_path(
                                show_id,
                                episode_number,
                                scene_number,
                                "start_condition",
                            )
                            if coverage_reference_scene_number and focus_target_id:
                                if "spatialInsertStill" not in show["prompts"]:
                                    raise SystemExit(
                                        f"{scene_label} requires prompts.spatialInsertStill"
                                    )
                                coverage_scene = next(
                                    candidate
                                    for candidate in episode.get(
                                        "_allScenes", episode["scenes"]
                                    )
                                    if candidate["sceneNumber"]
                                    == coverage_reference_scene_number
                                )
                                coverage_reference_path = start_still_path(
                                    out_dir, coverage_reference_scene_number
                                )
                                if not present(coverage_reference_path):
                                    raise SystemExit(
                                        f"{scene_label} needs coverage master "
                                        f"{coverage_reference_path}"
                                    )
                                target_position = spatial_target_screen_position(
                                    show,
                                    episode,
                                    coverage_scene,
                                    focus_target_id,
                                )
                                prompt = show_prompt(
                                    show,
                                    "spatialInsertStill",
                                    {
                                        "focusTargetId": focus_target_id,
                                        "locationPromptBlock": location["promptBlock"],
                                        "imagePrompt": scene["imagePrompt"],
                                    },
                                )
                                inject_images = (
                                    lambda graph,
                                    ref=coverage_reference_path,
                                    proxy=spatial_proxy_path,
                                    target=target_position: inject_qwen_spatial_insert_refs(
                                        graph, ref, proxy, target
                                    )
                                )
                            elif "spatialEnvironmentStill" not in show["prompts"]:
                                raise SystemExit(
                                    f"{scene_label} requires prompts.spatialEnvironmentStill"
                                )
                            else:
                                prompt = show_prompt(
                                    show,
                                    "spatialEnvironmentStill",
                                    {
                                        "proxyPictureNumber": "1",
                                        "locationPromptBlock": location["promptBlock"],
                                        "imagePrompt": scene["imagePrompt"],
                                    },
                                )
                                inject_images = (
                                    lambda graph,
                                    proxy=spatial_proxy_path: inject_qwen_environment_proxy(
                                        graph, proxy
                                    )
                                )
                        else:
                            if "environmentStill" not in show["prompts"]:
                                raise SystemExit(
                                    f"{scene_label} has no characterIds and requires "
                                    "prompts.environmentStill"
                                )
                            prompt = show_prompt(
                                show,
                                "environmentStill",
                                {
                                    "locationPromptBlock": location["promptBlock"],
                                    "imagePrompt": scene["imagePrompt"],
                                },
                            )
                            inject_images = inject_qwen_character_canvas
                    run_qwen_image(
                        workflow_template,
                        still_path,
                        prompt,
                        f"Qwen scene still ({names or 'environment'} @ {location['id']})",
                        inject_images,
                        seed,
                        expected_max_faces=len(scene["characterIds"]),
                    )
                if needs_end_guide and (not present(end_path) or force):
                    if not scene_has_spatial_change(episode, scene):
                        shutil.copy2(still_path, end_path)
                        print(
                            f"  Copied static spatial endpoint {end_path}",
                            flush=True,
                        )
                        episode_generated += 1
                        generated += 1
                        continue
                    characters = resolve_scene_characters(show, scene, episode)
                    proxy_end_path = proxy_frame_path(
                        show_id, episode_number, scene_number, "end_condition"
                    )
                    if len(characters) == 1:
                        if "spatialEndStill" not in show["prompts"]:
                            raise SystemExit(
                                f"{scene_label} requires prompts.spatialEndStill"
                            )
                        end_prompt = show_prompt(
                            show,
                            "spatialEndStill",
                            {
                                "characterId": scene["characterIds"][0],
                                "imagePrompt": scene["imagePrompt"],
                            },
                        )
                        inject_end_images = (
                            lambda graph,
                            character=characters[0],
                            start=still_path,
                            proxy=proxy_end_path: inject_qwen_end_refs(
                                graph, character, start, proxy
                            )
                        )
                    else:
                        if "spatialGroupEndStill" not in show["prompts"]:
                            raise SystemExit(
                                f"{scene_label} requires prompts.spatialGroupEndStill"
                            )
                        end_prompt = show_prompt(
                            show,
                            "spatialGroupEndStill",
                            {
                                "characterCount": str(len(characters)),
                                "characterIds": ", ".join(scene["characterIds"]),
                                "imagePrompt": scene["imagePrompt"],
                            },
                        )
                        inject_end_images = (
                            lambda graph,
                            start=still_path,
                            proxy=proxy_end_path: inject_qwen_group_end_refs(
                                graph, start, proxy
                            )
                        )
                    run_qwen_image(
                        workflow_template,
                        end_path,
                        end_prompt,
                        f"Qwen spatial end guide ({names} @ {location['id']})",
                        inject_end_images,
                        stable_seed(show_id, episode_number, scene_number, "end-frame"),
                        expected_max_faces=len(scene["characterIds"]),
                    )
            else:
                camera = scene.get("camera") or {}
                if scene_motion_mode(scene) == "cameraOnly":
                    start_position = camera.get("position") or [0.0, 0.0, 0.0]
                    end_position = camera.get("endPosition") or start_position
                    delta_x = float(end_position[0]) - float(start_position[0])
                    delta_z = float(end_position[2]) - float(start_position[2])
                    render_camera_move(
                        still_path,
                        dest,
                        float(scene["durationSeconds"]),
                        horizontal_direction=(
                            -1.0 if delta_x > 0 else 1.0 if delta_x < 0 else 0.0
                        ),
                        vertical_direction=(
                            -1.0 if delta_z > 0 else 1.0 if delta_z < 0 else 0.0
                        ),
                    )
                    print(
                        "  Rendered deterministic silent-anchor camera move",
                        flush=True,
                    )
                else:
                    prompt = show_prompt(
                        show,
                        "sceneVideo",
                        {
                            "videoPrompt": compile_spatial_video_prompt(episode, scene),
                        },
                    )
                    seed = (
                        seed_override
                        if seed_override is not None
                        else stable_seed(show_id, episode_number, scene_number, "video")
                    )
                    graph = inject_prompt(
                        workflow_template, prompt, os.environ.get("LTXV_API_KEY", "")
                    )
                    if scene.get("speakerId"):
                        inject_dialogue_multimodal_guider(graph)
                    inject_seed(graph, seed)
                    inject_scene_length(graph, duration_seconds=scene["durationSeconds"])
                    inject_start_frame(graph, stage_start_still(still_path))
                    if needs_end_guide:
                        inject_end_frame(graph, stage_named_image(end_path, "ltx_end"))
                    print(f"  Graph seed {seed}", flush=True)
                    execute_queued_graph(
                        graph,
                        dest,
                        prefer="video",
                        mode=f"I2V ({names} @ {location['id']})",
                    )
                scene_files.append(dest)
            episode_generated += 1
            generated += 1
        if stage == "video" and not partial:
            episode_mp4 = out_dir / "episode.mp4"
            concat_videos(scene_files, episode_mp4)
            print(f"Wrote {episode_mp4} ({format_bytes(episode_mp4.stat().st_size)})", flush=True)
        print(
            f"Episode {show_id}/{episode_number} {stage} finished in {format_duration(time.time() - episode_started)}"
            f" ({episode_generated} generated, {episode_skipped} skipped)",
            flush=True,
        )

    if stage == "frames":
        print(
            f"Stills for {show_id} are in {OUTPUT_DIR / show_id}. "
            "Review characters/, then each episode's scene_*_start.png. "
            "Replace a file by hand, or delete it and rerun `pnpm run content:frames`. "
            "When they look right, run `pnpm run content:generate`.",
            flush=True,
        )
    print(
        f"Show {show_id} {stage} finished in {format_duration(time.time() - started)}"
        f" ({generated} generated, {skipped} skipped)",
        flush=True,
    )


def stage_needs_comfy(shows: list[dict], stage: str) -> bool:
    for show in shows:
        show_id = show["id"]
        if stage == "frames":
            for character_id in show["characters"]:
                if not present(character_image_path(show_id, character_id)):
                    return True
            for episode in show["episodes"]:
                out_dir = episode_dir(show_id, episode["episodeNumber"])
                for scene in episode["scenes"]:
                    if not present(start_still_path(out_dir, scene["sceneNumber"])):
                        return True
                    if scene["endGuideFrame"] and not present(
                        end_still_path(out_dir, scene["sceneNumber"])
                    ):
                        return True
        else:
            for episode in show["episodes"]:
                out_dir = episode_dir(show_id, episode["episodeNumber"])
                for scene in episode["scenes"]:
                    dest = out_dir / f"scene_{int(scene['sceneNumber']):02d}.mp4"
                    if not present(dest):
                        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render show stills (Qwen-Image-Edit) or videos (LTX) via ComfyUI."
    )
    parser.add_argument("--stage", choices=("frames", "video"), default="video")
    parser.add_argument("--show", help="Render only this show id")
    parser.add_argument("--episode", type=int, help="Render only this episode number")
    parser.add_argument("--scene", type=int, help="Render only this scene number (requires --episode)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate an existing selected scene (requires --scene)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override the deterministic seed for one selected scene (requires --scene)",
    )
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
    workflow_path = QWEN_WORKFLOW_PATH if args.stage == "frames" else LTX_WORKFLOW_PATH
    if not workflow_path.is_file():
        raise SystemExit(f"Missing ComfyUI workflow: {workflow_path}")
    workflow_template = load_json(workflow_path)
    scripts = sorted(SCRIPTS_DIR.glob("*.json"))
    if args.show:
        scripts = [path for path in scripts if path.stem == args.show]
    if not scripts:
        target = f" for show {args.show!r}" if args.show else ""
        raise SystemExit(f"No show JSON files found{target} in {SCRIPTS_DIR}")
    shows = [load_show(path) for path in scripts]
    for show in shows:
        if args.episode is not None:
            show["episodes"] = [
                episode
                for episode in show["episodes"]
                if episode["episodeNumber"] == args.episode
            ]
            if not show["episodes"]:
                raise SystemExit(f"Show {show['id']!r} has no episode {args.episode}")
        if args.scene is not None:
            scenes = [
                scene
                for scene in show["episodes"][0]["scenes"]
                if scene["sceneNumber"] == args.scene
            ]
            if not scenes:
                raise SystemExit(
                    f"Show {show['id']!r} episode {args.episode} has no scene {args.scene}"
                )
            show["episodes"][0]["scenes"] = scenes
    needs_comfy = args.force or stage_needs_comfy(shows, args.stage)
    if needs_comfy:
        if args.stage == "video" and not os.environ.get("LTXV_API_KEY"):
            raise SystemExit("Missing LTXV_API_KEY in content-pipeline/.env")
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
    else:
        print(
            f"All scene files already present | stage={args.stage} | skipping ComfyUI",
            flush=True,
        )
    for show, script_path in zip(shows, scripts):
        print(f"{'Stills' if args.stage == 'frames' else 'Videos'} from {script_path}", flush=True)
        generate_show(
            show,
            workflow_template,
            args.stage,
            partial=args.scene is not None,
            force=args.force,
            seed_override=args.seed,
        )


if __name__ == "__main__":
    main()
