#!/usr/bin/env python3

"""Shared helpers for the LTX depth gate experiment (temporary)."""

from __future__ import annotations

import _scripts_path  # noqa: F401

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from clip_spec import CLIP_HEIGHT, CLIP_WIDTH, FPS, fit_to_clip
from ffmpeg_tools import encode_rgb_frames, find_ffmpeg
from pipeline_paths import (
    clay_24fps_path,
    depth_video_dir,
    depth_video_mp4_path,
    start_still_path,
)

ROOT = Path(__file__).resolve().parents[2]  # content-pipeline/
REPO = ROOT.parent
GATE_ROOT = REPO / "tmp" / "depth_gate"
PROMPTS_PATH = ROOT / "prompts.json"
COMFY_INPUT = ROOT / ".comfyui" / "input"
COMFY_OUTPUT = ROOT / ".comfyui" / "output"
GATE_SEED = 42
I2V_STRENGTH = 0.7
IC_STRENGTH = 1.0
GAMMA = 0.6


def scene_gate_dir(scene_number: int) -> Path:
    path = GATE_ROOT / f"scene_{int(scene_number):02d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_depth_png16(path: Path) -> np.ndarray:
    """Return float32 normalized depth in [0, 1]. Never use Pillow convert on I;16."""
    arr = np.asarray(Image.open(path), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    if float(arr.max()) > 255.0:
        return np.clip(arr / 65535.0, 0.0, 1.0)
    return np.clip(arr / 255.0, 0.0, 1.0)


def assert_depth_levels(gray: np.ndarray, label: str, minimum: int = 100) -> None:
    vals = np.asarray(gray, dtype=np.float32)
    if vals.max() <= 1.0 + 1e-6:
        valid = vals > 1e-6
        # Count at 16-bit quantization so smooth inverse maps aren't rejected after /255.
        quant = np.unique(np.rint(vals[valid] * 65535.0).astype(np.uint16)) if valid.any() else np.array([])
    else:
        valid = vals > 0
        quant = np.unique(np.rint(vals[valid]).astype(np.int32)) if valid.any() else np.array([])
    if quant.size == 0:
        raise RuntimeError(f"{label}: no valid depth pixels")
    if quant.size < minimum:
        raise RuntimeError(f"{label}: only {quant.size} distinct levels (need > {minimum})")


def decode_mp4_gray(path: Path, width: int = CLIP_WIDTH, height: int = CLIP_HEIGHT) -> list[np.ndarray]:
    ffmpeg = find_ffmpeg()
    raw = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    ).stdout
    frame_bytes = width * height * 3
    if len(raw) % frame_bytes:
        raise RuntimeError(f"{path}: decoded size {len(raw)} not multiple of {frame_bytes}")
    frames = []
    for offset in range(0, len(raw), frame_bytes):
        rgb = np.frombuffer(raw, dtype=np.uint8, count=frame_bytes, offset=offset).reshape(
            (height, width, 3)
        )
        frames.append(rgb[:, :, 0].astype(np.float32))
    return frames


def decode_mp4_rgb(path: Path, width: int = CLIP_WIDTH, height: int = CLIP_HEIGHT) -> list[np.ndarray]:
    ffmpeg = find_ffmpeg()
    raw = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    ).stdout
    frame_bytes = width * height * 3
    frames = []
    for offset in range(0, len(raw), frame_bytes):
        frames.append(
            np.frombuffer(raw, dtype=np.uint8, count=frame_bytes, offset=offset).reshape(
                (height, width, 3)
            ).copy()
        )
    return frames


def encode_norm_frames_mp4(norms: list[np.ndarray], dest: Path, *, gamma: float | None = None) -> None:
    """norms are float [0,1] at previs resolution; fit_to_clip then H.264 CRF 14."""
    rgb_frames: list[Image.Image] = []
    for norm in norms:
        encoded = np.clip(norm, 0.0, 1.0)
        if gamma is not None:
            encoded = np.power(encoded, gamma)
        gray8 = np.rint(encoded * 255.0).astype(np.uint8)
        rgb = Image.fromarray(np.stack((gray8, gray8, gray8), axis=-1), "RGB")
        fitted = fit_to_clip(rgb)
        assert_depth_levels(np.asarray(fitted)[:, :, 0].astype(np.float32), f"encode {dest.name}")
        rgb_frames.append(fitted)
    raw = b"".join(frame.tobytes() for frame in rgb_frames)
    encode_rgb_frames(raw, CLIP_WIDTH, CLIP_HEIGHT, FPS, dest, crf=14)


def write_gamma_depth_mp4(show_id: str, episode: int, scene_number: int) -> Path:
    depth_dir = depth_video_dir(show_id, episode, scene_number)
    pngs = sorted(depth_dir.glob("frame_*.png"))
    if not pngs:
        raise FileNotFoundError(f"missing depth PNGs in {depth_dir}")
    norms = [read_depth_png16(path) for path in pngs]
    assert_depth_levels(norms[0], f"png source scene {scene_number}")
    dest = scene_gate_dir(scene_number) / "depth_inverse_gamma06.mp4"
    encode_norm_frames_mp4(norms, dest, gamma=GAMMA)
    return dest


def write_linear_depth_mp4(show: dict, episode: dict, scene: dict) -> Path:
    """Temporarily set prompts.json mapping=linear, render to tmp, restore prompts."""
    from clip_spec import FPS as CLIP_FPS, clip_frame_count
    from spatial_previs import (
        PROXY_HEIGHT,
        PROXY_WIDTH,
        _depth_near_far,
        _normalize_depth_frame,
        load_depth_video_settings,
        render_blocked_frame,
    )

    original = PROMPTS_PATH.read_text(encoding="utf-8")
    data = json.loads(original)
    data.setdefault("depthVideo", {})["mapping"] = "linear"
    PROMPTS_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    try:
        settings = load_depth_video_settings()
        if settings["mapping"] != "linear":
            raise RuntimeError(f"expected linear mapping, got {settings['mapping']}")
        start, finish = (float(v) for v in scene["timeRangeSeconds"])
        count = clip_frame_count([start, finish])
        times = [start + i / CLIP_FPS for i in range(count)]
        zbufs = []
        for t in times:
            _img, zbuf, _people = render_blocked_frame(show, episode, scene, t)
            zbufs.append(np.asarray(zbuf, dtype=np.float32))
        near, far = _depth_near_far(zbufs, settings["nearPercentile"], settings["farPercentile"])
        norms = [_normalize_depth_frame(z, near, far, "linear") for z in zbufs]
        dest = scene_gate_dir(scene["sceneNumber"]) / "depth_linear.mp4"
        encode_norm_frames_mp4(norms, dest, gamma=None)
        meta = {
            "mapping": "linear",
            "near": near,
            "far": far,
            "frameCount": count,
            "sourceResolution": [PROXY_WIDTH, PROXY_HEIGHT],
            "clipResolution": [CLIP_WIDTH, CLIP_HEIGHT],
        }
        (scene_gate_dir(scene["sceneNumber"]) / "depth_linear_meta.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        return dest
    finally:
        PROMPTS_PATH.write_text(original, encoding="utf-8")


def depth_anything_workflow(clay_name: str, frame_count: int, prefix: str) -> dict:
    # Matches tmp/depth_preprocess.json from the spike (LoadVideo uses clay already at clip size).
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
            "inputs": {"model": "video_depth_anything_vits.pth"},
        },
        "3": {
            "class_type": "VideoDepthAnythingProcess",
            "inputs": {
                "vda_model": ["2", 0],
                "images": ["1", 0],
                "input_size": 518,
                "max_res": 1280,
                "precision": "fp16",
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
    # Ensure silent AAC so LoadVideo path is happy if anything probes audio later.
    return dest.name


def queue_and_download_video(graph: dict, dest: Path, prefix: str, timeout_s: int = 1800) -> dict:
    """Queue a non-LTX Comfy graph and copy the newest matching mp4 from Comfy output."""
    import time
    import urllib.error
    import urllib.request

    from generate_batch import COMFYUI_URL, free_comfy_models, queue_prompt

    free_comfy_models()
    prompt_id = queue_prompt(graph)
    started = time.time()
    peak = 0.0
    while time.time() - started < timeout_s:
        try:
            import subprocess

            used = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
            peak = max(peak, float(used.splitlines()[0]))
        except Exception:
            pass
        req = urllib.request.Request(f"{COMFYUI_URL}/history/{prompt_id}")
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
        raise TimeoutError(prompt_id)

    candidates = sorted(
        COMFY_OUTPUT.rglob(f"{prefix}*.mp4"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No Comfy output matching {prefix}*.mp4")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidates[0], dest)
    free_comfy_models()
    return {"peak_vram_mib": peak, "seconds": time.time() - started, "source": str(candidates[0])}


def mux_silent_aac(src: Path, dest: Path) -> None:
    ffmpeg = find_ffmpeg()
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


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    if denom < 1e-12:
        return float("nan")
    return float((ra * rb).sum() / denom)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB uint8 HxWx3 -> Lab float."""
    try:
        import cv2

        return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    except Exception:
        x = rgb.astype(np.float32) / 255.0
        # Approximate: treat as linear RGB->XYZ->Lab (good enough for ranking deltas)
        mask = x > 0.04045
        x = np.where(mask, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)
        r, g, b = x[..., 0], x[..., 1], x[..., 2]
        X = r * 0.4124 + g * 0.3576 + b * 0.1805
        Y = r * 0.2126 + g * 0.7152 + b * 0.0722
        Z = r * 0.0193 + g * 0.1192 + b * 0.9505
        X /= 0.95047
        Z /= 1.08883
        eps = 216 / 24389
        kappa = 24389 / 27

        def f(t: np.ndarray) -> np.ndarray:
            return np.where(t > eps, np.cbrt(t), (kappa * t + 16) / 116)

        fx, fy, fz = f(X), f(Y), f(Z)
        L = 116 * fy - 16
        A = 500 * (fx - fy)
        B = 200 * (fy - fz)
        return np.stack((L, A, B), axis=-1).astype(np.float32)
