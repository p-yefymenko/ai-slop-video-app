#!/usr/bin/env python3
"""Download LTX video weights plus the Qwen-Image-Edit still-generation stack."""

from __future__ import annotations

import json
import os
import shutil
import struct
import sys
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download, hf_hub_url

GGUF_REPO = "unsloth/LTX-2.3-GGUF"
GGUF_FILE = "distilled-1.1/ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf"
GGUF_NAME = "ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf"

VAE_REPO = "unsloth/LTX-2.3-GGUF"
VAE_FILE = "vae/ltx-2.3-22b-distilled_video_vae.safetensors"
VAE_NAME = "ltx-2.3-22b-distilled_video_vae.safetensors"
AUDIO_VAE_FILE = "vae/ltx-2.3-22b-distilled_audio_vae.safetensors"
AUDIO_VAE_NAME = "ltx-2.3-22b-distilled_audio_vae.safetensors"

META_REPO = "Lightricks/LTX-2.3"
META_FILE = "ltx-2.3-22b-distilled-1.1.safetensors"
STUB_NAME = "ltx-2.3-22b-distilled-api-id.safetensors"
OLD_STUB_NAMES = ("ltx-2-19b-distilled-api-id.safetensors",)

QWEN_GGUF_REPO = "unsloth/Qwen-Image-Edit-2511-GGUF"
QWEN_GGUF_FILE = "qwen-image-edit-2511-Q4_K_M.gguf"
QWEN_ENCODER_REPO = "Comfy-Org/Qwen-Image_ComfyUI"
QWEN_ENCODER_FILE = "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
QWEN_ENCODER_NAME = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
QWEN_VAE_REPO = "Comfy-Org/Qwen-Image_ComfyUI"
QWEN_VAE_FILE = "split_files/vae/qwen_image_vae.safetensors"
QWEN_VAE_NAME = "qwen_image_vae.safetensors"
QWEN_LORA_REPO = "lightx2v/Qwen-Image-Edit-2511-Lightning"
QWEN_LORA_FILE = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"


def cache_dir() -> Path:
    path = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "hf-hub"
    path.mkdir(parents=True, exist_ok=True)
    return path


def download(repo_id: str, filename: str, dest: Path, min_size: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size >= min_size:
        print(f"Already present: {dest}")
        return
    print(f"Downloading {filename} from {repo_id}...")
    src = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir())
    shutil.copy2(src, dest)
    print(dest)


def safetensors_header(data: bytes) -> dict:
    if len(data) < 8:
        raise RuntimeError("Safetensors header too short")
    (length,) = struct.unpack_from("<Q", data, 0)
    end = 8 + length
    if end > len(data):
        raise RuntimeError(f"Safetensors header is {length} bytes; fetched {len(data) - 8}")
    return json.loads(data[8:end].decode("utf-8"))


def fetch_bytes(url: str, nbytes: int) -> bytes:
    headers = {"Range": f"bytes=0-{nbytes - 1}"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with requests.get(url, headers=headers, timeout=120, allow_redirects=True, stream=True) as response:
        response.raise_for_status()
        chunks: list[bytes] = []
        remaining = nbytes
        for chunk in response.iter_content(64 * 1024):
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
            if remaining <= 0:
                break
    return b"".join(chunks)


def fetch_official_metadata() -> dict:
    url = hf_hub_url(repo_id=META_REPO, filename=META_FILE)
    print(f"Fetching official LTX checkpoint metadata from {META_REPO}/{META_FILE}...")
    nbytes = 4 * 1024 * 1024
    data = fetch_bytes(url, nbytes)
    if len(data) < 8:
        raise RuntimeError("Safetensors header too short")
    (length,) = struct.unpack_from("<Q", data, 0)
    need = 8 + length
    if need > len(data):
        data = fetch_bytes(url, need + 1024)
    header = safetensors_header(data)
    metadata = header.get("__metadata__") or {}
    if "encrypted_wandb_properties" not in metadata:
        raise RuntimeError("Official checkpoint header has no encrypted_wandb_properties metadata")
    return metadata


def write_metadata_stub(dest: Path, metadata: dict) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "__metadata__": metadata,
        "api_id_placeholder": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [0, 4],
        },
    }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((8 - (len(encoded) % 8)) % 8)
    dest.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\x00\x00\x00\x00")
    print(f"Wrote Gemma API model-id stub: {dest}")


def main() -> None:
    comfy_root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / ".comfyui")
    models = comfy_root / "models"
    download(GGUF_REPO, GGUF_FILE, models / "diffusion_models" / GGUF_NAME, 1_000_000_000)
    download(VAE_REPO, VAE_FILE, models / "vae" / VAE_NAME, 100_000_000)
    audio_vae = models / "vae" / AUDIO_VAE_NAME
    download(VAE_REPO, AUDIO_VAE_FILE, audio_vae, 50_000_000)
    # LTXVAudioVAELoader lists models/checkpoints, not models/vae.
    audio_vae_ckpt = models / "checkpoints" / AUDIO_VAE_NAME
    if not audio_vae_ckpt.exists() or audio_vae_ckpt.stat().st_size < 50_000_000:
        audio_vae_ckpt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(audio_vae, audio_vae_ckpt)
        print(f"Copied audio VAE into checkpoints for LTXVAudioVAELoader: {audio_vae_ckpt}")
    else:
        print(f"Already present: {audio_vae_ckpt}")
    download(QWEN_GGUF_REPO, QWEN_GGUF_FILE, models / "diffusion_models" / QWEN_GGUF_FILE, 1_000_000_000)
    download(QWEN_ENCODER_REPO, QWEN_ENCODER_FILE, models / "text_encoders" / QWEN_ENCODER_NAME, 1_000_000_000)
    download(QWEN_VAE_REPO, QWEN_VAE_FILE, models / "vae" / QWEN_VAE_NAME, 50_000_000)
    download(QWEN_LORA_REPO, QWEN_LORA_FILE, models / "loras" / QWEN_LORA_FILE, 100_000_000)
    checkpoints = models / "checkpoints"
    for old_name in OLD_STUB_NAMES:
        old_stub = checkpoints / old_name
        if old_stub.exists():
            old_stub.unlink()
            print(f"Removed stale Gemma API stub: {old_stub}")
    stub = checkpoints / STUB_NAME
    write_metadata_stub(stub, fetch_official_metadata())


if __name__ == "__main__":
    main()
