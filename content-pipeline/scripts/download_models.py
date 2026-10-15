#!/usr/bin/env python3
"""Download every model the pipeline runs: the Wan 2.2 chain, the character and object pictures, and the meshes.

    download_models.py [<comfy root>]

Each file lands in ComfyUI's ``models/<folder>`` under its own name; one already
there (at least its expected size) is kept.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

WAN = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
WAN21 = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"
VACE = "QuantStack/Wan2.2-VACE-Fun-A14B-GGUF"
QWEN = "Comfy-Org/Qwen-Image_ComfyUI"
TRELLIS = "Comfy-Org/TRELLIS.2"
PIXAL3D = "Comfy-Org/Pixal3D"
GB, MB = 1_000_000_000, 1_000_000

# (repository, file in it, models folder, smallest plausible size in bytes)
MODELS = [
    # Wan 2.2 VACE (stills and takes without a speaker), two experts at Q8.
    (VACE, "HighNoise/Wan2.2-VACE-Fun-A14B-high-noise-Q8_0.gguf", "diffusion_models", 10 * GB),
    (VACE, "LowNoise/Wan2.2-VACE-Fun-A14B-low-noise-Q8_0.gguf", "diffusion_models", 10 * GB),
    # Wan 2.2 S2V (talking takes) and its audio encoder.
    (WAN, "split_files/diffusion_models/wan2.2_s2v_14B_fp8_scaled.safetensors", "diffusion_models", 10 * GB),
    (WAN, "split_files/audio_encoders/wav2vec2_large_english_fp16.safetensors", "audio_encoders", 500 * MB),
    # Shared by every Wan model.
    (WAN21, "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "text_encoders", 5 * GB),
    (WAN, "split_files/vae/wan_2.1_vae.safetensors", "vae", 200 * MB),
    # Qwen-Image-2512: every picture (character reference photos, object plates).
    ("unsloth/Qwen-Image-2512-GGUF", "qwen-image-2512-Q8_0.gguf", "diffusion_models", 15 * GB),
    (QWEN, "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", "text_encoders", 5 * GB),
    (QWEN, "split_files/vae/qwen_image_vae.safetensors", "vae", 100 * MB),
    # InstantX ControlNet Union for Qwen-Image: a character's reference photo is drawn over one fixed standing pose.
    ("Comfy-Org/Qwen-Image-InstantX-ControlNets", "split_files/controlnet/Qwen-Image-InstantX-ControlNet-Union.safetensors", "controlnet", 3 * GB),
    # TRELLIS.2 (object meshes) and Pixal3D (character meshes).
    (TRELLIS, "diffusion_models/trellis_2_int8_convrot.safetensors", "diffusion_models", 5 * GB),
    (TRELLIS, "clip_vision/dino_v3_vit_l.safetensors", "clip_vision", 1 * GB),
    (TRELLIS, "vae/trellis_2_shape_vae_bf16.safetensors", "vae", 1 * GB),
    (PIXAL3D, "diffusion_models/pixal3d_int8_convrot.safetensors", "diffusion_models", 5 * GB),
    (PIXAL3D, "clip_vision/dino_v3_L_naf_fp32.safetensors", "clip_vision", 1 * GB),
    (PIXAL3D, "vae/trellis_2_texture_vae_bf16.safetensors", "vae", 800 * MB),
    ("Comfy-Org/MoGe", "geometry_estimation/moge_2_vitl_normal_fp16.safetensors", "geometry_estimation", 600 * MB),
    ("Comfy-Org/BiRefNet", "background_removal/birefnet.safetensors", "background_removal", 400 * MB),
]


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


def main() -> None:
    comfy_root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / ".comfyui")
    for repo_id, filename, folder, min_size in MODELS:
        download(repo_id, filename, comfy_root / "models" / folder / Path(filename).name, min_size)
    # RIFE's weights are fetched by ComfyUI-Frame-Interpolation the first time it runs.


if __name__ == "__main__":
    main()
