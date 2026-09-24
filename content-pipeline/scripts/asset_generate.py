"""Draw one place with Qwen, or turn a reviewed picture into a mesh with TRELLIS.2."""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path

from mesh_io import TRIANGLE_BUDGET

ROOT = Path(__file__).resolve().parents[1]
PROMPTS_PATH = ROOT / "prompts.json"
PLATE_WORKFLOW = ROOT / "workflows" / "qwen_asset_plate.json"

TRELLIS_UNET = "trellis_2_int8_convrot.safetensors"
TRELLIS_DINO = "dino_v3_vit_l.safetensors"
TRELLIS_SHAPE_VAE = "trellis_2_shape_vae_bf16.safetensors"
BACKGROUND_MODEL = "birefnet.safetensors"
PLATE_SIZE = 1024
# Second shape pass. 1024 is the node default and fits a 16GB card with the int8 weights.
UPSAMPLE_RESOLUTION = 1024


def asset_plate_prompt(appearance: str) -> str:
    prompts = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    template = prompts.get("assetPlate")
    if not isinstance(template, str) or "{appearance}" not in template:
        raise RuntimeError(f"{PROMPTS_PATH} is missing an assetPlate template with {{appearance}}")
    text = " ".join(appearance.split())
    if not text:
        raise RuntimeError("appearance is empty")
    return template.replace("{appearance}", text)


def trellis_graph(image_name: str, seed: int) -> dict:
    """Image to a shape mesh. Texture baking stays off; previs stores an untextured GLB."""
    return {
        "1": {
            "class_type": "LoadImage",
            "inputs": {"image": image_name},
        },
        "2": {
            "class_type": "LoadBackgroundRemovalModel",
            "inputs": {"bg_removal_name": BACKGROUND_MODEL},
        },
        "3": {
            "class_type": "RemoveBackground",
            "inputs": {"bg_removal_model": ["2", 0], "image": ["1", 0]},
        },
        "4": {
            "class_type": "ImageCropToMask",
            "inputs": {
                "images": ["1", 0],
                "masks": ["3", 0],
                "width": PLATE_SIZE,
                "height": PLATE_SIZE,
                "pad_factor": 1.0,
                "grow_mask": 0,
                "background": "#000000",
            },
        },
        "5": {
            "class_type": "CLIPVisionLoader",
            "inputs": {"clip_name": TRELLIS_DINO},
        },
        "6": {
            "class_type": "Trellis2Conditioning",
            "inputs": {"clip_vision_model": ["5", 0], "image": ["4", 0]},
        },
        "7": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": TRELLIS_UNET, "weight_dtype": "default"},
        },
        "8": {
            "class_type": "CFGOverride",
            "inputs": {"model": ["7", 0], "cfg": 1.0, "start_percent": 0.667, "end_percent": 1.0},
        },
        "9": {
            "class_type": "RescaleCFG",
            "inputs": {"model": ["8", 0], "multiplier": 0.7},
        },
        "10": {
            "class_type": "ModelSamplingSD3",
            "inputs": {"model": ["9", 0], "shift": 5.0},
        },
        "11": {
            "class_type": "EmptyTrellis2LatentStructure",
            "inputs": {"batch_size": 1},
        },
        "12": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": 12,
                "cfg": 7.5,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["10", 0],
                "positive": ["6", 0],
                "negative": ["6", 1],
                "latent_image": ["11", 0],
            },
        },
        "13": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": TRELLIS_SHAPE_VAE},
        },
        "14": {
            "class_type": "VaeDecodeStructureTrellis2",
            "inputs": {"samples": ["12", 0], "vae": ["13", 0], "resolution": "32"},
        },
        "15": {
            "class_type": "Trellis2ShapeStage",
            "inputs": {"positive": ["6", 0], "negative": ["6", 1], "voxel": ["14", 0]},
        },
        "16": {
            "class_type": "CFGOverride",
            "inputs": {"model": ["7", 0], "cfg": 1.0, "start_percent": 0.769, "end_percent": 1.0},
        },
        "17": {
            "class_type": "RescaleCFG",
            "inputs": {"model": ["16", 0], "multiplier": 0.5},
        },
        "18": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": 20,
                "cfg": 7.5,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["17", 0],
                "positive": ["15", 0],
                "negative": ["15", 1],
                "latent_image": ["15", 2],
            },
        },
        "19": {
            "class_type": "Trellis2UpsampleStage",
            "inputs": {
                "positive": ["15", 0],
                "negative": ["15", 1],
                "shape_latent": ["18", 0],
                "vae": ["13", 0],
                "target_resolution": UPSAMPLE_RESOLUTION,
            },
        },
        "20": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": 12,
                "cfg": 7.5,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
                "model": ["17", 0],
                "positive": ["19", 0],
                "negative": ["19", 1],
                "latent_image": ["19", 2],
            },
        },
        "21": {
            "class_type": "VaeDecodeShapeTrellis",
            "inputs": {"samples": ["20", 0], "vae": ["13", 0]},
        },
        "22": {
            "class_type": "DecimateMesh",
            "inputs": {
                "mesh": ["21", 0],
                "target_face_count": TRIANGLE_BUDGET,
                "placement_mode": "midpoint",
            },
        },
        "23": {
            "class_type": "SaveGLB",
            "inputs": {"mesh": ["22", 0], "filename_prefix": "reelshort_asset"},
        },
    }


def plate_is_ready(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 1024


def generate_asset_plate(appearance: str, raw_dir: Path) -> Path:
    """Write ``plate.png`` and stop. ComfyUI must already be running."""
    from generate_batch import (
        clone_workflow,
        execute_queued_graph,
        free_comfy_models,
        stable_seed,
        stage_named_image,
        write_solid_png,
    )

    raw_dir.mkdir(parents=True, exist_ok=True)
    seed = stable_seed("asset", " ".join(appearance.split()))
    blank = raw_dir / "blank.png"
    write_solid_png(blank, (210, 210, 210), PLATE_SIZE, PLATE_SIZE)
    plate_graph = clone_workflow(json.loads(PLATE_WORKFLOW.read_text(encoding="utf-8")))
    plate_graph["6"]["inputs"]["image"] = stage_named_image(blank, "asset_blank")
    plate_graph["7"]["inputs"]["prompt"] = asset_plate_prompt(appearance)
    plate_graph["10"]["inputs"]["seed"] = seed
    plate_path = raw_dir / "plate.png"
    try:
        free_comfy_models()
        execute_queued_graph(plate_graph, plate_path, prefer="image", mode="Qwen asset plate")
    except urllib.error.URLError as exc:
        raise RuntimeError(_comfy_unreachable()) from exc
    return plate_path


def generate_asset_mesh(appearance: str, raw_dir: Path) -> Path:
    """Turn an existing ``plate.png`` into ``model.glb``. ComfyUI must already be running."""
    from generate_batch import (
        execute_queued_graph,
        free_comfy_models,
        stable_seed,
        stage_named_image,
    )

    plate_path = raw_dir / "plate.png"
    if not plate_is_ready(plate_path):
        raise RuntimeError(
            f"No reviewed plate at {plate_path}. "
            "Run `pnpm run content:plates` and check the image before meshing."
        )
    mesh_path = raw_dir / "model.glb"
    seed = stable_seed("asset", " ".join(appearance.split()))
    try:
        free_comfy_models()
        execute_queued_graph(
            trellis_graph(stage_named_image(plate_path, "asset_plate"), seed),
            mesh_path,
            prefer="mesh",
            mode="TRELLIS.2 mesh",
        )
    except urllib.error.URLError as exc:
        raise RuntimeError(_comfy_unreachable()) from exc
    return mesh_path


def _comfy_unreachable() -> str:
    return (
        "ComfyUI is not reachable at http://127.0.0.1:8188. "
        "Run `pnpm run content:models`, then leave `pnpm run content:comfy` running."
    )
