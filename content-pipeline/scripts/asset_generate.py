"""Draw one place with Qwen, or turn a reviewed picture into a mesh.

A place or a landmark uses TRELLIS.2. A character uses Pixal3D, which shares
that shape VAE and adds pixel-aligned features so the front matches the plate.
"""

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
PIXAL3D_UNET = "pixal3d_int8_convrot.safetensors"
# Same DINOv3 encoder, with the NAF upsampler bundled so features can be back-projected.
PIXAL3D_DINO = "dino_v3_L_naf_fp32.safetensors"
PIXAL3D_TEXTURE_VAE = "trellis_2_texture_vae_bf16.safetensors"
MOGE_MODEL = "moge_2_vitl_normal_fp16.safetensors"
BACKGROUND_MODEL = "birefnet.safetensors"
PLATE_SIZE = 1024
# Second shape pass. 1024 is the node default and fits a 16GB card with the int8 weights.
UPSAMPLE_RESOLUTION = 1024
# Pixal3D leaves a margin so the projected features line up with the photo.
PIXAL3D_PAD = 1.1


def plate_prompt(appearance: str, *, landmark: bool = False, character: bool = False) -> str:
    prompts = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    if character:
        key = "characterPlate"
    elif landmark:
        key = "landmarkPlate"
    else:
        key = "locationPlate"
    template = prompts.get(key)
    token = "{description}" if character else "{appearance}"
    if not isinstance(template, str) or token not in template:
        raise RuntimeError(f"{PROMPTS_PATH} is missing a {key} template with {token}")
    text = " ".join(appearance.split())
    if not text:
        raise RuntimeError("appearance is empty")
    return template.replace(token, text)


def trellis_graph(image_name: str, seed: int, triangle_budget: int = TRIANGLE_BUDGET) -> dict:
    """Image to a shape mesh. Location meshes stay untextured."""
    return _shape_graph(image_name, seed, triangle_budget, pixel_aligned=False)


def pixal3d_graph(image_name: str, seed: int, triangle_budget: int = TRIANGLE_BUDGET) -> dict:
    """Same cascade as TRELLIS.2, then the plate texture painted onto the mesh."""
    graph = _shape_graph(image_name, seed, triangle_budget, pixel_aligned=True)
    _paint_plate_texture(graph, seed)
    return graph


def _shape_graph(
    image_name: str, seed: int, triangle_budget: int, *, pixel_aligned: bool
) -> dict:
    graph = {
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
                "target_face_count": triangle_budget,
                "placement_mode": "midpoint",
            },
        },
        "23": {
            "class_type": "SaveGLB",
            "inputs": {"mesh": ["22", 0], "filename_prefix": "reelshort_asset"},
        },
    }
    if not pixel_aligned:
        return graph
    graph["4"]["inputs"]["pad_factor"] = PIXAL3D_PAD
    graph["5"]["inputs"]["clip_name"] = PIXAL3D_DINO
    graph["6"] = {
        "class_type": "Pixal3DConditioning",
        "inputs": {
            "clip_vision_model": ["5", 0],
            "image": ["4", 0],
            "camera_angle_x": ["32", 0],
        },
    }
    graph["7"]["inputs"]["unet_name"] = PIXAL3D_UNET
    graph["30"] = {
        "class_type": "LoadMoGeModel",
        "inputs": {"model_name": MOGE_MODEL},
    }
    graph["31"] = {
        "class_type": "MoGeInference",
        "inputs": {
            "moge_model": ["30", 0],
            "image": ["4", 0],
            "resolution_level": 9,
            "fov_x_degrees": 0.0,
            "batch_size": 1,
            "force_projection": True,
            "apply_mask": True,
        },
    }
    graph["32"] = {
        "class_type": "MoGeGeometryToFOV",
        "inputs": {"moge_geometry": ["31", 0], "axis": "horizontal", "unit": "degrees"},
    }
    return graph


def _paint_plate_texture(graph: dict, seed: int) -> None:
    """Sample the plate's texture and store it on the mesh vertices.

    The full mesh is kept. Unwrapping it for an image atlas does not fit the GPU.
    """
    graph["33"] = {
        "class_type": "VAELoader",
        "inputs": {"vae_name": PIXAL3D_TEXTURE_VAE},
    }
    graph["34"] = {
        "class_type": "Trellis2TextureStage",
        "inputs": {
            "positive": ["19", 0],
            "negative": ["19", 1],
            "shape_latent": ["20", 0],
        },
    }
    graph["35"] = {
        "class_type": "KSampler",
        "inputs": {
            "seed": seed,
            "steps": 12,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
            "model": ["7", 0],
            "positive": ["34", 0],
            "negative": ["34", 1],
            "latent_image": ["34", 2],
        },
    }
    graph["36"] = {
        "class_type": "VaeDecodeTextureTrellis",
        "inputs": {
            "samples": ["35", 0],
            "vae": ["33", 0],
            "shape_subdivides": ["21", 1],
        },
    }
    graph["37"] = {
        "class_type": "PaintMesh",
        "inputs": {"mesh": ["22", 0], "voxel_colors": ["36", 0]},
    }
    graph["23"]["inputs"]["mesh"] = ["37", 0]


def plate_is_ready(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 1024


def generate_asset_plate(
    appearance: str, raw_dir: Path, *, landmark: bool = False, character: bool = False
) -> Path:
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
    plate_graph["7"]["inputs"]["prompt"] = plate_prompt(
        appearance, landmark=landmark, character=character
    )
    plate_graph["10"]["inputs"]["seed"] = seed
    plate_path = raw_dir / "plate.png"
    try:
        free_comfy_models()
        execute_queued_graph(
            plate_graph,
            plate_path,
            prefer="image",
            mode=(
                "Qwen character plate"
                if character
                else "Qwen landmark plate"
                if landmark
                else "Qwen location plate"
            ),
        )
    except urllib.error.URLError as exc:
        raise RuntimeError(_comfy_unreachable()) from exc
    return plate_path


def generate_asset_mesh(
    appearance: str,
    raw_dir: Path,
    triangle_budget: int = TRIANGLE_BUDGET,
    *,
    character: bool = False,
) -> Path:
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
    staged = stage_named_image(plate_path, "asset_plate")
    graph = (
        pixal3d_graph(staged, seed, triangle_budget)
        if character
        else trellis_graph(staged, seed, triangle_budget)
    )
    try:
        free_comfy_models()
        execute_queued_graph(
            graph,
            mesh_path,
            prefer="mesh",
            mode="Pixal3D mesh" if character else "TRELLIS.2 mesh",
        )
    except urllib.error.URLError as exc:
        raise RuntimeError(_comfy_unreachable()) from exc
    return mesh_path


def _comfy_unreachable() -> str:
    return (
        "ComfyUI is not reachable at http://127.0.0.1:8188. "
        "Run `pnpm run content:models`, then leave `pnpm run content:comfy` running."
    )
