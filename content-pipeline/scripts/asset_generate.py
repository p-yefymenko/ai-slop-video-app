"""Draw one object or character with Qwen, or turn a reviewed picture into a mesh.

A landmark or prop uses TRELLIS.2. A character uses Pixal3D, which shares
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
PLATE_PREFIX = "reelshort_asset_plate"
MESH_PREFIX = "reelshort_asset"

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


def object_text(appearance: str, materials: str) -> str:
    """A landmark's own words plus its location's shared materials."""
    prompts = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    template = prompts.get("landmarkMaterials")
    if not isinstance(template, str) or "{materials}" not in template:
        raise RuntimeError(f"{PROMPTS_PATH} is missing a landmarkMaterials template with {{materials}}")
    return f"{' '.join(appearance.split())} {template.replace('{materials}', ' '.join(materials.split()))}"


def plate_prompt(appearance: str, *, character: bool = False) -> str:
    """A character, or one isolated object (a landmark or a prop)."""
    prompts = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    key = "characterPlate" if character else "landmarkPlate"
    template = prompts.get(key)
    token = "{description}" if character else "{appearance}"
    if not isinstance(template, str) or token not in template:
        raise RuntimeError(f"{PROMPTS_PATH} is missing a {key} template with {token}")
    text = " ".join(appearance.split())
    if not text:
        raise RuntimeError("appearance is empty")
    return template.replace(token, text)


def object_graph(image_name: str, seed: int, triangle_budget: int = TRIANGLE_BUDGET) -> dict:
    """TRELLIS.2 shape in its upright, canonical frame, then its texture painted on.

    Pixel alignment would build the object in the plate's camera frame, tilted
    by however far above it the plate was drawn from.
    """
    graph = _shape_graph(image_name, seed, triangle_budget, pixel_aligned=False)
    _paint_plate_texture(graph, seed)
    return graph


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
            "inputs": {"mesh": ["22", 0], "filename_prefix": MESH_PREFIX},
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


def look_prompt(effects: list[str]) -> str:
    """The plate as it looks in the show: the same object, with its effects added."""
    prompts = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    template = prompts.get("lookPlate")
    if not isinstance(template, str) or "{effects}" not in template:
        raise RuntimeError(f"{PROMPTS_PATH} is missing a lookPlate template with {{effects}}")
    return template.replace("{effects}", "; ".join(" ".join(text.split()) for text in effects))


# A picture takes minutes, a mesh longer; a run past these has hung.
PICTURE_TIMEOUT_SECONDS = 1800
MESH_TIMEOUT_SECONDS = 3600


def _run(graph: dict, prefix: str, dest: Path, label: str, timeout_s: float) -> Path:
    from comfy import free_comfy_models, run_graph

    try:
        free_comfy_models()
        run_graph(graph, {prefix: dest}, label, timeout_s)
    except urllib.error.URLError as exc:
        raise RuntimeError(_comfy_unreachable()) from exc
    return dest


def generate_look_plate(plate: Path, effects: list[str], dest: Path) -> Path:
    """Write ``look.png``: the reviewed plate with its effects drawn in. ComfyUI must be running."""
    from comfy import clone_workflow, stable_seed, stage_input

    prompt = look_prompt(effects)
    graph = clone_workflow(json.loads(PLATE_WORKFLOW.read_text(encoding="utf-8")))
    graph["6"]["inputs"]["image"] = stage_input(plate, "asset_look_source")
    graph["7"]["inputs"]["prompt"] = prompt
    graph["10"]["inputs"]["seed"] = stable_seed("look", prompt)
    return _run(graph, PLATE_PREFIX, dest, "Qwen look plate", PICTURE_TIMEOUT_SECONDS)


def generate_reference(character: dict, dest: Path) -> Path:
    """Write a character's ``reference.png`` with Qwen-Image-2512, from their words, at a seed fixed by them."""
    from comfy import stable_seed, write_log
    from describe import reference_prompt
    from wan import REFERENCE_PREFIX, reference_graph
    from world import settings

    prompt = reference_prompt(character)
    negative = settings()["referenceNegative"]
    seed = stable_seed("reference", prompt)
    _run(reference_graph(prompt, negative, seed), REFERENCE_PREFIX, dest, "Qwen-Image-2512 reference", PICTURE_TIMEOUT_SECONDS)
    write_log(dest, {"prompt": prompt, "negative": negative, "seed": seed})
    return dest


def reference_is_current(character: dict, dest: Path) -> bool:
    """Whether ``reference.png`` exists and was drawn from the character's current words."""
    from comfy import inputs_dir, present
    from describe import reference_prompt

    log = inputs_dir(dest) / "log.json"
    return present(dest) and log.is_file() and json.loads(log.read_text(encoding="utf-8")).get("prompt") == reference_prompt(character)


def plate_is_ready(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 1024


def generate_asset_plate(appearance: str, raw_dir: Path, *, character: bool = False) -> Path:
    """Write ``plate.png``, the picture the mesh is built from, and stop. ComfyUI must already be running."""
    from comfy import clone_workflow, stable_seed, stage_input, write_solid_png

    raw_dir.mkdir(parents=True, exist_ok=True)
    blank = raw_dir / "blank.png"
    write_solid_png(blank, (210, 210, 210), PLATE_SIZE, PLATE_SIZE)
    graph = clone_workflow(json.loads(PLATE_WORKFLOW.read_text(encoding="utf-8")))
    graph["6"]["inputs"]["image"] = stage_input(blank, "asset_blank")
    graph["7"]["inputs"]["prompt"] = plate_prompt(appearance, character=character)
    graph["10"]["inputs"]["seed"] = stable_seed("asset", " ".join(appearance.split()))
    label = "Qwen character plate" if character else "Qwen object plate"
    return _run(graph, PLATE_PREFIX, raw_dir / "plate.png", label, PICTURE_TIMEOUT_SECONDS)


def generate_asset_mesh(
    appearance: str,
    raw_dir: Path,
    triangle_budget: int = TRIANGLE_BUDGET,
    *,
    character: bool = False,
) -> Path:
    """Turn an existing ``plate.png`` into ``model.glb``. ComfyUI must already be running."""
    from comfy import stable_seed, stage_input

    plate_path = raw_dir / "plate.png"
    if not plate_is_ready(plate_path):
        raise RuntimeError(
            f"No reviewed plate at {plate_path}. "
            "Run `pnpm run content:plates` and check the image before meshing."
        )
    staged = stage_input(plate_path, "asset_plate")
    # Every mesh keeps the plate's color. A character plate is a frontal photo,
    # so its mesh is pixel-aligned to it; an object keeps TRELLIS.2's upright frame.
    graph = (pixal3d_graph if character else object_graph)(staged, stable_seed("asset", " ".join(appearance.split())), triangle_budget)
    label = "Pixal3D mesh" if character else "TRELLIS.2 textured mesh"
    return _run(graph, MESH_PREFIX, raw_dir / "model.glb", label, MESH_TIMEOUT_SECONDS)


def _comfy_unreachable() -> str:
    return (
        "ComfyUI is not reachable at http://127.0.0.1:8188. "
        "Run `pnpm run content:models`, then leave `pnpm run content:comfy` running."
    )
