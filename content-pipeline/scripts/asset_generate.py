"""Draw the one picture of an object or character, or turn a reviewed picture into a mesh.

A character's picture is their reference photo (``reference.png``): the video
model's reference and Pixal3D's input, so the mesh and the look are one person.
It is drawn by hand in the Gemini app from words this stage writes
(``hand_drawn_picture``). An object's picture is its plate (``plate.png``),
drawn by Qwen-Image-2512, TRELLIS.2's input. Pixal3D shares TRELLIS.2's shape
VAE and adds pixel-aligned features so the front matches the photo; the mesh is
then painted from the photo it saw (``photo_paint``).
"""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path

from mesh_io import TRIANGLE_BUDGET

ROOT = Path(__file__).resolve().parents[1]
PROMPTS_PATH = ROOT / "prompts.json"
MESH_PREFIX = "reelshort_asset"
PICTURE_PREFIX = "reelshort_picture"
# The square crop of the reference photo Pixal3D saw, saved so the mesh can be painted from it.
CROP_PREFIX = "reelshort_asset_crop"

# Qwen-Image-2512 (Q8) draws every picture: it follows a description item by item
# and looks real. 50 steps, CFG 4, euler, AuraFlow shift 3.1: the model's own settings.
PICTURE_MODEL = "qwen-image-2512-Q8_0.gguf"
PICTURE_TEXT_ENCODER = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
PICTURE_VAE = "qwen_image_vae.safetensors"
PICTURE_STEPS = 50
PICTURE_CFG = 4.0
PICTURE_SHIFT = 3.1
# A standing person fills a 9:16 portrait; an object, Qwen-Image's native square.
CHARACTER_PICTURE_SIZE = (928, 1664)

OBJECT_PICTURE_SIZE = (1328, 1328)

# A character's photo is drawn over one fixed standing pose (InstantX ControlNet Union,
# OpenPose skeleton), so the whole figure, head to feet, is an input, not a request
# the model may ignore. One pose for everyone; height and build come from the words.
POSE_CONTROLNET = "Qwen-Image-InstantX-ControlNet-Union.safetensors"
POSE_NAME = "standing-front-v1"
# Who draws a picture: Qwen-Image here, or a person in the Gemini app from the words this stage writes.
QWEN_DRAWER = "qwen-image"
GEMINI_DRAWER = "gemini-app"
POSE_STRENGTH = 1.0
# Released for the last steps, so the pose fixes the figure and the words finish the detail.
POSE_END = 0.8
# The figure, head top to soles, as a share of the picture's height, and where it starts.
POSE_FIGURE_HEIGHT = 0.86
POSE_TOP = 0.06
# OpenPose body keypoints (BODY_18 order) of a person standing straight, facing the
# camera, arms at the sides: (x from the body's center line, y down from the head top),
# both in shares of the figure's height. The person's right is the picture's left.
STANDING_POSE = [
    (0.0, 0.075),     # nose
    (0.0, 0.135),     # neck
    (-0.12, 0.18),    # right shoulder
    (-0.13, 0.37),    # right elbow
    (-0.125, 0.52),   # right wrist
    (0.12, 0.18),     # left shoulder
    (0.13, 0.37),     # left elbow
    (0.125, 0.52),    # left wrist
    (-0.085, 0.5),    # right hip
    (-0.065, 0.72),   # right knee
    (-0.06, 0.955),   # right ankle
    (0.085, 0.5),     # left hip
    (0.065, 0.72),    # left knee
    (0.06, 0.955),    # left ankle
    (-0.025, 0.06),   # right eye
    (0.025, 0.06),    # left eye
    (-0.055, 0.07),   # right ear
    (0.055, 0.07),    # left ear
]
# OpenPose's limbs (pairs of keypoints) and colors, as its annotator draws them.
POSE_LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
              (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
POSE_COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
               (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
               (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]

TRELLIS_UNET = "trellis_2_int8_convrot.safetensors"
TRELLIS_DINO = "dino_v3_vit_l.safetensors"
TRELLIS_SHAPE_VAE = "trellis_2_shape_vae_bf16.safetensors"
PIXAL3D_UNET = "pixal3d_int8_convrot.safetensors"
# Same DINOv3 encoder, with the NAF upsampler bundled so features can be back-projected.
PIXAL3D_DINO = "dino_v3_L_naf_fp32.safetensors"
PIXAL3D_TEXTURE_VAE = "trellis_2_texture_vae_bf16.safetensors"
MOGE_MODEL = "moge_2_vitl_normal_fp16.safetensors"
BACKGROUND_MODEL = "birefnet.safetensors"
# The object cut out and cropped to a square this size, as TRELLIS.2 takes it.
TRELLIS_INPUT_SIZE = 1024
# Second shape pass. 1024 is the node default and fits a 16GB card with the int8 weights.
UPSAMPLE_RESOLUTION = 1024
# Pixal3D leaves a margin so the projected features line up with the photo.
PIXAL3D_PAD = 1.1


def object_prompt(appearance: str) -> str:
    """One isolated object (a landmark or a prop), from its words."""
    from describe import template

    text = " ".join(appearance.split())
    if not text:
        raise RuntimeError("appearance is empty")
    return template("objectPicture", {"appearance": text})


def standing_pose(size: tuple[int, int], dest: Path) -> Path:
    """Draw ``STANDING_POSE`` as an OpenPose skeleton on black, the figure centered and full height."""
    import math

    from PIL import Image, ImageDraw

    width, height = size
    figure = POSE_FIGURE_HEIGHT * height
    points = [(width / 2 + x * figure, POSE_TOP * height + y * figure) for x, y in STANDING_POSE]
    picture = Image.new("RGB", size, (0, 0, 0))
    limbs = Image.new("RGB", size, (0, 0, 0))
    draw = ImageDraw.Draw(limbs)
    stick = max(2, round(height / 220))
    for (a, b), color in zip(POSE_LIMBS, POSE_COLORS):
        (x0, y0), (x1, y1) = points[a], points[b]
        angle = math.atan2(y1 - y0, x1 - x0)
        across = (-math.sin(angle) * stick, math.cos(angle) * stick)
        draw.polygon([(x0 + across[0], y0 + across[1]), (x1 + across[0], y1 + across[1]),
                      (x1 - across[0], y1 - across[1]), (x0 - across[0], y0 - across[1])], fill=color)
    picture = Image.blend(picture, limbs, 0.6)
    draw = ImageDraw.Draw(picture)
    for (x, y), color in zip(points, POSE_COLORS):
        draw.ellipse((x - stick, y - stick, x + stick, y + stick), fill=color)
    dest.parent.mkdir(parents=True, exist_ok=True)
    picture.save(dest)
    return dest


def picture_graph(prompt: str, negative: str, size: tuple[int, int], seed: int, pose: str | None = None) -> dict:
    """Qwen-Image-2512 from words, over the ``pose`` skeleton (a ComfyUI input image) when given."""
    width, height = size
    graph = {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": PICTURE_MODEL}},
        "2": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": PICTURE_SHIFT}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": PICTURE_TEXT_ENCODER, "type": "qwen_image"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": prompt}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": negative}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": PICTURE_VAE}},
        "7": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "8": {"class_type": "KSampler", "inputs": {
            "model": ["2", 0], "seed": seed, "steps": PICTURE_STEPS, "cfg": PICTURE_CFG, "sampler_name": "euler",
            "scheduler": "simple", "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["7", 0], "denoise": 1.0}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["6", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": PICTURE_PREFIX}},
    }
    if pose is not None:
        graph["11"] = {"class_type": "LoadImage", "inputs": {"image": pose}}
        graph["12"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": POSE_CONTROLNET}}
        graph["13"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
            "positive": ["4", 0], "negative": ["5", 0], "control_net": ["12", 0], "image": ["11", 0],
            "strength": POSE_STRENGTH, "start_percent": 0.0, "end_percent": POSE_END, "vae": ["6", 0]}}
        graph["8"]["inputs"]["positive"] = ["13", 0]
        graph["8"]["inputs"]["negative"] = ["13", 1]
    return graph


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
                "width": TRELLIS_INPUT_SIZE,
                "height": TRELLIS_INPUT_SIZE,
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
    graph["40"] = {"class_type": "SaveImage", "inputs": {"images": ["4", 0], "filename_prefix": CROP_PREFIX}}
    graph["41"] = {"class_type": "PreviewAny", "inputs": {"source": ["32", 0]}}
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


def generate_picture(prompt: str, negative: str, size: tuple[int, int], dest: Path, pose: bool = False) -> Path:
    """Draw ``dest`` from words (over the standing pose when ``pose``) at a seed fixed by them, and log
    what it was drawn from. ComfyUI must be running."""
    from comfy import inputs_dir, stable_seed, stage_input, write_log

    seed = stable_seed("reference", prompt)
    files = {}
    staged = None
    if pose:
        files["pose"] = standing_pose(size, inputs_dir(dest) / "pose.png")
        staged = stage_input(files["pose"], "picture_pose")
    graph = picture_graph(prompt, negative, size, seed, staged)
    _run(graph, PICTURE_PREFIX, dest, "Qwen-Image-2512 picture", PICTURE_TIMEOUT_SECONDS)
    write_log(dest, {"prompt": prompt, "negative": negative, "size": list(size), "pose": POSE_NAME if pose else None, "seed": seed}, files)
    return dest


def picture_is_current(dest: Path, prompt: str, negative: str, pose: bool = False, drawer: str = QWEN_DRAWER) -> bool:
    """Whether ``dest`` exists and was drawn by ``drawer`` from these words (Qwen-Image over the
    standing pose, when ``pose``; by hand, the pose is in the words)."""
    from comfy import inputs_dir, present

    log = inputs_dir(dest) / "log.json"
    if not (present(dest) and log.is_file()):
        return False
    made = json.loads(log.read_text(encoding="utf-8"))
    over = POSE_NAME if pose and drawer == QWEN_DRAWER else None
    return (made.get("prompt"), made.get("negative"), made.get("pose"), made.get("drawer", QWEN_DRAWER)) == (prompt, negative, over, drawer)


# What a picture drawn by hand in the image model's app may be saved as, beside its own name.
HAND_DRAWN_FORMATS = (".png", ".jpg", ".jpeg", ".webp")


def hand_drawn_picture(prompt: str, negative: str, size: tuple[int, int], dest: Path, pose: bool) -> Path | None:
    """A picture drawn by hand in the Gemini app (the draw-stills skill): ``to_draw/prompt.txt``
    beside ``dest`` holds its words (``handPicture``, its size, then ``prompt``; the app takes no
    negative). A picture saved under ``dest``'s name in any of ``HAND_DRAWN_FORMATS`` after those
    words were written becomes ``dest`` (PNG), logged as drawn from them; until then the request
    is written (kept as it is when its words are already these) and None is returned."""
    from PIL import Image

    from comfy import write_log
    from describe import template

    del pose  # the standing pose is in the words
    request = dest.parent / "to_draw" / "prompt.txt"
    words = template("handPicture", {"width": str(size[0]), "height": str(size[1])}) + " " + prompt
    if request.is_file() and request.read_text(encoding="utf-8").strip() == words:
        asked = request.stat().st_mtime
        saved = [path for path in (dest.with_suffix(suffix) for suffix in HAND_DRAWN_FORMATS) if path.is_file() and path.stat().st_mtime > asked]
        if len(saved) > 1:
            raise RuntimeError(f"Keep one drawn picture, not {', '.join(path.name for path in saved)}")
        if saved:
            picture = Image.open(saved[0]).convert("RGB")
            picture.save(dest)
            if saved[0] != dest:
                saved[0].unlink()
            write_log(dest, {"prompt": prompt, "negative": negative, "size": list(picture.size), "pose": None, "drawer": GEMINI_DRAWER})
            return dest
    else:
        request.parent.mkdir(parents=True, exist_ok=True)
        request.write_text(words + "\n", encoding="utf-8")
    print(f"  To draw by hand: the words in {request}; save the picture beside it as {dest.parent / dest.stem}.<png|jpg|webp>", flush=True)
    return None


def draw_picture(prompt: str, negative: str, size: tuple[int, int], dest: Path, pose: bool) -> Path | None:
    """A character's reference photo is drawn by hand in Gemini (``hand_drawn_picture``): Qwen-Image's
    photos lost detail, and Pixal3D's mesh and its colors carry whatever the photo has. An object's
    plate is drawn by Qwen-Image (``generate_picture``)."""
    if pose:
        return hand_drawn_picture(prompt, negative, size, dest, pose)
    return generate_picture(prompt, negative, size, dest, pose)


def generate_asset_mesh(
    appearance: str,
    raw_dir: Path,
    triangle_budget: int = TRIANGLE_BUDGET,
    *,
    character: bool = False,
) -> Path:
    """Turn ``picture.png`` in ``raw_dir`` into ``model.glb``. ComfyUI must already be running."""
    from comfy import present, stable_seed, stage_input

    picture = raw_dir / "picture.png"
    if not present(picture):
        raise RuntimeError(
            f"No reviewed picture at {picture}. "
            "Run `pnpm run content:plates` and check the image before meshing."
        )
    staged = stage_input(picture, "asset_picture")
    # Every mesh keeps the picture's color. A character's is a frontal photo,
    # so its mesh is pixel-aligned to it; an object keeps TRELLIS.2's upright frame.
    seed = stable_seed("asset", " ".join(appearance.split()))
    if not character:
        return _run(object_graph(staged, seed, triangle_budget), MESH_PREFIX, raw_dir / "model.glb", "TRELLIS.2 textured mesh", MESH_TIMEOUT_SECONDS)
    return _character_mesh(pixal3d_graph(staged, seed, triangle_budget), raw_dir)


def _character_mesh(graph: dict, raw_dir: Path) -> Path:
    """Run Pixal3D, keep the crop it saw (``photo_crop.png``) and its field of view
    (``photo_camera.json``), and paint the mesh from the crop (``photo_paint``)."""
    import numpy as np
    from PIL import Image

    from comfy import download_output, free_comfy_models, output_named, queue_prompt, wait_for_files
    from mesh_io import read_schema_mesh, read_vertex_colors, write_schema_glb
    from photo_paint import paint_from_photo, surface_depth

    mesh, crop_path = raw_dir / "model.glb", raw_dir / "photo_crop.png"
    print("  Pixal3D mesh...", flush=True)
    try:
        free_comfy_models()
        files, item = wait_for_files(queue_prompt(graph), MESH_TIMEOUT_SECONDS)
    except urllib.error.URLError as exc:
        raise RuntimeError(_comfy_unreachable()) from exc
    download_output(output_named(files, MESH_PREFIX), mesh)
    download_output(output_named(files, CROP_PREFIX), crop_path)
    fov = float(str((item["outputs"]["41"].get("text") or [""])[0]).strip())
    (raw_dir / "photo_camera.json").write_text(json.dumps({"fovDegrees": fov}, indent=2), encoding="utf-8")
    vertices, faces = read_schema_mesh(mesh)
    colors = read_vertex_colors(mesh)
    if colors is None:
        colors = np.zeros_like(vertices, dtype=np.float32)
    crop = np.asarray(Image.open(crop_path).convert("RGB"))
    painted = paint_from_photo(vertices, colors, crop, fov, surface_depth(vertices, faces, fov, crop.shape[0]))
    write_schema_glb(mesh, vertices, faces, painted)
    return mesh


def _comfy_unreachable() -> str:
    return (
        "ComfyUI is not reachable at http://127.0.0.1:8188. "
        "Run `pnpm run content:models`, then leave `pnpm run content:comfy` running."
    )
