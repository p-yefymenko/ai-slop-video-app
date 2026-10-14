"""ComfyUI graphs for the Wan 2.2 chain, and the settings every scene shares.

    character reference (Qwen-Image-2512, ``reference_graph``)
      -> start still (Wan VACE: the references + the shot's depth, ``vace_graph`` with ``still=True``)
      -> clip: an on-screen speaker's face -> Wan S2V from the still and the voice (``s2v_graph``);
               anything else -> Wan VACE over the whole take (``vace_graph``)
      -> 24 fps (RIFE, ``rife_graph``)

Each graph follows ComfyUI's own template for its model, at full steps and real
CFG: the few-step LoRAs flattened the light and the acting. Every value here is
the same for every scene; a value that would be right for one scene only does
not belong in this file.
"""

from __future__ import annotations

import math

# Wan 2.2 14B is trained at 16 fps, and S2V ties its audio windows to it.
FPS = 16
# The cut plays at 24 fps: RIFE triples 16 to 48, then every other frame is dropped.
OUTPUT_FPS = 24
RIFE_MULTIPLIER = 3
RIFE_CHECKPOINT = "rife49.pth"

# Frame sizes, width x height. 480p drafts answer most questions; 720p is the final look.
SIZES = {"480": (480, 832), "720": (720, 1280)}
DEFAULT_SIZE = "480"

TEXT_ENCODER = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
VAE = "wan_2.1_vae.safetensors"

# Wan 2.2 VACE Fun A14B: a high-noise expert for layout and motion, a low-noise expert for detail.
VACE_HIGH = "Wan2.2-VACE-Fun-A14B-high-noise-Q8_0.gguf"
VACE_LOW = "Wan2.2-VACE-Fun-A14B-low-noise-Q8_0.gguf"
VACE_STEPS = 20
VACE_CFG = 3.5
VACE_SHIFT = 8.0
# How hard the control video (the shot's depth) holds the picture. At 1.0 the
# flat depth silhouettes are copied and the face freezes; 0.6 keeps the framing
# and leaves the face alive.
VACE_STRENGTH = 0.6
# A still is the first frame of a short take: a one-frame take ignores the control video.
STILL_FRAMES = 9

# Wan 2.2 S2V 14B: a start picture and a voice track make a talking take.
S2V_MODEL = "wan2.2_s2v_14B_fp8_scaled.safetensors"
AUDIO_ENCODER = "wav2vec2_large_english_fp16.safetensors"
S2V_STEPS = 20
S2V_CFG = 6.0
S2V_SHIFT = 8.0
# S2V's first latent frame decodes badly; the template decodes it twice and drops the first frames.
S2V_DROP_FRAMES = 3

STILL_PREFIX = "reelshort_still"
VACE_PREFIX = "reelshort_vace"
S2V_PREFIX = "reelshort_s2v"
RIFE_PREFIX = "reelshort_rife"


def frames_for(seconds: float) -> int:
    """Wan takes 4n+1 frames: the fewest that cover ``seconds`` at 16 fps."""
    return int(math.ceil((max(seconds, 1.0 / FPS) * FPS - 1) / 4)) * 4 + 1


def _video_out(images: list, prefix: str, audio: list | None = None, frame_rate: int = FPS) -> dict:
    inputs = {
        "images": images, "frame_rate": frame_rate, "loop_count": 0, "filename_prefix": prefix, "format": "video/h264-mp4",
        "pix_fmt": "yuv420p", "crf": 16, "save_metadata": False, "pingpong": False, "save_output": True,
    }
    if audio is not None:
        inputs["audio"] = audio
    return {"class_type": "VHS_VideoCombine", "inputs": inputs}


def vace_graph(
    prompt: str, negative: str, size: str, length: int, seed: int, control: str, reference: str | None, still: bool
) -> dict:
    """Wan VACE over a take of ``length`` frames: ``control`` is the depth video, ``reference`` the people's sheet.

    ``still`` keeps only the take's first frame, saved as an image.
    """
    width, height = SIZES[size]
    split = VACE_STEPS // 2
    graph = {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": VACE_HIGH}},
        "2": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": VACE_LOW}},
        "3": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": VACE_SHIFT}},
        "4": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["2", 0], "shift": VACE_SHIFT}},
        "5": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER, "type": "wan"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["5", 0], "text": prompt}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["5", 0], "text": negative}},
        "8": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "9": {"class_type": "LoadVideo", "inputs": {"file": control}},
        "10": {"class_type": "GetVideoComponents", "inputs": {"video": ["9", 0]}},
        "11": {"class_type": "WanVaceToVideo", "inputs": {
            "positive": ["6", 0], "negative": ["7", 0], "vae": ["8", 0], "width": width, "height": height,
            "length": length, "batch_size": 1, "strength": VACE_STRENGTH, "control_video": ["10", 0]}},
        "12": {"class_type": "KSamplerAdvanced", "inputs": {
            "model": ["3", 0], "add_noise": "enable", "noise_seed": seed, "steps": VACE_STEPS, "cfg": VACE_CFG,
            "sampler_name": "euler", "scheduler": "simple", "positive": ["11", 0], "negative": ["11", 1],
            "latent_image": ["11", 2], "start_at_step": 0, "end_at_step": split, "return_with_leftover_noise": "enable"}},
        "13": {"class_type": "KSamplerAdvanced", "inputs": {
            "model": ["4", 0], "add_noise": "disable", "noise_seed": seed, "steps": VACE_STEPS, "cfg": VACE_CFG,
            "sampler_name": "euler", "scheduler": "simple", "positive": ["11", 0], "negative": ["11", 1],
            "latent_image": ["12", 0], "start_at_step": split, "end_at_step": 10000, "return_with_leftover_noise": "disable"}},
        # The reference picture rides in front of the take as extra latent frames; trim them off.
        "14": {"class_type": "TrimVideoLatent", "inputs": {"samples": ["13", 0], "trim_amount": ["11", 3]}},
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["8", 0]}},
    }
    if reference is not None:
        graph["16"] = {"class_type": "LoadImage", "inputs": {"image": reference}}
        graph["11"]["inputs"]["reference_image"] = ["16", 0]
    if still:
        graph["17"] = {"class_type": "ImageFromBatch", "inputs": {"image": ["15", 0], "batch_index": 0, "length": 1}}
        graph["18"] = {"class_type": "SaveImage", "inputs": {"images": ["17", 0], "filename_prefix": STILL_PREFIX}}
    else:
        graph["18"] = _video_out(["15", 0], VACE_PREFIX)
    return graph


def s2v_graph(prompt: str, negative: str, size: str, length: int, seed: int, start: str, audio: str) -> dict:
    """Wan S2V: the start still talks to the voice track for ``length`` frames."""
    width, height = SIZES[size]
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": S2V_MODEL, "weight_dtype": "default"}},
        "2": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": S2V_SHIFT}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER, "type": "wan"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": prompt}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": negative}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "7": {"class_type": "AudioEncoderLoader", "inputs": {"audio_encoder_name": AUDIO_ENCODER}},
        "8": {"class_type": "LoadAudio", "inputs": {"audio": audio}},
        "9": {"class_type": "AudioEncoderEncode", "inputs": {"audio_encoder": ["7", 0], "audio": ["8", 0]}},
        "10": {"class_type": "LoadImage", "inputs": {"image": start}},
        "11": {"class_type": "WanSoundImageToVideo", "inputs": {
            "positive": ["4", 0], "negative": ["5", 0], "vae": ["6", 0], "width": width, "height": height,
            "length": length, "batch_size": 1, "audio_encoder_output": ["9", 0], "ref_image": ["10", 0]}},
        "12": {"class_type": "KSampler", "inputs": {
            "model": ["2", 0], "seed": seed, "steps": S2V_STEPS, "cfg": S2V_CFG, "sampler_name": "uni_pc",
            "scheduler": "simple", "positive": ["11", 0], "negative": ["11", 1], "latent_image": ["11", 2], "denoise": 1.0}},
        "13": {"class_type": "LatentCut", "inputs": {"samples": ["12", 0], "dim": "t", "index": 0, "amount": 1}},
        "14": {"class_type": "LatentConcat", "inputs": {"samples1": ["13", 0], "samples2": ["12", 0], "dim": "t"}},
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["6", 0]}},
        "16": {"class_type": "ImageFromBatch", "inputs": {"image": ["15", 0], "batch_index": S2V_DROP_FRAMES, "length": 4096}},
        "17": _video_out(["16", 0], S2V_PREFIX, audio=["8", 0]),
    }


def rife_graph(video: str) -> dict:
    """A 16 fps take tripled to 48 fps by RIFE, its sound kept."""
    return {
        "1": {"class_type": "LoadVideo", "inputs": {"file": video}},
        "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
        "3": {"class_type": "RIFE VFI", "inputs": {
            "ckpt_name": RIFE_CHECKPOINT, "frames": ["2", 0], "clear_cache_after_n_frames": 10, "multiplier": RIFE_MULTIPLIER,
            "fast_mode": False, "ensemble": True, "scale_factor": 1.0, "dtype": "float32", "torch_compile": False, "batch_size": 1}},
        "4": _video_out(["3", 0], RIFE_PREFIX, audio=["2", 1], frame_rate=FPS * RIFE_MULTIPLIER),
    }
