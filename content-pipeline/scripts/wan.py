"""ComfyUI graphs for the clips, and the settings every scene shares.

    start still (drawn by an image model from the shot's 3D layout, outside ComfyUI)
      -> clip: an on-screen speaker's face -> InfiniteTalk from the still and the voice, the voice
               tied to the speaker and silence to everyone else (``talk_graph``);
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
# The cut plays at 24 fps: RIFE fills a take out (``rife_multiplier``), then 24 fps is taken from it.
OUTPUT_FPS = 24
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
# How hard the control video (the shot's depth) holds a clip. At 1.0 the flat
# depth silhouettes are copied and the face freezes; 0.6 keeps the framing and
# leaves the face alive.
VACE_STRENGTH = 0.6

# Wan 2.2 S2V 14B: a start picture and a voice track make a talking take.
S2V_MODEL = "wan2.2_s2v_14B_fp8_scaled.safetensors"
AUDIO_ENCODER = "wav2vec2_large_english_fp16.safetensors"
S2V_STEPS = 20
S2V_CFG = 6.0
S2V_SHIFT = 8.0
# S2V's first latent frame decodes badly; the template decodes it twice and drops the first frames.
S2V_DROP_FRAMES = 3

# InfiniteTalk (Wan 2.1 I2V 14B with its audio patch): a start picture and voices, each voice tied
# to one person's mask, so only the speaker's mouth moves; S2V takes one voice and no mask, and
# moved every face in the frame. As ComfyUI's InfiniteTalk template, at full steps and real CFG in
# place of its speed LoRA, guided as InfiniteTalk's own sampler guides it, at its own scales and 480p shift.
TALK_MODEL = "wan2.1_i2v_480p_14B_fp8_scaled.safetensors"
TALK_PATCH = "wan2.1_infiniteTalk_multi_fp16.safetensors"
TALK_AUDIO_ENCODER = "wav2vec2-chinese-base_fp16.safetensors"
# InfiniteTalk reads a voice at 25 video frames a second.
TALK_FPS = 25
TALK_STEPS = 20
# Each step is pushed toward the words and toward the voice, each by its own gap and scale
# (``InfiniteTalkGuider``, the pipeline's ComfyUI node ``comfy_nodes/infinitetalk``): ComfyUI's
# plain CFG gave the voice to both passes and never pushed toward it (loose lips), and one CFG of
# 5 over a pass without words or voice overshot (burnt contrast and color).
TALK_TEXT_CFG = 5.0
TALK_AUDIO_CFG = 4.0
TALK_SHIFT = 7.0
# Wan 2.1 makes at most 81 frames at once (what it was trained on); a longer take is made in windows
# of 81, each going on from the last 9 frames before it, as InfiniteTalk and its template do.
TALK_WINDOW = 81
TALK_MOTION_FRAMES = 9
# Everyone but the speaker gets a silent track, which the node lays after the voice: it only has
# to be long enough for the audio encoder to read, and it is never heard.
TALK_SILENCE_SECONDS = 1.0
TALK_SILENCE_RATE = 16000

VACE_PREFIX = "reelshort_vace"
S2V_PREFIX = "reelshort_s2v"
TALK_PREFIX = "reelshort_talk"
RIFE_PREFIX = "reelshort_rife"


def frames_for(seconds: float, fps: int = FPS) -> int:
    """Wan takes 4n+1 frames: the fewest that cover ``seconds`` at ``fps``."""
    return int(math.ceil((max(seconds, 1.0 / fps) * fps - 1) / 4)) * 4 + 1


def _video_out(images: list, prefix: str, audio: list | None = None, frame_rate: int = FPS) -> dict:
    inputs = {
        "images": images, "frame_rate": frame_rate, "loop_count": 0, "filename_prefix": prefix, "format": "video/h264-mp4",
        "pix_fmt": "yuv420p", "crf": 16, "save_metadata": False, "pingpong": False, "save_output": True,
    }
    if audio is not None:
        inputs["audio"] = audio
    return {"class_type": "VHS_VideoCombine", "inputs": inputs}


def vace_graph(prompt: str, negative: str, size: str, length: int, seed: int, control: str, reference: str | None) -> dict:
    """Wan VACE over a take of ``length`` frames: ``control`` is the depth video, ``reference`` the people's sheet."""
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


def talk_windows(length: int) -> list[int]:
    """The windows a talking take of ``length`` frames is made in: one of ``length`` up to
    ``TALK_WINDOW``; past it, windows of up to ``TALK_WINDOW``, each after the first going on from
    the last ``TALK_MOTION_FRAMES`` of the take so far (so it adds that many fewer), the last one
    as short as covers the rest (``4n+1`` frames, as Wan takes)."""
    windows = [min(length, TALK_WINDOW)]
    made = windows[0]
    while made < length:
        frames = min(TALK_WINDOW, math.ceil((length - made + TALK_MOTION_FRAMES - 1) / 4) * 4 + 1)
        windows.append(frames)
        made += frames - TALK_MOTION_FRAMES
    return windows


def talk_graph(prompt: str, negative: str, size: str, length: int, seed: int, start: str, voice: str,
               others: tuple[str, str, str] | None) -> dict:
    """InfiniteTalk: the start still talks to the ``voice`` track for ``length`` frames at
    ``TALK_FPS``, in ``talk_windows`` (the take may run a few frames past ``length``).

    ``others``, when anyone else is in the frame, is (the speaker's mask, everyone else's mask,
    a silent track): the voice is tied to the speaker's mask and the silence to everyone else's.
    The node lays the second track after the first, so through the voice everyone else hears
    nothing, which MultiTalk was trained to read as a listener (white in a mask is the person)."""
    width, height = SIZES[size]
    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": TALK_MODEL, "weight_dtype": "default"}},
        "2": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": TALK_SHIFT}},
        "3": {"class_type": "ModelPatchLoader", "inputs": {"name": TALK_PATCH}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER, "type": "wan"}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 0], "text": prompt}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 0], "text": negative}},
        "7": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "8": {"class_type": "AudioEncoderLoader", "inputs": {"audio_encoder_name": TALK_AUDIO_ENCODER}},
        "9": {"class_type": "LoadAudio", "inputs": {"audio": voice}},
        "10": {"class_type": "AudioEncoderEncode", "inputs": {"audio_encoder": ["8", 0], "audio": ["9", 0]}},
        "11": {"class_type": "LoadImage", "inputs": {"image": start}},
        "22": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "23": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
    }
    speakers = {"mode": "single_speaker"}
    if others is not None:
        speaker_mask, other_mask, silence = others
        graph["16"] = {"class_type": "LoadImage", "inputs": {"image": speaker_mask}}
        graph["17"] = {"class_type": "ImageToMask", "inputs": {"image": ["16", 0], "channel": "red"}}
        graph["18"] = {"class_type": "LoadImage", "inputs": {"image": other_mask}}
        graph["19"] = {"class_type": "ImageToMask", "inputs": {"image": ["18", 0], "channel": "red"}}
        graph["20"] = {"class_type": "LoadAudio", "inputs": {"audio": silence}}
        graph["21"] = {"class_type": "AudioEncoderEncode", "inputs": {"audio_encoder": ["8", 0], "audio": ["20", 0]}}
        speakers = {"mode": "two_speakers", "mode.audio_encoder_output_2": ["21", 0], "mode.mask_1": ["17", 0], "mode.mask_2": ["19", 0]}
    taken = None
    for index, frames in enumerate(talk_windows(length)):
        talk, sample, decode = f"w{index}_talk", f"w{index}_sample", f"w{index}_decode"
        graph[talk] = {"class_type": "WanInfiniteTalkToVideo", "inputs": {
            **speakers, "model": ["2", 0], "model_patch": ["3", 0], "positive": ["5", 0], "negative": ["6", 0],
            "vae": ["7", 0], "width": width, "height": height, "length": frames, "start_image": ["11", 0],
            "audio_encoder_output_1": ["10", 0], "motion_frame_count": TALK_MOTION_FRAMES, "audio_scale": 1.0}}
        graph[f"w{index}_guide"] = {"class_type": "InfiniteTalkGuider", "inputs": {
            "model": [talk, 0], "positive": [talk, 1], "negative": [talk, 2], "text_cfg": TALK_TEXT_CFG, "audio_cfg": TALK_AUDIO_CFG}}
        graph[f"w{index}_steps"] = {"class_type": "BasicScheduler", "inputs": {"model": [talk, 0], "scheduler": "normal", "steps": TALK_STEPS, "denoise": 1.0}}
        graph[sample] = {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["22", 0], "guider": [f"w{index}_guide", 0], "sampler": ["23", 0], "sigmas": [f"w{index}_steps", 0], "latent_image": [talk, 3]}}
        graph[decode] = {"class_type": "VAEDecode", "inputs": {"samples": [sample, 0], "vae": ["7", 0]}}
        if taken is None:
            taken = [decode, 0]
            continue
        # A later window starts with the take's last frames again (its ``trim_image``); they are cut off before it is added.
        graph[talk]["inputs"]["previous_frames"] = taken
        graph[f"w{index}_new"] = {"class_type": "ImageFromBatch", "inputs": {"image": [decode, 0], "batch_index": [talk, 4], "length": 4096}}
        graph[f"w{index}_take"] = {"class_type": "ImageBatch", "inputs": {"image1": taken, "image2": [f"w{index}_new", 0]}}
        taken = [f"w{index}_take", 0]
    graph["15"] = _video_out(taken, TALK_PREFIX, audio=["9", 0], frame_rate=TALK_FPS)
    return graph


def rife_multiplier(fps: int) -> int:
    """How many times RIFE fills a take of ``fps``: to at least twice ``OUTPUT_FPS``, so each frame
    of the cut is a filled frame or within a quarter of a cut frame of one (16 -> 48, 25 -> 50)."""
    return math.ceil(2 * OUTPUT_FPS / fps)


def rife_graph(video: str, fps: int = FPS) -> dict:
    """A take of ``fps`` filled out by RIFE (``rife_multiplier``), its sound kept."""
    multiplier = rife_multiplier(fps)
    return {
        "1": {"class_type": "LoadVideo", "inputs": {"file": video}},
        "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
        "3": {"class_type": "RIFE VFI", "inputs": {
            "ckpt_name": RIFE_CHECKPOINT, "frames": ["2", 0], "clear_cache_after_n_frames": 10, "multiplier": multiplier,
            "fast_mode": False, "ensemble": True, "scale_factor": 1.0, "dtype": "float32", "torch_compile": False, "batch_size": 1}},
        "4": _video_out(["3", 0], RIFE_PREFIX, audio=["2", 1], frame_rate=fps * multiplier),
    }
