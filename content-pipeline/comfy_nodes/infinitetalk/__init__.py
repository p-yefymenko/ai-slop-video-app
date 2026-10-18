"""ComfyUI's InfiniteTalk, guided as InfiniteTalk itself guides it, and taking any batch.

Guidance (``InfiniteTalkGuider``). Each step the model guesses three times: with the words and
the voice, with no words but the voice, and with neither. The step is the last guess, pushed
toward the voice by the gap between the second and third (``audio_cfg``) and toward the words by
the gap between the first and second (``text_cfg``): InfiniteTalk's own sampler, at its own
scales. ComfyUI's node gives the voice to every pass, so a plain CFG never pushes toward the
voice at all (the lips followed it loosely); pushing both with one CFG over a pass without
either overshot, and burnt the contrast and color. Which pass hears no voice is passed to the
audio attention in ``transformer_options``; under any other guider every pass hears it, as in
ComfyUI.

Batches. ComfyUI runs a step's passes as one batch whenever the card has room. Its InfiniteTalk
audio attention (``comfy/ldm/wan/model_multitalk.py``) is written for one at a time: the map of
which person each part of the picture is (two-speaker mode) and the audio attention both take the
first dimension for one sample, and with more the run stops ("The size of tensor a (2) must match
the size of tensor b (4)"). Here each sample of a batch goes through the same code alone, which is
exactly what ComfyUI runs when it does not batch.

Delete each part once ComfyUI's own code does the same.
"""

from __future__ import annotations

import torch

import comfy.samplers
from comfy.ldm.wan import model_multitalk as multitalk

_map_alone = multitalk.MultiTalkGetAttnMapPatch.__call__
_audio_alone = multitalk.MultiTalkCrossAttnPatch.__call__
# In ``transformer_options``: which of the step's passes (their place in ``cond_or_uncond``) hear no voice.
SILENT = "infinitetalk_silent_passes"


def _map(self, kwargs):
    """The person map of each sample alone, kept as a list, one per sample."""
    q, k = kwargs["q"], kwargs["k"]
    if self.ref_target_masks is None or q.shape[0] == 1:
        return _map_alone(self, kwargs)
    transformer_options = kwargs.get("transformer_options", {})
    masks = self.ref_target_masks.to(kwargs["x"].device)
    transformer_options["x_ref_attn_map"] = [
        multitalk.get_attn_map_with_target(q[index:index + 1], k[index:index + 1], transformer_options["grid_sizes"], ref_target_masks=masks)
        for index in range(q.shape[0])]
    return kwargs["x"]


def _passes(transformer_options: dict, batch: int) -> list[int]:
    """Which pass each sample of the batch is (``cond_or_uncond``, one entry per equal chunk)."""
    chunks = transformer_options.get("cond_or_uncond") or [0]
    return [chunks[index * len(chunks) // batch] for index in range(batch)]


def _audio(self, kwargs):
    """The audio attention of each sample alone, with its own person map, joined back into the
    batch; a sample of a pass in ``SILENT`` hears nothing."""
    x = kwargs["x"]
    transformer_options = kwargs.get("transformer_options", {})
    if transformer_options.get("block_index") is None:
        return _audio_alone(self, kwargs)
    passes = _passes(transformer_options, x.shape[0])
    silent = transformer_options.get(SILENT, ())
    maps = transformer_options.pop("x_ref_attn_map", None)
    out = []
    for index in range(x.shape[0]):
        if passes[index] in silent:
            out.append(x[index:index + 1])
            continue
        options = dict(transformer_options)
        if maps is not None:
            options["x_ref_attn_map"] = maps[index] if isinstance(maps, list) else maps
        out.append(_audio_alone(self, {**kwargs, "x": x[index:index + 1], "transformer_options": options}))
    return torch.cat(out, dim=0)


multitalk.MultiTalkGetAttnMapPatch.__call__ = _map
multitalk.MultiTalkCrossAttnPatch.__call__ = _audio


class GuiderInfiniteTalk(comfy.samplers.CFGGuider):
    """The step's three guesses: the negative without the voice, the negative with it, and the
    positive (with it): silent + audio_cfg * (heard - silent) + text_cfg * (positive - heard)."""

    def set_cfg(self, text_cfg: float, audio_cfg: float) -> None:
        self.text_cfg, self.audio_cfg = text_cfg, audio_cfg

    def set_conds(self, positive, negative) -> None:
        self.inner_set_conds({"positive": positive, "negative": negative})

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        options = {**model_options, "transformer_options": {**model_options.get("transformer_options", {}), SILENT: (0,)}}
        negative, positive = self.conds.get("negative"), self.conds.get("positive")
        silent, heard, said = comfy.samplers.calc_cond_batch(self.inner_model, [negative, negative, positive], x, timestep, options)
        return silent + self.audio_cfg * (heard - silent) + self.text_cfg * (said - heard)


class InfiniteTalkGuider:
    """Guides an InfiniteTalk take as InfiniteTalk does: toward the words and toward the voice, each at its own scale."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",), "positive": ("CONDITIONING",), "negative": ("CONDITIONING",),
            "text_cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            "audio_cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1}),
        }}

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(self, model, positive, negative, text_cfg, audio_cfg):
        guider = GuiderInfiniteTalk(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(text_cfg, audio_cfg)
        return (guider,)


NODE_CLASS_MAPPINGS = {"InfiniteTalkGuider": InfiniteTalkGuider}
NODE_DISPLAY_NAME_MAPPINGS = {"InfiniteTalkGuider": "InfiniteTalk Guider (words and voice)"}
