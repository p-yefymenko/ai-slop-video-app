"""Qwen edit encoder with a fourth picture for empty space.

Depth, edges, and backdrop are reference latents. A clothes cutout is shown
to the text encoder only, so its flat colors are not copied as a cartoon.
"""

from __future__ import annotations

import math
from typing_extensions import override

import node_helpers
import comfy.utils
from comfy_api.latest import ComfyExtension, io

_LLAMA_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the key features of the input image (color, shape, size, texture, objects, background), "
    "then explain how the user's text instruction should alter or modify the image. "
    "Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."
    "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)


def _append_picture(image, index: int, *, vae, images_vl: list, ref_latents: list, as_latent: bool) -> str:
    samples = image.movedim(-1, 1)
    total = int(384 * 384)
    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
    width = round(samples.shape[3] * scale_by)
    height = round(samples.shape[2] * scale_by)
    preview = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
    images_vl.append(preview.movedim(1, -1))
    if as_latent and vae is not None:
        total = int(1024 * 1024)
        scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
        width = round(samples.shape[3] * scale_by / 8.0) * 8
        height = round(samples.shape[2] * scale_by / 8.0) * 8
        encoded = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
        ref_latents.append(vae.encode(encoded.movedim(1, -1)[:, :, :, :3]))
    return "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(index)


class TextEncodeQwenBackdrop(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeQwenBackdrop",
            display_name="Text Encode Qwen Image Edit (Backdrop Label)",
            category="model/conditioning/qwen image",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Vae.Input("vae", optional=True),
                io.Image.Input("image1", optional=True),
                io.Image.Input("image2", optional=True),
                io.Image.Input("image3", optional=True),
                io.Image.Input("image4", optional=True),
                io.Int.Input("latent1", default=1, min=0, max=1, optional=True),
                io.Int.Input("latent2", default=1, min=0, max=1, optional=True),
                io.Int.Input("latent3", default=1, min=0, max=1, optional=True),
                io.Int.Input("latent4", default=1, min=0, max=1, optional=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip,
        prompt,
        vae=None,
        image1=None,
        image2=None,
        image3=None,
        image4=None,
        latent1=1,
        latent2=1,
        latent3=1,
        latent4=1,
    ) -> io.NodeOutput:
        ref_latents = []
        images_vl = []
        image_prompt = ""
        slots = (
            (image1, bool(latent1)),
            (image2, bool(latent2)),
            (image3, bool(latent3)),
            (image4, bool(latent4)),
        )
        for index, (image, as_latent) in enumerate(slots, start=1):
            if image is None:
                continue
            image_prompt += _append_picture(
                image,
                index,
                vae=vae,
                images_vl=images_vl,
                ref_latents=ref_latents,
                as_latent=as_latent,
            )
        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=_LLAMA_TEMPLATE)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if ref_latents:
            conditioning = node_helpers.conditioning_set_values(
                conditioning,
                {"reference_latents": ref_latents},
                append=True,
            )
        return io.NodeOutput(conditioning)


class QwenBackdropExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [TextEncodeQwenBackdrop]


async def comfy_entrypoint() -> QwenBackdropExtension:
    return QwenBackdropExtension()
