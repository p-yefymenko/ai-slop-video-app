"""Treat Gemma API embeddings as already projected for LTX AV GGUF models."""


class LTXAVUseProcessedAPIEmbeds:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "reelshort"
    TITLE = "LTX AV: use processed Gemma API embeds"

    def patch(self, model):
        patched = model.clone()
        diffusion = patched.model.diffusion_model
        if not hasattr(diffusion, "caption_proj_before_connector"):
            raise RuntimeError("This node only works with an LTX AV diffusion model")
        diffusion.caption_proj_before_connector = True
        return (patched,)


NODE_CLASS_MAPPINGS = {
    "LTXAVUseProcessedAPIEmbeds": LTXAVUseProcessedAPIEmbeds,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXAVUseProcessedAPIEmbeds": "LTX AV: use processed Gemma API embeds",
}
