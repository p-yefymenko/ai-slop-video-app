"""Speak a list of texts with Chatterbox. Runs in ``content-pipeline/.voice``, not the ComfyUI venv.

    voice_worker.py <jobs.json>

Each job is ``{"text", "voice" (a reference recording or null), "seed", "out"}``;
each ``out`` is written as 16-bit mono WAV at the model's own rate.
"""

from __future__ import annotations

import json
import sys

import soundfile
import torch
from chatterbox.tts import ChatterboxTTS


def main() -> None:
    jobs = json.loads(open(sys.argv[1], encoding="utf-8").read())
    settings = jobs["settings"]
    model = ChatterboxTTS.from_pretrained(device="cuda")
    for job in jobs["speeches"]:
        torch.manual_seed(int(job["seed"]))
        wav = model.generate(
            job["text"],
            audio_prompt_path=job["voice"],
            exaggeration=settings["exaggeration"],
            cfg_weight=settings["cfgWeight"],
        )
        soundfile.write(job["out"], wav.squeeze(0).cpu().numpy(), model.sr, subtype="PCM_16")
        print(f"  {job['out']}: {wav.shape[-1] / model.sr:.2f}s", flush=True)


if __name__ == "__main__":
    main()
