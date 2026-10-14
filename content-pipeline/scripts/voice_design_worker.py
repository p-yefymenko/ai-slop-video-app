"""Make voices from descriptions with Qwen3-TTS VoiceDesign. Runs in ``content-pipeline/.voicedesign``, not the ComfyUI venv.

    voice_design_worker.py <jobs.json>

Each job is ``{"text", "instruct" (the voice's description), "seed", "out"}``;
each ``out`` is written as 16-bit mono WAV at the model's own rate.
"""

from __future__ import annotations

import json
import sys

import soundfile
import torch
from qwen_tts import Qwen3TTSModel


def main() -> None:
    jobs = json.loads(open(sys.argv[1], encoding="utf-8").read())
    settings = jobs["settings"]
    model = Qwen3TTSModel.from_pretrained(settings["model"], device_map="cuda:0", dtype=torch.bfloat16)
    for job in jobs["voices"]:
        torch.manual_seed(int(job["seed"]))
        wavs, rate = model.generate_voice_design(text=job["text"], instruct=job["instruct"], language=settings["language"])
        soundfile.write(job["out"], wavs[0], rate, subtype="PCM_16")
        print(f"  {job['out']}: {len(wavs[0]) / rate:.2f}s", flush=True)


if __name__ == "__main__":
    main()
