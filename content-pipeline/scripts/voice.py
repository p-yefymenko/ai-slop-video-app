"""A scene's voice track: every line spoken by Chatterbox, placed where the script says, as long as the scene.

One speaker's lines in a row are one speech, spoken in one go so the voice runs
on naturally, starting at the first line's ``startSeconds``. The track is the
scene's length exactly: S2V paces the take by it, and a silent scene gets a
silent track so every clip has sound to join.

A character's voice is made once, in ``content:plates``, from their ``voice``
description: Qwen3-TTS VoiceDesign speaks ``voiceSample`` (prompts.json) in it,
and that recording is ``voice.wav`` beside their reference (``world.voice_path``).
Chatterbox speaks every line of theirs in it. A speaker without one stops the run.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

from comfy import inputs_dir, present, stable_seed, write_log
from world import settings, voice_path

ROOT = Path(__file__).resolve().parents[1]
VOICE_PYTHON = ROOT / ".voice" / "Scripts" / "python.exe"
WORKER = Path(__file__).resolve().with_name("voice_worker.py")
DESIGN_PYTHON = ROOT / ".voicedesign" / "Scripts" / "python.exe"
DESIGN_WORKER = Path(__file__).resolve().with_name("voice_design_worker.py")
# Makes a voice from words; Chatterbox cannot.
DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
# Every script is written in English.
LANGUAGE = "English"
# Chatterbox's own output rate.
SAMPLE_RATE = 24_000
# How much the delivery is acted (0.5 is neutral) and how closely the pace follows the reference voice.
EXAGGERATION = 0.6
CFG_WEIGHT = 0.4


def design_entry(character: dict) -> dict:
    """Everything a character's ``voice.wav`` is made from."""
    return {"model": DESIGN_MODEL, "language": LANGUAGE, "instruct": character["voice"], "text": settings()["voiceSample"]}


def voice_is_current(character: dict, dest: Path) -> bool:
    """Whether ``voice.wav`` exists and was made from the character's current description."""
    log = inputs_dir(dest) / "log.json"
    if not (present(dest) and log.is_file()):
        return False
    made = json.loads(log.read_text(encoding="utf-8"))
    return all(made.get(key) == value for key, value in design_entry(character).items())


def design_voices(show: dict, character_ids: list[str]) -> None:
    """Make each character's ``voice.wav`` from their ``voice`` description, at a seed fixed by it."""
    if not DESIGN_PYTHON.is_file():
        raise SystemExit(f"Missing {DESIGN_PYTHON}. Run `pnpm run content:voice-deps` once.")
    jobs = []
    for cid in character_ids:
        entry = design_entry(show["characters"][cid])
        dest = voice_path(show, cid)
        dest.parent.mkdir(parents=True, exist_ok=True)
        jobs.append({"id": cid, "entry": entry, "dest": dest, "seed": stable_seed("voice", entry["instruct"])})
    with tempfile.TemporaryDirectory() as folder:
        job_file = Path(folder) / "jobs.json"
        job_file.write_text(json.dumps({
            "settings": {"model": DESIGN_MODEL, "language": LANGUAGE},
            "voices": [{"text": job["entry"]["text"], "instruct": job["entry"]["instruct"], "seed": job["seed"], "out": str(job["dest"])} for job in jobs],
        }), encoding="utf-8")
        subprocess.run([str(DESIGN_PYTHON), str(DESIGN_WORKER), str(job_file)], check=True)
    for job in jobs:
        write_log(job["dest"], {**job["entry"], "seed": job["seed"]})


def speeches(scene: dict) -> list[dict]:
    """The scene's lines grouped into speeches: ``{"speakerId", "text", "start"}`` in order."""
    grouped: list[dict] = []
    for line in scene.get("lines") or []:
        words = " ".join(line["line"].split())
        if grouped and grouped[-1]["speakerId"] == line["speakerId"]:
            grouped[-1]["text"] += " " + words
            continue
        grouped.append({"speakerId": line["speakerId"], "text": words, "start": float(line.get("startSeconds") or 0.0)})
    return grouped


def lay_out(clips: list[tuple[float, np.ndarray]], seconds: float, rate: int = SAMPLE_RATE) -> tuple[np.ndarray, list[str]]:
    """Each ``(start, samples)`` placed on a silent track ``seconds`` long. Returns the track and what did not fit."""
    track = np.zeros(int(round(seconds * rate)), dtype=np.float32)
    problems = []
    ends = [start + len(samples) / rate for start, samples in clips]
    for index, (start, samples) in enumerate(clips):
        if index + 1 < len(clips) and ends[index] > clips[index + 1][0] + 1e-3:
            problems.append(f"speech {index + 1} runs {ends[index] - clips[index + 1][0]:.2f}s into the next one")
        if ends[index] > seconds + 1e-3:
            problems.append(f"speech {index + 1} ends at {ends[index]:.2f}s, after the scene's {seconds:.2f}s; its end is cut")
        at = int(round(start * rate))
        piece = samples[: max(0, len(track) - at)]
        track[at:at + len(piece)] += piece
    return np.clip(track, -1.0, 1.0), problems


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2 or handle.getnchannels() != 1 or handle.getframerate() != SAMPLE_RATE:
            raise RuntimeError(f"{path} is not 16-bit mono at {SAMPLE_RATE} Hz")
        return np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def write_wav(path: Path, samples: np.ndarray, rate: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(np.rint(np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes())


def write_voice_track(show: dict, episode: dict, scene: dict, dest: Path) -> list[str]:
    """Speak the scene's lines and lay them out on a track as long as the scene. Returns what did not fit."""
    start, finish = (float(value) for value in scene["timeRangeSeconds"])
    said = speeches(scene)
    problems: list[str] = []
    clips: list[tuple[float, np.ndarray]] = []
    if said:
        if not VOICE_PYTHON.is_file():
            raise SystemExit(f"Missing {VOICE_PYTHON}. Run `pnpm run content:voice-deps` once.")
        with tempfile.TemporaryDirectory() as folder:
            jobs = []
            for index, speech in enumerate(said):
                voice = voice_path(show, speech["speakerId"])
                if not voice.is_file():
                    raise SystemExit(f"{speech['speakerId']} has no voice ({voice}). Run `pnpm run content:plates`.")
                jobs.append({
                    "text": speech["text"],
                    "voice": str(voice),
                    "seed": stable_seed(show["id"], episode["episodeNumber"], scene["sceneNumber"], "voice", index),
                    "out": str(Path(folder) / f"speech_{index}.wav"),
                })
            job_file = Path(folder) / "jobs.json"
            job_file.write_text(json.dumps({"settings": {"exaggeration": EXAGGERATION, "cfgWeight": CFG_WEIGHT}, "speeches": jobs}), encoding="utf-8")
            subprocess.run([str(VOICE_PYTHON), str(WORKER), str(job_file)], check=True)
            clips = [(speech["start"], read_wav(Path(job["out"]))) for speech, job in zip(said, jobs)]
    track, fit = lay_out(clips, finish - start)
    write_wav(dest, track)
    return problems + fit
