"""Subtitles and name cards, burned onto the finished episode.

The script says what is said and by whom; the clips decide when. Each speaking
clip's audio is transcribed with word times (Whisper), and the script's lines
are laid over the words it heard, in order and in proportion to their length,
so a subtitle appears while its words are spoken. A character's name card
appears the first time they speak on screen in the episode.
"""

from __future__ import annotations

import json
import subprocess
from functools import cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WHISPER = "openai/whisper-large-v3-turbo"
FONTS = Path("C:/Windows/Fonts")
SUBTITLE_FONT, CARD_FONT = "segoeuib.ttf", "georgiaz.ttf"
# Where the text sits, as shares of the frame height; how long a subtitle stays after its last word.
SUBTITLE_Y, CARD_Y, SUBTITLE_HOLD = 0.70, 0.62, 0.35
# Text height as a share of the frame width.
SUBTITLE_SIZE, CARD_SIZE = 0.05, 0.06
# A name card shows this long, starting this long after the speaker's first word.
CARD_SECONDS, CARD_DELAY = 2.5, 0.3


@cache
def _transcriber():
    import torch
    from transformers import pipeline

    cuda = torch.cuda.is_available()
    return pipeline(
        "automatic-speech-recognition", model=WHISPER, device=0 if cuda else -1, dtype=torch.float16 if cuda else torch.float32
    )


def spoken_words(clip: Path) -> list[tuple[float, float]]:
    """When each word heard in the clip is spoken: (start, end) in seconds."""
    audio = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(clip), "-ac", "1", "-ar", "16000", "-f", "f32le", "-"], capture_output=True, check=True
    ).stdout
    import numpy as np

    samples = np.frombuffer(audio, dtype=np.float32)
    if not len(samples):
        return []
    result = _transcriber()({"raw": samples, "sampling_rate": 16000}, return_timestamps="word", generate_kwargs={"language": "en"})
    words = []
    for chunk in result.get("chunks") or []:
        start, end = chunk["timestamp"]
        if start is not None:
            words.append((float(start), float(end if end is not None else start + 0.3)))
    return words


def line_times(lines: list[dict], words: list[tuple[float, float]], duration: float) -> list[tuple[float, float]]:
    """When each script line is on screen, from the words heard.

    The heard words are shared out over the lines in order, in proportion to
    each line's word count. A line's ``startSeconds`` is the earliest it shows.
    With nothing heard, the lines share the scene from their own start times.
    """
    counts = [max(1, len(line["line"].split())) for line in lines]
    times = []
    if words:
        total, taken = sum(counts), 0
        for index, count in enumerate(counts):
            first = round(taken * len(words) / total)
            taken += count
            last = max(first, round(taken * len(words) / total) - 1)
            start, end = words[min(first, len(words) - 1)][0], words[min(last, len(words) - 1)][1]
            times.append((start, end + SUBTITLE_HOLD))
    else:
        starts = [line.get("startSeconds") for line in lines]
        for index, line in enumerate(lines):
            start = starts[index] if starts[index] is not None else (times[-1][1] if times else 0.0)
            times.append((start, start + duration / len(lines)))
    fixed = []
    for line, (start, end) in zip(lines, times):
        start = max(start, float(line.get("startSeconds") or 0.0))
        fixed.append((min(start, duration), min(max(end, start + 0.6), duration)))
    # A subtitle leaves when the next one comes.
    return [(start, min(end, fixed[i + 1][0]) if i + 1 < len(fixed) else end) for i, (start, end) in enumerate(fixed)]


def _text_image(text: str, font_name: str, size: int, fill: tuple, dest: Path, width: int) -> Path:
    """The text, centered and wrapped, with a dark outline, on a transparent strip as wide as the video."""
    try:
        font = ImageFont.truetype(str(FONTS / font_name), size)
    except OSError:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    words, rows, row = text.split(), [], ""
    for word in words:
        trial = f"{row} {word}".strip()
        if draw.textlength(trial, font=font) > width * 0.8 and row:
            rows.append(row)
            row = word
        else:
            row = trial
    rows.append(row)
    height = int(len(rows) * size * 1.3 + size * 0.6)
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    for index, row in enumerate(rows):
        x = (width - draw.textlength(row, font=font)) / 2
        draw.text((x, size * 0.3 + index * size * 1.3), row, font=font, fill=fill, stroke_width=max(2, size // 12), stroke_fill=(0, 0, 0, 220))
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest)
    return dest


def overlays(show: dict, episode: dict, clips: list[Path], work: Path, width: int) -> list[tuple[Path, float, float, float]]:
    """Every subtitle and name card as (picture, from, to, height share), on the episode clock, for a video ``width`` wide."""
    found, offset, named = [], 0.0, set()
    for scene, clip in zip(episode["scenes"], clips):
        duration = float(
            subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(clip)], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
        lines = scene.get("lines") or []
        if lines:
            times = line_times(lines, spoken_words(clip), duration)
            for index, (line, (start, end)) in enumerate(zip(lines, times)):
                picture = _text_image(line["line"], SUBTITLE_FONT, round(width * SUBTITLE_SIZE), (255, 255, 255, 255), work / f"s{scene['sceneNumber']:02d}_{index}.png", width)
                found.append((picture, offset + start, offset + end, SUBTITLE_Y))
                speaker = line["speakerId"]
                name = show["characters"][speaker].get("name")
                if name and speaker in (scene.get("performances") or {}) and speaker not in named:
                    named.add(speaker)
                    card = _text_image(name, CARD_FONT, round(width * CARD_SIZE), (232, 196, 120, 255), work / f"card_{speaker}.png", width)
                    found.append((card, offset + start + CARD_DELAY, min(offset + duration, offset + start + CARD_DELAY + CARD_SECONDS), CARD_Y))
        offset += duration
    return found


def caption_episode(show: dict, episode: dict, clips: list[Path], video: Path) -> Path:
    """Burn subtitles and name cards onto ``video``; the plain cut is kept as ``<name>_plain.mp4``."""
    work = video.parent / "inputs" / "captions"
    width = int(
        subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width", "-of", "csv=p=0", str(video)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    )
    found = overlays(show, episode, clips, work, width)
    plain = video.with_name(video.stem + "_plain.mp4")
    video.replace(plain)
    if not found:
        plain.replace(video)
        return video
    inputs, chain, last = ["-i", str(plain)], [], "0:v"
    for index, (picture, start, end, share) in enumerate(found, start=1):
        inputs += ["-i", str(picture)]
        out = f"v{index}"
        chain.append(
            f"[{last}][{index}:v]overlay=x=(W-w)/2:y=H*{share}-h/2:enable='between(t,{start:.2f},{end:.2f})'[{out}]"
        )
        last = out
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(chain), "-map", f"[{last}]", "-map", "0:a?",
         "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "copy", str(video)],
        check=True,
    )
    (work / "captions.json").write_text(
        json.dumps([{"picture": p.name, "from": round(a, 2), "to": round(b, 2)} for p, a, b, _s in found], indent=2) + "\n", encoding="utf-8"
    )
    return video
