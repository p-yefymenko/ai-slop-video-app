#!/usr/bin/env python3
"""Controlled sound-prompt A/B on one scene across fixed seeds.

Variants (prompt/guider only; same seeds):
  V1 — previous run-on composition (reuse existing clips + logged prompts)
  V2 — restructured sentences, music phrase kept
  V3 — restructured sentences, music phrase omitted
  V4 — restructured sentences, music phrase kept, MultimodalGuider forced
"""

from __future__ import annotations

import _scripts_path  # noqa: F401

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from clip_spec import CLIP_HEIGHT, CLIP_WIDTH
from ffmpeg_tools import find_ffmpeg
from generate_batch import (
    compile_ltx_prompt,
    execute_queued_graph,
    free_comfy_models,
    inject_dialogue_multimodal_guider,
    inject_prompt,
    inject_scene_length,
    inject_seed,
    inject_start_frame,
    load_dotenv,
    load_json,
    load_show,
    stage_start_still,
    write_clip_generation_log,
)
from pipeline_paths import ROOT, discover_show_scripts, start_still_path

SEEDS = (1001, 2002, 3003)
SAMPLE_RATE = 22050
V1_PROMPT = (
    "Continue directly from this image as the exact first frame. Preserve its people, "
    "wardrobe, props, set, composition, and lighting. Use one continuous take. Animate "
    "only the motion, performance, camera, dialogue, and sound described here: the "
    "location holds deep roar of white-gold fire in the well, steady wind across open "
    "basalt, distant storm rumble in wide open air with no walls, little echo, the fire "
    "surges upward with a deep furnace boom and rising crackle, and no music plays"
)


def out_dir(show_id: str, episode: int, scene: int) -> Path:
    return (
        ROOT
        / "output"
        / "experiments"
        / "sound_ab"
        / show_id
        / str(episode)
        / f"scene_{scene:02d}"
    )


def clip_path(dest_root: Path, variant: str, seed: int) -> Path:
    return dest_root / f"{variant}_seed_{seed}.mp4"


def log_path(dest_root: Path, variant: str, seed: int) -> Path:
    return dest_root / f"{variant}_seed_{seed}.log.json"


def load_audio_mono(path: Path) -> np.ndarray:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise SystemExit("ffmpeg missing")
    raw = subprocess.check_output(
        [
            ffmpeg,
            "-i",
            str(path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "f32le",
            "-",
        ],
        stderr=subprocess.DEVNULL,
    )
    return np.frombuffer(raw, dtype=np.float32)


def integrated_lufs(path: Path) -> float | None:
    ffmpeg = find_ffmpeg()
    result = subprocess.run(
        [ffmpeg, "-i", str(path), "-af", "ebur128=peak=true", "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=False,
    )
    value = None
    for line in (result.stderr or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("I:"):
            # Final summary line looks like: I:         -8.6 LUFS
            parts = stripped.replace("LUFS", "").split()
            if len(parts) >= 2:
                try:
                    value = float(parts[1])
                except ValueError:
                    continue
    return value


def spectral_metrics(samples: np.ndarray) -> dict:
    if samples.size < SAMPLE_RATE // 4:
        return {
            "fundamental_hz": None,
            "harmonic_ratios_2f_5f": None,
            "spectral_flatness": None,
            "envelope_modulation": None,
        }
    win = 4096
    hops = max(1, (samples.size - win) // (win // 2))
    window = np.hanning(win)
    acc = np.zeros(win // 2 + 1, dtype=np.float64)
    for i in range(hops):
        start = i * (win // 2)
        chunk = samples[start : start + win]
        if chunk.size < win:
            break
        spectrum = np.abs(np.fft.rfft(chunk * window)) ** 2
        acc += spectrum
    acc /= max(hops, 1)
    freqs = np.fft.rfftfreq(win, d=1 / SAMPLE_RATE)

    # Spectral flatness on power spectrum (avoid DC).
    power = np.maximum(acc[1:], 1e-20)
    geo = np.exp(np.mean(np.log(power)))
    arith = np.mean(power)
    flatness = float(geo / arith) if arith > 0 else None

    # Fundamental: strongest peak in 40–250 Hz (music / bed region).
    band = (freqs >= 40) & (freqs <= 250)
    if not np.any(band):
        fundamental = None
        ratios = None
    else:
        local = acc.copy()
        local[~band] = 0
        peak_idx = int(np.argmax(local))
        fundamental = float(freqs[peak_idx])
        fund_amp = float(acc[peak_idx]) + 1e-20
        ratios = []
        for harm in (2, 3, 4, 5):
            target = fundamental * harm
            if target >= freqs[-1]:
                ratios.append(None)
                continue
            # Nearest bin within +/- 3 Hz
            nearest = int(np.argmin(np.abs(freqs - target)))
            if abs(freqs[nearest] - target) > 3.0:
                ratios.append(None)
            else:
                ratios.append(float(acc[nearest] / fund_amp))

    # Envelope modulation: coefficient of variation of frame RMS.
    frame = SAMPLE_RATE // 40  # 25 ms
    n_frames = samples.size // frame
    if n_frames < 4:
        modulation = None
    else:
        rms = np.array(
            [
                float(np.sqrt(np.mean(samples[i * frame : (i + 1) * frame] ** 2) + 1e-12))
                for i in range(n_frames)
            ]
        )
        mean_rms = float(np.mean(rms))
        modulation = float(np.std(rms) / mean_rms) if mean_rms > 1e-8 else None

    return {
        "fundamental_hz": None if fundamental is None else round(fundamental, 2),
        "harmonic_ratios_2f_5f": None
        if ratios is None
        else [None if r is None else round(r, 4) for r in ratios],
        "spectral_flatness": None if flatness is None else round(flatness, 6),
        "envelope_modulation": None if modulation is None else round(modulation, 4),
    }


def analyze_clip(path: Path) -> dict:
    samples = load_audio_mono(path)
    metrics = spectral_metrics(samples)
    metrics["lufs"] = integrated_lufs(path)
    metrics["path"] = str(path)
    return metrics


def reuse_v1(dest_root: Path, generate_dir: Path) -> None:
    for seed in SEEDS:
        src = generate_dir / f"scene_02_sound_seed_{seed}.mp4"
        src_log = generate_dir / f"scene_02_sound_seed_{seed}.log.json"
        if not src.is_file():
            raise SystemExit(f"Missing V1 control clip {src}")
        dest = clip_path(dest_root, "V1", seed)
        shutil.copy2(src, dest)
        payload = {
            "variant": "V1",
            "seed": seed,
            "prompt": V1_PROMPT,
            "graph": "ltx_gemma_api.json",
            "multimodal_guider": False,
            "include_music": True,
            "note": "Control: prior run-on sound composition; clip reused from previous regen.",
        }
        if src_log.is_file():
            try:
                old = json.loads(src_log.read_text(encoding="utf-8"))
                passes = old.get("passes") or []
                if passes and isinstance(passes[0], dict) and passes[0].get("prompt"):
                    payload["prompt"] = passes[0]["prompt"]
            except (OSError, json.JSONDecodeError):
                pass
        log_path(dest_root, "V1", seed).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )


def render_variant(
    *,
    show: dict,
    episode: dict,
    scene: dict,
    location: dict,
    still: Path,
    dest_root: Path,
    variant: str,
    seed: int,
    include_music: bool,
    force_multimodal: bool,
) -> Path:
    dest = clip_path(dest_root, variant, seed)
    if dest.is_file() and dest.stat().st_size > 1024:
        print(f"  skip existing {dest.name}", flush=True)
        return dest

    from depth_control import choose_ltx_workflow

    prompt = compile_ltx_prompt(show, scene, location, include_music=include_music)
    wf_path, graph_name = choose_ltx_workflow(scene, episode, spatial=False)
    workflow = load_json(wf_path)
    api_key = os.environ.get("LTXV_API_KEY", "")
    graph = inject_prompt(workflow, prompt, api_key)
    inject_scene_length(graph, duration_seconds=scene["durationSeconds"])
    inject_seed(graph, seed)
    if force_multimodal or scene.get("speakerId"):
        inject_dialogue_multimodal_guider(graph)
    inject_start_frame(graph, stage_start_still(still, fit_clip=True), strength=0.7)
    graph["10"]["inputs"]["filename_prefix"] = (
        f"sound_ab_{variant}_s{seed}_{scene['sceneNumber']:02d}"
    )

    write_clip_generation_log(
        dest,
        prompt=prompt,
        graph_name=graph_name,
        seed=seed,
        still_path=still,
        enhance_prompt=False,
    )
    # Move the auto log beside the experiment clip name.
    auto_log = dest.parent / "inputs" / dest.stem / "log.json"
    payload = {
        "variant": variant,
        "seed": seed,
        "prompt": prompt,
        "graph": graph_name,
        "multimodal_guider": bool(force_multimodal or scene.get("speakerId")),
        "include_music": include_music,
        "size": f"{CLIP_WIDTH}x{CLIP_HEIGHT}",
    }
    if auto_log.is_file():
        try:
            logged = json.loads(auto_log.read_text(encoding="utf-8"))
            payload["clip_log"] = logged
        except (OSError, json.JSONDecodeError):
            pass
    log_path(dest_root, variant, seed).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"  {variant} seed={seed} multimodal={payload['multimodal_guider']} "
        f"music={include_music} chars={len(prompt)}",
        flush=True,
    )
    mode = f"sound-ab {variant}"
    execute_queued_graph(graph, dest, prefer="video", mode=mode)
    free_comfy_models()
    return dest


def format_ratios(ratios: list | None) -> str:
    if not ratios:
        return "-"
    parts = []
    for idx, value in enumerate(ratios, start=2):
        parts.append(f"{idx}f={value if value is not None else '-'}")
    return " ".join(parts)


def write_report(dest_root: Path, rows: list[dict]) -> Path:
    md = dest_root / "report.md"
    json_path = dest_root / "report.json"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Sound prompt A/B — scene 02",
        "",
        "Metrics only. No winner claim.",
        "",
        "| Variant | Seed | LUFS | f0 Hz | harmonic ratios 2f-5f | spectral flatness | envelope mod | clip |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        m = row["metrics"]
        lines.append(
            "| {v} | {s} | {lufs} | {f0} | {harm} | {flat} | {env} | `{clip}` |".format(
                v=row["variant"],
                s=row["seed"],
                lufs=m.get("lufs"),
                f0=m.get("fundamental_hz"),
                harm=format_ratios(m.get("harmonic_ratios_2f_5f")),
                flat=m.get("spectral_flatness"),
                env=m.get("envelope_modulation"),
                clip=Path(row["clip"]).name,
            )
        )
    lines.extend(["", "## Prompts", ""])
    for row in rows:
        lines.append(f"### {row['variant']} seed {row['seed']}")
        lines.append("")
        lines.append("```")
        lines.append(row["prompt"])
        lines.append("```")
        lines.append("")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", default="the-iron-bride")
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--scene", type=int, default=2)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    if not os.environ.get("LTXV_API_KEY") and not args.analyze_only:
        raise SystemExit("Missing LTXV_API_KEY in content-pipeline/.env")

    show = load_show(discover_show_scripts(args.show)[0])
    episode = next(ep for ep in show["episodes"] if ep["episodeNumber"] == args.episode)
    scene = next(sc for sc in episode["scenes"] if sc["sceneNumber"] == args.scene)
    location = show["locations"][scene["locationId"]]
    still = start_still_path(show["id"], args.episode, args.scene)
    if not still.is_file():
        raise SystemExit(f"Missing start still {still}")

    dest_root = out_dir(show["id"], args.episode, args.scene)
    dest_root.mkdir(parents=True, exist_ok=True)
    generate_dir = ROOT / "output" / "generate" / show["id"] / str(args.episode)

    if not args.analyze_only:
        print("V1: reuse prior control clips", flush=True)
        reuse_v1(dest_root, generate_dir)

        free_comfy_models()
        for seed in SEEDS:
            render_variant(
                show=show,
                episode=episode,
                scene=scene,
                location=location,
                still=still,
                dest_root=dest_root,
                variant="V2",
                seed=seed,
                include_music=True,
                force_multimodal=False,
            )
        for seed in SEEDS:
            render_variant(
                show=show,
                episode=episode,
                scene=scene,
                location=location,
                still=still,
                dest_root=dest_root,
                variant="V3",
                seed=seed,
                include_music=False,
                force_multimodal=False,
            )
        for seed in SEEDS:
            render_variant(
                show=show,
                episode=episode,
                scene=scene,
                location=location,
                still=still,
                dest_root=dest_root,
                variant="V4",
                seed=seed,
                include_music=True,
                force_multimodal=True,
            )

    rows = []
    for variant in ("V1", "V2", "V3", "V4"):
        for seed in SEEDS:
            clip = clip_path(dest_root, variant, seed)
            meta_file = log_path(dest_root, variant, seed)
            if not clip.is_file() or clip.stat().st_size < 1024:
                print(f"skip missing {clip.name}", flush=True)
                continue
            if not meta_file.is_file():
                print(f"skip missing log {meta_file.name}", flush=True)
                continue
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            print(f"analyze {clip.name}", flush=True)
            metrics = analyze_clip(clip)
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "prompt": meta.get("prompt", ""),
                    "multimodal_guider": meta.get("multimodal_guider"),
                    "include_music": meta.get("include_music"),
                    "clip": str(clip),
                    "metrics": metrics,
                }
            )
    if not rows:
        raise SystemExit("No clips to analyze")

    report = write_report(dest_root, rows)
    print(f"Wrote {report}", flush=True)
    print(f"Clips in {dest_root}", flush=True)


if __name__ == "__main__":
    main()
