#!/usr/bin/env python3
"""Render episode scripts via a local ComfyUI + LTX workflow."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts_input"
OUTPUT_DIR = ROOT / "output"
WORKFLOW_PATH = ROOT / "workflows" / "ltx_gemma_api.json"
COMFYUI_URL = "http://127.0.0.1:8188"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def queue_prompt(prompt_graph: dict) -> str:
    payload = json.dumps({"prompt": prompt_graph}).encode("utf-8")
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as res:
        body = json.loads(res.read().decode("utf-8"))
    return body["prompt_id"]


def wait_for_output(prompt_id: str, timeout_s: int = 1800) -> list[dict]:
    started = time.time()
    while time.time() - started < timeout_s:
        req = urllib.request.Request(f"{COMFYUI_URL}/history/{prompt_id}")
        with urllib.request.urlopen(req) as res:
            history = json.loads(res.read().decode("utf-8"))
        item = history.get(prompt_id)
        if item and item.get("outputs"):
            files: list[dict] = []
            for node_output in item["outputs"].values():
                for video in node_output.get("gifs", []) + node_output.get("videos", []):
                    files.append(video)
                for image in node_output.get("images", []):
                    files.append(image)
            if files:
                return files
        time.sleep(2)
    raise TimeoutError(f"ComfyUI prompt {prompt_id} did not finish in time")


def download_output(file_info: dict, dest: Path) -> None:
    filename = file_info["filename"]
    subfolder = file_info.get("subfolder", "")
    file_type = file_info.get("type", "output")
    query = urllib.parse.urlencode(
        {"filename": filename, "subfolder": subfolder, "type": file_type}
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"{COMFYUI_URL}/view?{query}") as res, dest.open("wb") as handle:
        shutil.copyfileobj(res, handle)


def inject_prompt(workflow: dict, prompt: str) -> dict:
    raw = workflow["prompt"] if "prompt" in workflow else workflow
    graph = json.loads(json.dumps(raw))
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type", "")
        title = str((node.get("_meta") or {}).get("title", ""))
        if class_type in {"CLIPTextEncode", "CLIPTextEncodeLTXV"} or "Gemma" in title:
            node.setdefault("inputs", {})["text"] = prompt
            return graph
    raise RuntimeError("Could not find a text-conditioning node in the ComfyUI workflow")


def concat_with_ffmpeg(scene_files: list[Path], dest: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not scene_files:
        return False
    list_file = dest.with_suffix(".txt")
    list_file.write_text("".join(f"file '{path.as_posix()}'\n" for path in scene_files), encoding="utf-8")
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(dest)],
        check=False,
        capture_output=True,
        text=True,
    )
    list_file.unlink(missing_ok=True)
    return result.returncode == 0


def generate_episode(script_path: Path, workflow_template: dict) -> None:
    script = load_json(script_path)
    series = script["series"]
    episode_number = script["episodeNumber"]
    out_dir = OUTPUT_DIR / series / str(episode_number)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(script, indent=2), encoding="utf-8")

    scene_files: list[Path] = []
    for scene in script["scenes"]:
        graph = inject_prompt(workflow_template, scene["prompt"])
        prompt_id = queue_prompt(graph)
        outputs = wait_for_output(prompt_id)
        dest = out_dir / f"scene_{scene['sceneNumber']:02d}.mp4"
        download_output(outputs[0], dest)
        scene_files.append(dest)
        print(f"Wrote {dest}")

    episode_mp4 = out_dir / "episode.mp4"
    if concat_with_ffmpeg(scene_files, episode_mp4):
        print(f"Concatenated {episode_mp4}")
    elif len(scene_files) == 1:
        shutil.copyfile(scene_files[0], episode_mp4)
        print(f"Copied single scene to {episode_mp4}")
    else:
        print("ffmpeg not found; left individual scene files. Install ffmpeg to concat.")


def main() -> None:
    workflow_template = load_json(WORKFLOW_PATH)
    scripts = sorted(SCRIPTS_DIR.glob("*/*.json"))
    if not scripts:
        raise SystemExit(f"No JSON scripts found in {SCRIPTS_DIR}")
    try:
        urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=3)
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"ComfyUI is not reachable at {COMFYUI_URL}. Start it locally with the LTX Q4_K_M workflow loaded."
        ) from exc
    for script_path in scripts:
        print(f"Generating {script_path}")
        generate_episode(script_path, workflow_template)


if __name__ == "__main__":
    main()
