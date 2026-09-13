#!/usr/bin/env python3
"""Render episode scripts via a local ComfyUI + LTX workflow."""

from __future__ import annotations

import json
import os
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


def load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


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
    node_errors = body.get("node_errors") or {}
    if node_errors:
        raise RuntimeError(f"ComfyUI rejected the workflow: {json.dumps(node_errors)}")
    return body["prompt_id"]


def wait_for_output(prompt_id: str, timeout_s: int = 7200) -> list[dict]:
    started = time.time()
    while time.time() - started < timeout_s:
        req = urllib.request.Request(f"{COMFYUI_URL}/history/{prompt_id}")
        with urllib.request.urlopen(req) as res:
            history = json.loads(res.read().decode("utf-8"))
        item = history.get(prompt_id)
        if item:
            status = item.get("status") or {}
            for message in status.get("messages") or []:
                if isinstance(message, (list, tuple)) and message and message[0] == "execution_error":
                    payload = message[1] if len(message) > 1 else {}
                    if isinstance(payload, dict):
                        raise RuntimeError(
                            payload.get("exception_message")
                            or payload.get("exception_type")
                            or json.dumps(payload)
                        )
                    raise RuntimeError(str(payload))
            files: list[dict] = []
            for node_output in (item.get("outputs") or {}).values():
                for video in node_output.get("gifs", []) + node_output.get("videos", []):
                    files.append(video)
                for image in node_output.get("images", []):
                    files.append(image)
            if files:
                return files
            if status.get("completed"):
                raise RuntimeError(
                    f"ComfyUI prompt {prompt_id} finished without a video or image output"
                )
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


def inject_prompt(workflow: dict, prompt: str, api_key: str) -> dict:
    raw = workflow["prompt"] if "prompt" in workflow else workflow
    graph = json.loads(json.dumps(raw))
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type", "")
        title = str((node.get("_meta") or {}).get("title", ""))
        if class_type in {"GemmaAPITextEncode", "CLIPTextEncode", "CLIPTextEncodeLTXV"} or "Gemma" in title:
            node.setdefault("inputs", {})["prompt" if class_type == "GemmaAPITextEncode" else "text"] = prompt
            if class_type == "GemmaAPITextEncode" and api_key:
                node["inputs"]["api_key"] = api_key
                node["inputs"]["ckpt_name"] = node["inputs"].get("ckpt_name") or "ltx-2.3-22b-distilled-api-id.safetensors"
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
        print(f"Queued scene {scene['sceneNumber']}...", flush=True)
        dest = out_dir / f"scene_{scene['sceneNumber']:02d}.mp4"
        if dest.exists() and dest.stat().st_size > 1024:
            print(f"Already present: {dest}", flush=True)
            scene_files.append(dest)
            continue
        graph = inject_prompt(workflow_template, scene["prompt"], os.environ.get("LTXV_API_KEY", ""))
        prompt_id = queue_prompt(graph)
        outputs = wait_for_output(prompt_id)
        download_output(outputs[0], dest)
        scene_files.append(dest)
        print(f"Wrote {dest}", flush=True)

    episode_mp4 = out_dir / "episode.mp4"
    if concat_with_ffmpeg(scene_files, episode_mp4):
        print(f"Concatenated {episode_mp4}")
    elif len(scene_files) == 1:
        shutil.copyfile(scene_files[0], episode_mp4)
        print(f"Copied single scene to {episode_mp4}")
    else:
        print("ffmpeg not found; left individual scene files. Install ffmpeg to concat.")


def main() -> None:
    load_dotenv()
    if not os.environ.get("LTXV_API_KEY"):
        raise SystemExit("Missing LTXV_API_KEY in content-pipeline/.env")
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
        print(f"Generating {script_path}", flush=True)
        generate_episode(script_path, workflow_template)


if __name__ == "__main__":
    main()
