"""The ComfyUI client every stage shares: queue a graph, wait, download its outputs.

Graphs are built in Python (``wan.py``, ``asset_generate.py``); this module only
runs them. Every output a graph saves has a filename prefix, and the caller says
which prefix goes to which file, so a preview another node saves is never taken
for the result.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMFY_INPUT_DIR = ROOT / ".comfyui" / "input"
COMFYUI_URL = "http://127.0.0.1:8188"
POLL_SECONDS = 2


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


def format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{seconds:.1f}s"


def _post(path: str, payload: dict, timeout: float = 30) -> dict:
    request = urllib.request.Request(
        f"{COMFYUI_URL}{path}", data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body.strip() else {}


def keep_awake() -> None:
    """Ask Windows not to sleep while this process runs (the screen may still turn off). A busy GPU
    is not activity to Windows: on its idle timer it slept in the middle of a render, which froze
    for hours. The request ends when the process exits."""
    if os.name == "nt":
        import ctypes

        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)


def require_comfy() -> None:
    try:
        urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=3)
    except urllib.error.URLError as exc:
        raise SystemExit(f"ComfyUI is not reachable at {COMFYUI_URL}. Start it with `pnpm run content:comfy`.") from exc
    keep_awake()


def free_comfy_models() -> None:
    """Unload every model, so the next graph (or another GPU process) has the whole card."""
    try:
        _post("/free", {"unload_models": True, "free_memory": True})
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"Could not unload ComfyUI models ({exc}); continuing.", flush=True)


def interrupt_comfy() -> None:
    """Stop whatever ComfyUI is running and drop its queue: a hung prompt blocks every later one."""
    for path, payload in (("/interrupt", {}), ("/queue", {"clear": True})):
        try:
            _post(path, payload)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"Could not reach ComfyUI {path} ({exc}).", flush=True)


def queue_prompt(graph: dict) -> str:
    body = _post("/prompt", {"prompt": graph}, timeout=60)
    if body.get("node_errors"):
        raise RuntimeError(f"ComfyUI rejected the workflow: {json.dumps(body['node_errors'])}")
    return body["prompt_id"]


def _files(item: dict) -> list[dict]:
    files: list[dict] = []
    for node_output in (item.get("outputs") or {}).values():
        if isinstance(node_output, dict):
            for key in ("images", "gifs", "videos", "audio", "3d"):
                files.extend(node_output.get(key) or [])
    return files


def wait_for_files(prompt_id: str, timeout_s: float) -> tuple[list[dict], dict]:
    """Every file the finished prompt saved, and its history entry. Raises on a model error or a timeout."""
    started = time.time()
    last_note = started
    while time.time() - started < timeout_s:
        with urllib.request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}") as response:
            item = json.loads(response.read().decode("utf-8")).get(prompt_id)
        if item:
            status = item.get("status") or {}
            for message in status.get("messages") or []:
                if isinstance(message, (list, tuple)) and message and message[0] == "execution_error":
                    payload = message[1] if len(message) > 1 else {}
                    raise RuntimeError(payload.get("exception_message") if isinstance(payload, dict) else str(payload))
            if status.get("completed"):
                return _files(item), item
        if time.time() - last_note >= 60:
            print(f"  ... {format_duration(time.time() - started)} elapsed", flush=True)
            last_note = time.time()
        time.sleep(POLL_SECONDS)
    raise TimeoutError(f"ComfyUI prompt {prompt_id} did not finish in {format_duration(timeout_s)}")


def output_named(files: list[dict], prefix: str) -> dict:
    """The one saved file whose name starts with this prefix."""
    matches = [info for info in files if str(info.get("filename", "")).startswith(f"{prefix}_")]
    # VHS saves a silent copy beside the one with sound; the one with sound is the take.
    with_audio = [info for info in matches if "-audio." in str(info.get("filename", ""))]
    matches = with_audio or matches
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {prefix} output, got {[info.get('filename') for info in files]}")
    return matches[0]


def download_output(file_info: dict, dest: Path) -> None:
    query = urllib.parse.urlencode(
        {"filename": file_info["filename"], "subfolder": file_info.get("subfolder", ""), "type": file_info.get("type", "output")}
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"{COMFYUI_URL}/view?{query}") as response, dest.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def run_graph(graph: dict, outputs: dict[str, Path], label: str, timeout_s: float) -> float:
    """Queue ``graph``, wait, and download each ``prefix -> file`` of ``outputs``. Returns the seconds it took.

    A run past ``timeout_s`` is stopped in ComfyUI too, so it never blocks the next one.
    """
    print(f"  {label}...", flush=True)
    started = time.time()
    prompt_id = queue_prompt(graph)
    try:
        files, _item = wait_for_files(prompt_id, timeout_s)
    except TimeoutError:
        interrupt_comfy()
        raise
    for prefix, dest in outputs.items():
        download_output(output_named(files, prefix), dest)
    elapsed = time.time() - started
    print(f"  {label}: done in {format_duration(elapsed)} -> {', '.join(str(path) for path in outputs.values())}", flush=True)
    return elapsed


def stage_input(path: Path, prefix: str) -> str:
    """Copy a file into ComfyUI's input folder under a prefixed name; returns that name for a Load node."""
    COMFY_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = COMFY_INPUT_DIR / f"{prefix}_{path.name}"
    shutil.copy2(path, dest)
    return dest.name


def stable_seed(*parts: object) -> int:
    """A reproducible unsigned 32-bit seed for one render identity."""
    key = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2s(key, digest_size=4).digest(), "big")


def clone_workflow(workflow_template: dict) -> dict:
    raw = workflow_template["prompt"] if "prompt" in workflow_template else workflow_template
    return json.loads(json.dumps(raw))


def write_solid_png(path: Path, color: tuple[int, int, int], width: int = 768, height: int = 1360) -> None:
    pixel = bytes(color)
    raw = b"".join(b"\x00" + (pixel * width) for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def present(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 1024


def inputs_dir(dest: Path) -> Path:
    """Beside every output: what it was made from (prompts, copies of its pictures, the check report)."""
    return dest.parent / "inputs" / dest.stem


def write_log(dest: Path, entry: dict, files: dict[str, Path] | None = None) -> None:
    """Record how ``dest`` was made, with a copy of each input file, in ``inputs/<name>/log.json``."""
    folder = inputs_dir(dest)
    folder.mkdir(parents=True, exist_ok=True)
    copied = {}
    for role, source in (files or {}).items():
        target = folder / f"{role}{source.suffix.lower()}"
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        copied[role] = target.name
    (folder / "log.json").write_text(json.dumps({"output": dest.name, **entry, "files": copied}, indent=2) + "\n", encoding="utf-8")

