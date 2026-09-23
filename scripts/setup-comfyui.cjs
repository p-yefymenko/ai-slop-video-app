const { spawnSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");
const { findPython } = require("./find-python.cjs");
const { comfyPaths, ensureComfyVenv, run: runPython } = require("./comfy-env.cjs");

const repoRoot = path.resolve(__dirname, "..");
const comfyDir = path.join(repoRoot, "content-pipeline", ".comfyui");
const customNodes = path.join(comfyDir, "custom_nodes");
const workflowSrcDir = path.join(repoRoot, "content-pipeline", "workflows");

const REPOS = [
  { url: "https://github.com/comfyanonymous/ComfyUI.git", dir: comfyDir },
  { url: "https://github.com/Lightricks/ComfyUI-LTXVideo.git", dir: path.join(customNodes, "ComfyUI-LTXVideo") },
  { url: "https://github.com/city96/ComfyUI-GGUF.git", dir: path.join(customNodes, "ComfyUI-GGUF") },
  { url: "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git", dir: path.join(customNodes, "ComfyUI-VideoHelperSuite") },
];

function run(cmd, args, cwd) {
  const result = spawnSync(cmd, args, { stdio: "inherit", cwd, shell: process.platform === "win32" });
  if (result.status !== 0) {
    throw new Error(`${cmd} ${args.join(" ")} failed with exit ${result.status}`);
  }
}

function cloneIfMissing(url, dir) {
  if (fs.existsSync(path.join(dir, ".git"))) {
    console.log(`Already present: ${dir}`);
    return;
  }
  fs.mkdirSync(path.dirname(dir), { recursive: true });
  console.log(`Cloning ${url}`);
  run("git", ["clone", "--depth", "1", url, dir], repoRoot);
}

const python = findPython();
if (!python) {
  console.error("Python 3 was not found. Install it, then rerun `pnpm run content:setup-comfy`.");
  process.exit(1);
}

for (const repo of REPOS) {
  cloneIfMissing(repo.url, repo.dir);
}

const pipPython = ensureComfyVenv();
const { comfyDir: ensuredComfyDir } = comfyPaths();
if (ensuredComfyDir !== comfyDir) {
  throw new Error("ComfyUI path mismatch");
}

console.log("Installing ComfyUI Python deps (this can take several minutes)...");
runPython(pipPython, ["-m", "pip", "install", "--upgrade", "pip"], comfyDir);
runPython(pipPython, ["-m", "pip", "install", "-r", "requirements.txt"], comfyDir);
runPython(pipPython, ["-m", "pip", "install", "huggingface_hub", "imageio-ffmpeg"], comfyDir);
console.log("Installing CUDA 12.8 PyTorch (required for RTX 50-series)...");
runPython(pipPython, ["-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"], comfyDir);
runPython(
  pipPython,
  [
    "-m",
    "pip",
    "install",
    "--upgrade",
    "torch",
    "torchvision",
    "torchaudio",
    "--index-url",
    "https://download.pytorch.org/whl/cu128",
  ],
  comfyDir,
);

const workflowDestDir = path.join(comfyDir, "user", "default", "workflows");
fs.mkdirSync(workflowDestDir, { recursive: true });
for (const name of fs.readdirSync(workflowSrcDir)) {
  if (name.endsWith(".json")) {
    fs.copyFileSync(path.join(workflowSrcDir, name), path.join(workflowDestDir, name));
  }
}

const nodeSrc = path.join(repoRoot, "content-pipeline", "comfy_nodes", "reelshort_ltx");
const nodeDest = path.join(customNodes, "reelshort_ltx");
fs.cpSync(nodeSrc, nodeDest, { recursive: true });

for (const folder of ["diffusion_models", "checkpoints", "vae", "text_encoders", "loras", "controlnet"]) {
  fs.mkdirSync(path.join(comfyDir, "models", folder), { recursive: true });
}

console.log(`
ComfyUI is cloned at content-pipeline/.comfyui
Workflows copied to ComfyUI user/default/workflows/ (ltx_gemma_api.json, qwen_image_edit.json, qwen_image_edit_spatial.json)

Next:
  1. pnpm run content:comfy-torch  # CUDA PyTorch for the 5070 Ti (if setup did not already install it)
  2. pnpm run content:models       # downloads LTX video weights, the Qwen-Image-Edit still stack, and the InstantX Union ControlNet
  3. pnpm run content:comfy        # leave this running (http://127.0.0.1:8188)
  4. In the ComfyUI UI: Load → qwen_image_edit.json, qwen_image_edit_spatial.json, then ltx_gemma_api.json, and confirm no missing-node errors
  5. pnpm run content:frames, then pnpm run content:generate
`);
