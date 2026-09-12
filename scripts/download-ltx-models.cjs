const { spawnSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");
const { findPython } = require("./find-python.cjs");

const repoRoot = path.resolve(__dirname, "..");
const comfyDir = path.join(repoRoot, "content-pipeline", ".comfyui");
const script = path.join(repoRoot, "content-pipeline", "scripts", "download_models.py");
const python = findPython([
  path.join(comfyDir, ".venv", "Scripts"),
  path.join(comfyDir, ".venv", "bin"),
  path.join(comfyDir, ".venv"),
]);

if (!python) {
  console.error("ComfyUI is not set up yet. Run `pnpm run content:setup-comfy` first.");
  process.exit(1);
}

fs.mkdirSync(path.join(comfyDir, "models", "diffusion_models"), { recursive: true });
fs.mkdirSync(path.join(comfyDir, "models", "vae"), { recursive: true });
fs.mkdirSync(path.join(comfyDir, "models", "checkpoints"), { recursive: true });

const workflowSrc = path.join(repoRoot, "content-pipeline", "workflows", "ltx_gemma_api.json");
const workflowDestDir = path.join(comfyDir, "user", "default", "workflows");
fs.mkdirSync(workflowDestDir, { recursive: true });
fs.copyFileSync(workflowSrc, path.join(workflowDestDir, "ltx_gemma_api.json"));

const result = spawnSync(python, [script, comfyDir], {
  stdio: "inherit",
  cwd: repoRoot,
});

process.exit(result.status ?? 1);
