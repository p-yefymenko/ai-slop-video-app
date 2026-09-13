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

for (const folder of ["diffusion_models", "vae", "checkpoints", "text_encoders", "loras"]) {
  fs.mkdirSync(path.join(comfyDir, "models", folder), { recursive: true });
}

const workflowSrcDir = path.join(repoRoot, "content-pipeline", "workflows");
const workflowDestDir = path.join(comfyDir, "user", "default", "workflows");
fs.mkdirSync(workflowDestDir, { recursive: true });
for (const name of fs.readdirSync(workflowSrcDir)) {
  if (name.endsWith(".json")) {
    fs.copyFileSync(path.join(workflowSrcDir, name), path.join(workflowDestDir, name));
  }
}

const nodeSrc = path.join(repoRoot, "content-pipeline", "comfy_nodes", "reelshort_ltx");
const nodeDest = path.join(comfyDir, "custom_nodes", "reelshort_ltx");
fs.cpSync(nodeSrc, nodeDest, { recursive: true });

const result = spawnSync(python, [script, comfyDir], {
  stdio: "inherit",
  cwd: repoRoot,
});

process.exit(result.status ?? 1);
