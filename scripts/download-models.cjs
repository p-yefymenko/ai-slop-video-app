const { spawnSync } = require("node:child_process");
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

// download_models.py creates each models folder it writes into.

const result = spawnSync(python, [script, comfyDir], {
  stdio: "inherit",
  cwd: repoRoot,
});

process.exit(result.status ?? 1);
