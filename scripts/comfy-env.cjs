const { spawnSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");
const { findPython, findPythonInDirs } = require("./find-python.cjs");

function comfyPaths() {
  const repoRoot = path.resolve(__dirname, "..");
  const comfyDir = path.join(repoRoot, "content-pipeline", ".comfyui");
  const venvDir = path.join(comfyDir, ".venv");
  return { repoRoot, comfyDir, venvDir };
}

function findComfyPython() {
  const { venvDir } = comfyPaths();
  return findPythonInDirs([venvDir, path.join(venvDir, "Scripts"), path.join(venvDir, "bin")]);
}

function run(python, args, cwd) {
  const result = spawnSync(python, args, { stdio: "inherit", cwd });
  if (result.status !== 0) {
    throw new Error(`${python} ${args.join(" ")} failed with exit ${result.status}`);
  }
}

function ensureComfyVenv() {
  const { comfyDir, venvDir } = comfyPaths();
  if (!fs.existsSync(path.join(comfyDir, "main.py"))) {
    console.error("ComfyUI is not set up yet. Run `pnpm run content:setup-comfy` first.");
    process.exit(1);
  }
  let python = findComfyPython();
  if (!python) {
    const systemPython = findPython();
    if (!systemPython) {
      console.error("Python 3 was not found.");
      process.exit(1);
    }
    console.log("Creating ComfyUI virtualenv...");
    run(systemPython, ["-m", "venv", venvDir], comfyDir);
    python = findComfyPython();
  }
  if (!python) {
    console.error("Could not find the ComfyUI venv interpreter.");
    process.exit(1);
  }
  return python;
}

module.exports = { comfyPaths, findComfyPython, ensureComfyVenv, run };
