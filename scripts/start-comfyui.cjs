const { spawn } = require("node:child_process");
const path = require("node:path");
const { comfyPaths, findComfyPython } = require("./comfy-env.cjs");

const { comfyDir } = comfyPaths();
const python = findComfyPython();

if (!python) {
  console.error("ComfyUI is not set up yet. Run `pnpm run content:setup-comfy` first.");
  process.exit(1);
}

// The Wan 14B experts are offloaded to RAM on a 16 GB card; reserving 1.5 GB of VRAM keeps
// ComfyUI from spilling into shared memory, which runs many times slower.
// COMFY_ARGS replaces these defaults entirely when set.
const extraArgs = (process.env.COMFY_ARGS || "--reserve-vram 1.5")
  .split(/\s+/)
  .map((part) => part.trim())
  .filter(Boolean);
// The pipeline's own nodes (content-pipeline/comfy_nodes), whatever the other args.
const pipelinePaths = path.join(__dirname, "..", "content-pipeline", "comfy_paths.yaml");
const comfyArgs = ["main.py", "--listen", "127.0.0.1", "--port", "8188", "--extra-model-paths-config", pipelinePaths, ...extraArgs];
if (extraArgs.length) {
  console.log(`ComfyUI args from COMFY_ARGS: ${extraArgs.join(" ")}`);
} else {
  console.log("ComfyUI args: defaults only (set COMFY_ARGS to override, e.g. --reserve-vram 2)");
}
console.log(`Starting ComfyUI with ${python} ${comfyArgs.slice(1).join(" ")}`);
const child = spawn(python, comfyArgs, {
  cwd: comfyDir,
  stdio: "inherit",
  env: process.env,
});

child.on("exit", (code) => {
  process.exit(code ?? 1);
});
