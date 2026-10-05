const { spawn } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");
const { comfyPaths, findComfyPython } = require("./comfy-env.cjs");

const { comfyDir } = comfyPaths();
const python = findComfyPython();
const backdropNode = path.join(__dirname, "..", "content-pipeline", "comfy_nodes", "qwen_backdrop.py");
const customNodes = path.join(comfyDir, "custom_nodes");
fs.mkdirSync(customNodes, { recursive: true });
fs.copyFileSync(backdropNode, path.join(customNodes, "qwen_backdrop.py"));

if (!python) {
  console.error("ComfyUI is not set up yet. Run `pnpm run content:setup-comfy` first.");
  process.exit(1);
}

// No default --reserve-vram: content:frames regressed with it (slower stills).
// Override any args via COMFY_ARGS (replaces defaults entirely when set).
const extraArgs = (process.env.COMFY_ARGS || "")
  .split(/\s+/)
  .map((part) => part.trim())
  .filter(Boolean);
const comfyArgs = ["main.py", "--listen", "127.0.0.1", "--port", "8188", ...extraArgs];
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
