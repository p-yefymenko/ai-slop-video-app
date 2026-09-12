const { spawn } = require("node:child_process");
const { comfyPaths, findComfyPython } = require("./comfy-env.cjs");

const { comfyDir } = comfyPaths();
const python = findComfyPython();

if (!python) {
  console.error("ComfyUI is not set up yet. Run `pnpm run content:setup-comfy` first.");
  process.exit(1);
}

console.log(`Starting ComfyUI with ${python}`);
const child = spawn(python, ["main.py", "--listen", "127.0.0.1", "--port", "8188"], {
  cwd: comfyDir,
  stdio: "inherit",
  env: process.env,
});

child.on("exit", (code) => process.exit(code ?? 1));
