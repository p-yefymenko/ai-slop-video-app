const { comfyPaths, ensureComfyVenv, run } = require("./comfy-env.cjs");

const python = ensureComfyVenv();
const { comfyDir } = comfyPaths();

console.log(`Using ComfyUI venv interpreter: ${python}`);
run(python, ["-m", "pip", "install", "--upgrade", "pip"], comfyDir);
run(python, ["-m", "pip", "install", "-r", "requirements.txt"], comfyDir);
run(python, ["-m", "pip", "install", "huggingface_hub"], comfyDir);

console.log("Installing CUDA 12.8 PyTorch (required for RTX 50-series). Expect a large download the first time.");
run(python, ["-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"], comfyDir);
run(
  python,
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

const check = require("node:child_process").spawnSync(
  python,
  ["-c", "import torch; print(torch.__version__); print('cuda', torch.cuda.is_available()); print(torch.version.cuda)"],
  { encoding: "utf8" },
);
process.stdout.write(check.stdout || "");
process.stderr.write(check.stderr || "");
if (check.status !== 0 || !check.stdout.includes("cuda True")) {
  console.error("CUDA PyTorch did not report a GPU. Check NVIDIA drivers, then rerun `pnpm run content:comfy-torch`.");
  process.exit(1);
}
