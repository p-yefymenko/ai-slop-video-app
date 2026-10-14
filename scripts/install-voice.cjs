// The Chatterbox voice env, content-pipeline/.voice (Python 3.11): its own env because
// Chatterbox pins packages that clash with ComfyUI's. Installed with uv, as tested.
const { spawnSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");
const { findPython } = require("./find-python.cjs");

const repoRoot = path.resolve(__dirname, "..");
const pipeline = path.join(repoRoot, "content-pipeline");
const venv = path.join(pipeline, ".voice");
const voicePython = path.join(venv, process.platform === "win32" ? "Scripts/python.exe" : "bin/python");
const python = findPython();
if (!python) {
  console.error("Python 3 was not found. Install it from https://www.python.org/downloads/ and rerun this command.");
  process.exit(1);
}

function run(args) {
  const result = spawnSync(python, ["-m", "uv", ...args], { stdio: "inherit", cwd: pipeline });
  if (result.status !== 0) {
    throw new Error(`uv ${args.join(" ")} failed with exit ${result.status}`);
  }
}

spawnSync(python, ["-m", "pip", "install", "--quiet", "uv"], { stdio: "inherit" });
if (!fs.existsSync(voicePython)) {
  run(["venv", venv, "--python", "3.11"]);
}
run(["pip", "install", "--python", voicePython, "torch==2.7.1", "torchaudio==2.7.1", "--index-url", "https://download.pytorch.org/whl/cu128"]);
run(["pip", "install", "--python", voicePython, "--no-deps", "-r", path.join(pipeline, "voice-requirements.txt")]);
console.log(`Chatterbox is installed in ${venv}. content:generate speaks each scene's lines with it.`);
