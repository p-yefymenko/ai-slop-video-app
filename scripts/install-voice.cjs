// The two voice envs, each its own because their pins clash with ComfyUI's and with each other's
// (Chatterbox needs transformers 5.x, qwen-tts pins 4.57.3). Installed with uv, as tested.
// - content-pipeline/.voice (Python 3.11): Chatterbox, which speaks the lines.
// - content-pipeline/.voicedesign (Python 3.12): Qwen3-TTS VoiceDesign, which makes each character's voice from its description.
const { spawnSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");
const { findPython } = require("./find-python.cjs");

const repoRoot = path.resolve(__dirname, "..");
const pipeline = path.join(repoRoot, "content-pipeline");
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

function installEnv(folder, pythonVersion, requirements) {
  const venv = path.join(pipeline, folder);
  const envPython = path.join(venv, process.platform === "win32" ? "Scripts/python.exe" : "bin/python");
  if (!fs.existsSync(envPython)) {
    run(["venv", venv, "--python", pythonVersion]);
  }
  run(["pip", "install", "--python", envPython, "torch==2.7.1", "torchaudio==2.7.1", "--index-url", "https://download.pytorch.org/whl/cu128"]);
  run(["pip", "install", "--python", envPython, "--no-deps", "-r", path.join(pipeline, requirements)]);
  return venv;
}

spawnSync(python, ["-m", "pip", "install", "--quiet", "uv"], { stdio: "inherit" });
const voice = installEnv(".voice", "3.11", "voice-requirements.txt");
const design = installEnv(".voicedesign", "3.12", "voice-design-requirements.txt");
console.log(`Chatterbox is installed in ${voice}; content:generate speaks each scene's lines with it.`);
console.log(`Qwen3-TTS VoiceDesign is installed in ${design}; content:plates makes each character's voice with it.`);
