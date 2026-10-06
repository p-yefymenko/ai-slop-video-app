const { spawnSync } = require("node:child_process");
const path = require("node:path");
const { findPython } = require("./find-python.cjs");

const script = process.argv[2];
if (!script) {
  console.error("Usage: node scripts/run-python.cjs <script> [args...]");
  process.exit(1);
}

const extraArgs = process.argv.slice(3);
const repoRoot = path.resolve(__dirname, "..");
const comfyDir = path.join(repoRoot, "content-pipeline", ".comfyui");
const python = findPython([
  path.join(comfyDir, ".venv", "Scripts"),
  path.join(comfyDir, ".venv", "bin"),
  path.join(comfyDir, ".venv"),
]) || findPython();
if (!python) {
  console.error(
    "Python 3 was not found. Install it from https://www.python.org/downloads/ and rerun this command.",
  );
  process.exit(1);
}

// The ShowScript schema is the one strict gate. Every pipeline stage reads the
// scripts, so none starts on a script that fails it. Test runs (-m) skip this.
if (script.startsWith("content-pipeline/scripts/")) {
  const validation = spawnSync("pnpm", ["--silent", "run", "content:validate"], {
    stdio: "inherit",
    cwd: repoRoot,
    shell: process.platform === "win32",
  });
  if (validation.error) {
    console.error(validation.error.message);
    process.exit(1);
  }
  if (validation.status !== 0) {
    console.error("Show scripts failed validation; fix them before running this stage.");
    process.exit(validation.status ?? 1);
  }
}

const env = { ...process.env, PYTHONUNBUFFERED: "1" };
const result = spawnSync(python, [script, ...extraArgs], {
  stdio: "inherit",
  cwd: repoRoot,
  env,
});

if (result.error) {
  console.error(result.error.message);
  process.exit(1);
}

process.exit(result.status ?? 1);
