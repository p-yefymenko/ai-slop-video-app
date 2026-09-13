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
const python = findPython();
if (!python) {
  console.error(
    "Python 3 was not found. Install it from https://www.python.org/downloads/ and rerun this command.",
  );
  process.exit(1);
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
