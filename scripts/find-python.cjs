const { spawnSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

function isUsablePython(filePath) {
  if (!filePath || !fs.existsSync(filePath)) return false;
  const normalized = filePath.replaceAll("\\", "/").toLowerCase();
  if (normalized.includes("/windowsapps/")) return false;
  return true;
}

function pythonFromDir(dir) {
  const names =
    process.platform === "win32"
      ? ["python.exe", path.join("Scripts", "python.exe")]
      : ["bin/python", "bin/python3", "python3", "python"];
  for (const name of names) {
    const candidate = path.join(dir, name);
    if (isUsablePython(candidate)) return candidate;
  }
  return null;
}

function findWindowsPython() {
  const localPrograms = path.join(process.env.LOCALAPPDATA ?? "", "Programs", "Python");
  if (fs.existsSync(localPrograms)) {
    const versions = fs
      .readdirSync(localPrograms, { withFileTypes: true })
      .filter((entry) => entry.isDirectory() && /^Python3\d+$/i.test(entry.name))
      .map((entry) => entry.name)
      .sort()
      .reverse();
    for (const version of versions) {
      const found = pythonFromDir(path.join(localPrograms, version));
      if (found) return found;
    }
  }

  for (const envName of ["VIRTUAL_ENV", "CONDA_PREFIX"]) {
    const prefix = process.env[envName];
    if (!prefix) continue;
    const found = pythonFromDir(prefix);
    if (found) return found;
  }

  const which = spawnSync("where.exe", ["python"], { encoding: "utf8" });
  if (which.status === 0) {
    for (const line of which.stdout.split(/\r?\n/)) {
      const candidate = line.trim();
      if (isUsablePython(candidate)) return candidate;
    }
  }

  return null;
}

function findUnixPython() {
  for (const cmd of ["python3", "python"]) {
    const result = spawnSync(cmd, ["-c", "import sys; print(sys.executable)"], {
      encoding: "utf8",
    });
    if (result.status === 0) {
      const executable = result.stdout.trim();
      if (executable) return executable;
    }
  }
  return null;
}

function findPythonInDirs(preferredDirs = []) {
  for (const dir of preferredDirs) {
    const found = pythonFromDir(dir);
    if (found) return found;
  }
  return null;
}

function findPython(preferredDirs = []) {
  return findPythonInDirs(preferredDirs) ?? (process.platform === "win32" ? findWindowsPython() : findUnixPython());
}

module.exports = { findPython, findPythonInDirs };
