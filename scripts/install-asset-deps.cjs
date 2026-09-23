const path = require("node:path");
const { ensureComfyVenv, findComfyPython, run } = require("./comfy-env.cjs");

const python = findComfyPython() || ensureComfyVenv();
const repoRoot = path.resolve(__dirname, "..");

console.log(`Installing trimesh into ${python}`);
run(python, ["-m", "pip", "install", "trimesh"], repoRoot);
console.log("trimesh is installed. Downloaded glTF and OBJ files can be read by content:assets.");
