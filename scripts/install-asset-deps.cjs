const path = require("node:path");
const { ensureComfyVenv, findComfyPython, run } = require("./comfy-env.cjs");

const python = findComfyPython() || ensureComfyVenv();
const repoRoot = path.resolve(__dirname, "..");

console.log(`Installing trimesh and fast-simplification into ${python}`);
run(python, ["-m", "pip", "install", "trimesh", "fast-simplification"], repoRoot);
console.log("trimesh and fast-simplification are installed. content:assets can read meshes and simplify them.");
