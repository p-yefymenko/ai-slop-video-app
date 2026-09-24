const path = require("node:path");
const { ensureComfyVenv, findComfyPython, run } = require("./comfy-env.cjs");

const python = findComfyPython() || ensureComfyVenv();
const repoRoot = path.resolve(__dirname, "..");

console.log(`Installing trimesh and moderngl into ${python}`);
run(python, ["-m", "pip", "install", "trimesh", "moderngl"], repoRoot);
console.log("trimesh and moderngl are installed. content:assets can read glTF and OBJ meshes, and content:previs draws clay frames on the GPU.");
