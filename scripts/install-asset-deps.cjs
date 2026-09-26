const path = require("node:path");
const { ensureComfyVenv, findComfyPython, run } = require("./comfy-env.cjs");

const python = findComfyPython() || ensureComfyVenv();
const repoRoot = path.resolve(__dirname, "..");

console.log(`Installing trimesh, moderngl, transformers, timm, and einops into ${python}`);
run(python, ["-m", "pip", "install", "trimesh", "moderngl", "transformers", "timm", "einops"], repoRoot);
console.log("trimesh, moderngl, transformers, timm, and einops are installed. content:assets can read glTF and OBJ meshes, content:previs draws clay frames on the GPU, and content:landmarks grounds landmark names with Florence-2.");
