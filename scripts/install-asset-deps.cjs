const path = require("node:path");
const { ensureComfyVenv, findComfyPython, run } = require("./comfy-env.cjs");

const python = findComfyPython() || ensureComfyVenv();
const repoRoot = path.resolve(__dirname, "..");

console.log(`Installing trimesh, moderngl, and pymatting into ${python}`);
run(python, ["-m", "pip", "install", "trimesh", "moderngl", "pymatting"], repoRoot);
console.log(
  "trimesh, moderngl, and pymatting are installed. content:assets can read glTF and OBJ meshes, content:previs draws the world on the GPU, and content:frames can lift a cut-out person's own colors off their backdrop.",
);
