# Location meshes

An empty location becomes one picture, then one mesh, for that show only. A location with people becomes one picture and one mesh per landmark, fitted to the size in the script.

`pnpm run content:plates` has Qwen-Image-Edit-2511 draw and stop. An empty location's picture is `output/plates/<show>/<locationId>/plate.png`, prompted as a movie set of that place. Landmark names are not in that prompt. A location with people writes `output/plates/<show>/<locationId>/<landmarkId>/plate.png` from that landmark's `appearance`, prompted as one isolated object.

`pnpm run content:assets` has TRELLIS.2 turn each reviewed picture into one mesh. An empty location's mesh is `output/assets/<show>/<locationId>/model.glb`, fitted inside `sizeMeters`, with `location.json` beside it. A landmark's mesh is `output/assets/<show>/<locationId>/<landmarkId>/model.glb`, fitted inside that landmark's `size`, with `landmark.json` beside it. People and held props stay out of those meshes. A missing plate stops the build. If `plate.json` records a different description than the current text, the mesh step stops and asks for `content:plates` again. An unchanged description, polygon limit, and fit reuses the mesh. `--force` and `--refresh` rebuild it. Redrawing a plate deletes the mesh made from the previous picture.

ComfyUI has to be running at `http://127.0.0.1:8188`. `pnpm run content:models` downloads the Qwen still stack, the TRELLIS.2 int8 weights, the DINOv3 vision encoder, the shape VAE, and BiRefNet. The asset pass unloads Qwen before the mesh pass so both are not resident on a 16GB card.

`--offline` does not generate. A missing plate or mesh fails the build. `pnpm run content:assets -- --credits` rewrites `docs/CREDITS.md` from the `location.json` and `landmark.json` files under `output/assets/`.

The plate is a 1024 square. TRELLIS.2 uses the int8 checkpoint, removes the background, and runs the shape cascade at 1024. The stored mesh is untextured. ComfyUI's DecimateMesh node then caps the mesh before it is saved. The default polygon limit is 300,000. `pnpm run content:assets -- --triangles 300000` sets another limit, up to 50,000,000. A mesh saved under a different limit is rebuilt. The mesh is scaled uniformly so it fits inside its box without changing its proportions. An empty location uses `sizeMeters`. A landmark uses its `size`.

Schema space is X right, Y forward, Z up. glTF is Y-up, and forward is -Z. `content-pipeline/scripts/coords.py` is the only converter. The viewer reads Y-up files and does not convert them.

## Models

Qwen-Image-Edit-2511 is Apache-2.0. TRELLIS.2 is MIT. BiRefNet removes the plate background before the mesh pass. Generated meshes are recorded in `docs/CREDITS.md`.

Mixamo is not a source. Its terms restrict automated download. People in previs are clay mannequins.

`pnpm run content:landmarks` runs after an empty location's mesh exists. It shades that mesh from known cameras, asks Florence-2 where each landmark name is, and writes `position` into `shows/<id>/script.json`. It skips a location that has people, because those positions, sizes, and appearances are already in the script. Review images land in `output/landmarks/<show>/<locationId>/`. A landmark that already has a position is left alone unless `--force` is set. A landmark the model does not find stays empty. ComfyUI does not need to be running. If ComfyUI is holding the GPU, Florence-2 can run out of memory.

`pnpm run content:asset-deps` installs trimesh, moderngl, transformers, timm, and einops into the ComfyUI Python. trimesh is what glTF and OBJ files need. moderngl draws clay previs. transformers, timm, and einops load Florence-2 for landmark coordinates. Do not install `content-pipeline/requirements.txt` into that virtualenv.
