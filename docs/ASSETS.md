# Location meshes

Each location becomes one picture, then one mesh, for that show only.

`pnpm run content:plates` has Qwen-Image-Edit-2511 draw the location text as an open floor and stop. The picture is `output/plates/<show>/<locationId>/plate.png`. Landmark names are not in that prompt. A landmark is a coordinate you fill in after the mesh exists, because the picture is not a measured floor plan and the mesh does not label its parts.

`pnpm run content:assets` has TRELLIS.2 turn that reviewed picture into one mesh, fitted inside the location's `sizeMeters`. The mesh is `output/assets/<show>/<locationId>/model.glb`, with `location.json` beside it. People and held props stay out of that mesh. A missing plate stops the build. If `plate.json` records a different description than the current location text, the mesh step stops and asks for `content:plates` again. An unchanged description, polygon limit, and fit reuses the mesh. `--force` and `--refresh` rebuild it. Redrawing a plate deletes the mesh made from the previous picture.

ComfyUI has to be running at `http://127.0.0.1:8188`. `pnpm run content:models` downloads the Qwen still stack, the TRELLIS.2 int8 weights, the DINOv3 vision encoder, the shape VAE, and BiRefNet. The asset pass unloads Qwen before the mesh pass so both are not resident on a 16GB card.

`--offline` does not generate. A missing plate or mesh fails the build. `pnpm run content:assets -- --credits` rewrites `docs/CREDITS.md` from the `location.json` files under `output/assets/`.

The plate is a 1024 square. TRELLIS.2 uses the int8 checkpoint, removes the background, and runs the shape cascade at 1024. The stored mesh is untextured. ComfyUI's DecimateMesh node then caps the mesh before it is saved. The default polygon limit is 300,000. `pnpm run content:assets -- --triangles 300000` sets another limit, up to 50,000,000. A mesh saved under a different limit is rebuilt. The mesh is scaled uniformly so it fits inside the location `sizeMeters` without changing its proportions.

Schema space is X right, Y forward, Z up. glTF is Y-up, and forward is -Z. `content-pipeline/scripts/coords.py` is the only converter. The viewer reads Y-up files and does not convert them.

## Models

Qwen-Image-Edit-2511 is Apache-2.0. TRELLIS.2 is MIT. BiRefNet removes the plate background before the mesh pass. Generated meshes are recorded in `docs/CREDITS.md`.

Mixamo is not a source. Its terms restrict automated download. People in previs are clay mannequins.

`pnpm run content:asset-deps` installs trimesh and moderngl into the ComfyUI Python. trimesh is what glTF and OBJ files need. moderngl draws clay previs. Do not install `content-pipeline/requirements.txt` into that virtualenv.
