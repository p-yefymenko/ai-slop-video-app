# Stage assets

Each location becomes one mesh. Qwen-Image-Edit-2511 draws the whole place, including its ground, from the location text and the landmark notes. TRELLIS.2 turns that picture into a single mesh fitted inside the location's `sizeMeters`. People and held props stay out of that mesh. A pinned location `prefabId` skips generation. If generation fails, `content:assets` stops.

ComfyUI has to be running at `http://127.0.0.1:8188`. `pnpm run content:models` downloads the Qwen still stack, the TRELLIS.2 int8 weights, the DINOv3 vision encoder, the shape VAE, and BiRefNet. The asset pass unloads Qwen before the mesh pass so both are not resident on a 16GB card.

Resolution order:

1. Pinned `prefabId` in `shows/<id>/assets/`, then `library/prefabs/`.
2. A show asset whose `assetHash` matches.
3. A library prefab whose `assetHash` matches.
4. `library/lock.json`, when the entry is not provisional and `--refresh` / `--force` was not passed.
5. Generate the plate, then the mesh. The hash is the place description plus the location `sizeMeters`.

`--offline` does not generate. A mesh that is not already in the library fails the build. After a successful build, library prefabs that no show script pins or describes are deleted, along with their lock entries and raw plates. `pnpm run content:assets -- --credits` rewrites `docs/CREDITS.md`.

The plate is a 1024 square. TRELLIS.2 uses the int8 checkpoint, removes the background, and runs the shape cascade at 1024. The stored prefab is untextured. ComfyUI's DecimateMesh node then caps the mesh at 10,000,000 triangles before it is saved. The mesh is scaled uniformly so it fits inside the location `sizeMeters` without changing its proportions. A library mesh that was stretched onto that box is refit from the raw GLB the next time `content:assets` runs. A library mesh simplified with an older decimator is rebuilt through the same graph. The plate and the raw GLB stay in `library/raw/trellis2/` (gitignored).

Schema space is X right, Y forward, Z up. glTF is Y-up, and forward is -Z. `content-pipeline/scripts/coords.py` is the only converter. The viewer reads Y-up files and does not convert them.

## Models

Qwen-Image-Edit-2511 is Apache-2.0. TRELLIS.2 is MIT. BiRefNet removes the plate background before the mesh pass. Generated meshes are recorded in `docs/CREDITS.md`.

Mixamo is not a source. Its terms restrict automated download. Export a rigged humanoid by hand into `shows/<id>/assets/` if you want to replace the clay mannequin.

`pnpm run content:asset-deps` installs trimesh into the ComfyUI Python. That is what glTF and OBJ files need. Do not install `content-pipeline/requirements.txt` into that virtualenv.
