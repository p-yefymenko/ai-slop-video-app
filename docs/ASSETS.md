# Stage assets

Landmarks and props describe one object with `appearance`. Example: `One blacksmith anvil of dark forged iron, a single object, no tools and no workshop.` The field is required. Qwen-Image-Edit-2511 draws that object on a plain background. TRELLIS.2 turns the picture into a mesh. A catalog id is not a field, and there is no primitive stand-in. The same description and size share one prefab. A pinned `prefabId` skips generation. If generation fails, `content:assets` stops.

ComfyUI has to be running at `http://127.0.0.1:8188`. `pnpm run content:models` downloads the Qwen still stack, the TRELLIS.2 int8 weights, the DINOv3 vision encoder, the shape VAE, and BiRefNet. The asset pass unloads Qwen before the mesh pass so both are not resident on a 16GB card.

Resolution order:

1. Pinned `prefabId` in `shows/<id>/assets/`, then `library/prefabs/`.
2. A show asset whose `assetHash` matches.
3. A library prefab whose `assetHash` matches.
4. `library/lock.json`, when the entry is not provisional and `--refresh` / `--force` was not passed.
5. Generate the plate, then the mesh. The hash is the description plus `sizeMeters`.

`--offline` does not generate. A mesh that is not already in the library fails the build. After a successful build, library prefabs that no show script pins or describes are deleted, along with their lock entries and raw plates. `pnpm run content:assets -- --credits` rewrites `docs/CREDITS.md`.

The plate is a 1024 square. TRELLIS.2 uses the int8 checkpoint, removes the background, and runs the shape cascade at 1024. The stored prefab is untextured. Anything over 100,000 triangles is welded, then reduced to that budget with quadric edge collapse, before it is fitted to `sizeMeters`. A library mesh simplified with an older decimator is rebuilt from the raw GLB the next time `content:assets` runs. The plate and the raw GLB stay in `library/raw/trellis2/` (gitignored).

Schema space is X right, Y forward, Z up. glTF is Y-up, and forward is -Z. `content-pipeline/scripts/coords.py` is the only converter. The viewer reads Y-up files and does not convert them.

## Models

Qwen-Image-Edit-2511 is Apache-2.0. TRELLIS.2 is MIT. BiRefNet removes the plate background before the mesh pass. Generated meshes are recorded in `docs/CREDITS.md`.

Mixamo is not a source. Its terms restrict automated download. Export a rigged humanoid by hand into `shows/<id>/assets/` if you want to replace the clay mannequin.

`pnpm run content:asset-deps` installs trimesh into the ComfyUI Python. That is what glTF and OBJ files need. Do not install `content-pipeline/requirements.txt` into that virtualenv.
