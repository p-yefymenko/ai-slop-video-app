# Stage assets

Landmarks and props name an exact catalog model with `assetId`, written as `source:id`. Example: `polyhaven:coast_rocks_05`. The field is required. A search query is not a field, and there is no primitive stand-in. The same source, id, and size share one prefab. A pinned `prefabId` skips the download. If the model cannot be fetched, `content:assets` stops.

Sources: `polyhaven`, `sketchfab`, `smithsonian`, `kenney`, `quaternius`.

Resolution order:

1. Pinned `prefabId` in `shows/<id>/assets/`, then `library/prefabs/`.
2. A show asset whose `assetHash` matches.
3. A library prefab whose `assetHash` matches.
4. `library/lock.json`, when the entry is not provisional and `--refresh` / `--force` was not passed.
5. A lookup of that exact id. No keyword ranking.

`--offline` does not download. A model that is not already in the library fails the build. `pnpm run content:assets -- --credits` rewrites `docs/CREDITS.md`.

Only CC0 and CC-BY are kept. NC, ND, and SA are dropped. Downloaded meshes are recentered on the base, scaled to `sizeMeters`, and stored as a texture-free GLB. Anything over 25,000 triangles is reduced to that budget before it is written, including a prefab already in the library the next time it is read. The original download stays in `library/raw/` (gitignored).

Schema space is X right, Y forward, Z up. glTF is Y-up, and forward is -Z. `content-pipeline/scripts/coords.py` is the only converter. The viewer reads Y-up files and does not convert them.

## Catalogs

Poly Haven (`https://api.polyhaven.com`) serves CC0 models. An `assetId` of `polyhaven:coast_rocks_05` is the page slug from `https://polyhaven.com/a/coast_rocks_05`. The client loads the model list and takes that id. File metadata comes from `GET /files/{id}`, and the lowest glTF resolution is downloaded. Dimensions are millimeters when present. The live API asks for a unique User-Agent and a visible “Powered by Poly Haven” credit. The assets themselves do not require attribution. No API key.

Sketchfab Data API v3 loads one model by uid when `SKETCHFAB_TOKEN` is set in `content-pipeline/.env`. The id is `sketchfab:` plus that uid. The download endpoint currently expects an end-user OAuth login unless Sketchfab grants an exception. The pipeline tries the token, skips the model when the API refuses, and records attribution for CC-BY. Short-lived archive URLs are not cached. Calls are spaced about 0.25s apart.

Smithsonian Open Access is CC0 and needs `SMITHSONIAN_API_KEY` from api.data.gov. The id is `smithsonian:` plus the Open Access record id. Do not use `DEMO_KEY`. With no key, the lookup returns nothing.

Kenney and Quaternius packs are CC0 and have no per-item API. Drop files you already have into `library/raw/kenney` or `library/raw/quaternius`. The id is the source plus the file stem or relative path, for example `kenney:wooden-chair.glb`. The pipeline does not download packs. A file dropped in those folders is treated as CC0, so do not put another license there.

Objaverse stays off unless `OBJAVERSE_ENABLE=1`. The dataset as a whole is ODC-By and includes NC and SA objects. Only an individual CC0 or CC-BY record would be eligible. The adapter does not download the annotation index.

Mixamo is not a source. Its terms restrict automated download. Export a rigged humanoid by hand into `shows/<id>/assets/` if you want to replace the clay mannequin. Text-to-3D is not registered.

`pnpm run content:fetch-asset -- <url>` accepts a Poly Haven asset page, a Sketchfab model page after the license check, or a direct Smithsonian `.glb` / `.gltf` URL. It refuses Mixamo and any other host.

`pnpm run content:asset-deps` installs trimesh into the ComfyUI Python. That is what downloaded glTF and OBJ files need. Do not install `content-pipeline/requirements.txt` into that virtualenv.
