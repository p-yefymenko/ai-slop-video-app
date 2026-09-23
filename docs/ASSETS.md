# Stage assets

Landmarks and props name an exact Sketchfab model with `assetId`, written as `sketchfab:<uid>`. Example: `sketchfab:e63f1154ee0b41f8a797db683526142a`. The uid is the 32-character id at the end of the model page URL. The field is required. A search query is not a field, and there is no primitive stand-in. The same id and size share one prefab. A pinned `prefabId` skips the download. If the model cannot be fetched, `content:assets` stops.

Resolution order:

1. Pinned `prefabId` in `shows/<id>/assets/`, then `library/prefabs/`.
2. A show asset whose `assetHash` matches.
3. A library prefab whose `assetHash` matches.
4. `library/lock.json`, when the entry is not provisional and `--refresh` / `--force` was not passed.
5. A lookup of that exact id. No keyword ranking.

`--offline` does not download. A model that is not already in the library fails the build. After a successful build, library prefabs that no show script pins or names are deleted, along with their lock entries and raw downloads. `pnpm run content:assets -- --credits` rewrites `docs/CREDITS.md`.

Only CC0 and CC-BY are kept. NC, ND, and SA are dropped. Downloaded meshes are recentered on the base, scaled to `sizeMeters`, and stored as a texture-free GLB. Anything over 100,000 triangles is welded, then reduced to that budget with quadric edge collapse, before it is fitted to `sizeMeters`. A library mesh simplified to an older budget is rebuilt from the raw download the next time `content:assets` runs. The original download stays in `library/raw/` (gitignored).

Schema space is X right, Y forward, Z up. glTF is Y-up, and forward is -Z. `content-pipeline/scripts/coords.py` is the only converter. The viewer reads Y-up files and does not convert them.

## Catalog

Sketchfab Data API v3 is the only catalog. Set `SKETCHFAB_TOKEN` in `content-pipeline/.env`. The id is `sketchfab:` plus the model uid. Lookup is `GET /v3/models/{uid}`. Download is `GET /v3/models/{uid}/download`, which returns a short-lived archive URL. The client prefers the glb entry, then glTF, then the source archive. Calls are spaced about 0.25s apart. CC-BY models are recorded in `docs/CREDITS.md`.

Sketchfab's label `CC Attribution` is CC-BY. A model that is not downloadable is refused.

Mixamo is not a source. Its terms restrict automated download. Export a rigged humanoid by hand into `shows/<id>/assets/` if you want to replace the clay mannequin. Text-to-3D is not registered.

`pnpm run content:fetch-asset -- <url>` accepts a Sketchfab model page after the license check. It refuses Mixamo and any other host.

`pnpm run content:asset-deps` installs trimesh into the ComfyUI Python. That is what downloaded glTF and OBJ files need. Do not install `content-pipeline/requirements.txt` into that virtualenv.
