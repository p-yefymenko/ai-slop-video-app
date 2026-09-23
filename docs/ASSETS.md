# Stage assets

Landmark and prop `need` fields in `shows/<id>/script.json` are resolved into GLB prefabs. The same query, tags, style, and size share one prefab. Editing the query changes the hash and searches again. A pinned `prefabId` skips the search. If nothing acceptable is found, the pipeline builds a primitive and keeps going.

Resolution order:

1. Pinned `prefabId` in `shows/<id>/assets/`, then `library/prefabs/`, then `output/<id>/assets/fallback/`.
2. A show asset whose `needHash` matches.
3. A library prefab whose `needHash` matches.
4. `library/lock.json`, when the entry is not provisional and `--refresh` / `--force` was not passed.
5. Catalog search.
6. A primitive in `output/<id>/assets/fallback/`.

`--offline` never searches. A provisional lock is reused offline and retried when the machine is online. `--review` keeps the top candidates under `output/<id>/asset-review/<hash>/candidates.json`. `pnpm run view` can lock one of them. `pnpm run content:assets -- --credits` rewrites `docs/CREDITS.md`.

Only CC0 and CC-BY are kept. NC, ND, and SA are dropped. Downloaded meshes are recentered on the base, scaled to `sizeMeters`, and stored as a texture-free GLB. The original download stays in `library/raw/` (gitignored).

Schema space is X right, Y forward, Z up. glTF is Y-up, and forward is -Z. `content-pipeline/scripts/coords.py` is the only converter. The viewer reads Y-up files and does not convert them.

## Catalogs

Poly Haven (`https://api.polyhaven.com`) serves CC0 models. The asset list has no keyword parameter, so the client downloads the model list once and filters it. File metadata comes from `GET /files/{id}`. Dimensions are millimeters when present. The live API asks for a unique User-Agent and a visible “Powered by Poly Haven” credit. The assets themselves do not require attribution. No API key.

Sketchfab Data API v3 searches downloadable models with `licenses=cc0,by` when `SKETCHFAB_TOKEN` is set in `content-pipeline/.env`. The download endpoint currently expects an end-user OAuth login unless Sketchfab grants an exception. The pipeline tries the token, skips the model when the API refuses, and records attribution for CC-BY. Short-lived archive URLs are not cached. Calls are spaced about 0.25s apart. There is no published numeric rate limit; each need is one search plus a few downloads.

Smithsonian Open Access is CC0 and needs `SMITHSONIAN_API_KEY` from api.data.gov. Search uses `q` plus `online_media_type:"3D Models"`. Do not use `DEMO_KEY`. With no key, the source returns nothing.

Kenney and Quaternius packs are CC0 and have no per-item API. Drop files you already have into `library/raw/kenney` or `library/raw/quaternius`. The pipeline indexes `.glb`, `.gltf`, and `.obj` there. It does not download packs. A file dropped in those folders is treated as CC0, so do not put another license there.

Objaverse stays off unless `OBJAVERSE_ENABLE=1`. The dataset as a whole is ODC-By and includes NC and SA objects. Only an individual CC0 or CC-BY record would be eligible. The adapter does not download the annotation index.

Mixamo is not a source. Its terms restrict automated download. Export a rigged humanoid by hand into `shows/<id>/assets/` if you want to replace the clay mannequin. Text-to-3D is not registered.

`pnpm run content:fetch-asset -- <url>` accepts a Poly Haven asset page, a Sketchfab model page after the license check, or a direct Smithsonian `.glb` / `.gltf` URL. It refuses Mixamo and any other host.

`pnpm run content:asset-deps` installs trimesh into the ComfyUI Python. That is what downloaded glTF and OBJ files need. Do not install `content-pipeline/requirements.txt` into that virtualenv.
