# Reelshort

Vertical episodes are authored as one `ShowScript` JSON file and rendered on a local GPU. Every command lives in the root `package.json`. Run `pnpm run` with no arguments to list them. Do not run raw `wrangler` or `python` commands.

`PLAN.md` is the setup and deploy contract. This file is daily usage.

## Render an episode

`content:render` still does the full pass. It runs four stages in order and forwards the same flags to each. It does **not** pause for review between stills and clips.

| Stage | Command | Writes |
| --- | --- | --- |
| 1. Prefabs and sets | `content:assets` | `output/<show>/sets/`, `assets/fallback/`, `library/lock.json` |
| 2. Clay previs | `content:previs` | `output/<show>/<episode>/01_previs/` — `blockout.mp4`, `start.png`, `shot.json`, guides |
| 3. Qwen stills | `content:frames` | `output/<show>/characters/` and `02_postvis/stills/` |
| 4. LTX clips | `content:generate` | `03_postvis/clips/` and `04_edit/episode.mp4` |

Clay previs is **only** `content:previs` (or that second stage inside `content:render`). `content:frames` reads those clay frames; it does not create them. `content:assets` and `content:previs` do not need ComfyUI.

Leave ComfyUI running in another terminal before the GPU stages:

```bash
pnpm run content:comfy
```

Then either:

```bash
pnpm run content:validate
pnpm run content:render -- --show the-iron-bride --episode 1
```

or the same four stages by hand, so you can inspect the clay blockout and stills:

```bash
pnpm run content:validate
pnpm run content:assets -- --show the-iron-bride
pnpm run content:previs -- --show the-iron-bride --episode 1
pnpm run view
pnpm run content:frames -- --show the-iron-bride --episode 1
pnpm run content:generate -- --show the-iron-bride --episode 1
```

Use `pnpm run view`, not `pnpm view` (that is pnpm’s package lookup). The viewer is `http://127.0.0.1:5174`.

Useful flags (all stages accept them; assets ignores `--episode` / `--scene` / `--seed` for which prefabs it builds):

| Flag | Meaning |
| --- | --- |
| `--show <id>` | One show |
| `--episode N` | One episode |
| `--scene N` | One scene (needs `--episode`) |
| `--force` | Rebuild that selected scene (needs `--scene`) |
| `--offline` | Assets only: locks and primitives, no catalog download |

Existing files are skipped unless `--force` is set on a selected scene. After a structural script rewrite, archive first:

```bash
pnpm run content:archive -- the-iron-bride
```

## Where files live

| Path | What it is |
| --- | --- |
| `content-pipeline/shows/<id>/script.json` | Authored show. Folder name must match `id`. |
| `content-pipeline/shows/<id>/assets/` | Hand-placed GLBs only |
| `content-pipeline/library/` | Shared prefab cache (`raw/` is gitignored) |
| `content-pipeline/output/<id>/` | Everything generated from the script |

Per episode, generated stages sort as:

```text
output/<show>/<episode>/
  01_previs/scene_XX/{blockout.mp4, start.png, shot.json, guides/}
  02_postvis/stills/scene_XX_start.png
  03_postvis/clips/scene_XX.mp4
  04_edit/episode.mp4
```

Character identity PNGs stay at `output/<show>/characters/`, not per episode. Sets live at `output/<show>/sets/<locationId>/`.

Write the script, then `pnpm run content:validate`. Field semantics are in `packages/shared/src/script.ts`. Screenwriting guidance is `.cursor/rules/Short-reel-scripts-writer.mdc`.

## Viewer

`pnpm run view` lists prefabs, sets, and shots. Deep links: `/prefab/<id>`, `/set/<show>/<location>`, `/shot/<show>/<episode>/<scene>`.

A landmark or prop that should be a real model sets `assetId` to `source:id`, for example `polyhaven:coast_rocks_05`. Omit it to keep the primitive. Meshes used for previs are capped at 25,000 triangles.

God camera (OrbitControls):

- Left-drag: orbit
- Right-drag, middle-drag, or Ctrl/Cmd + left-drag: pan
- Scroll: zoom

**Shot camera** is the authored lens; orbit/pan/zoom are off. Play / Pause and the timeline slider only change time.

## GPU setup (once)

Needed for `content:frames` and `content:generate`. Clay previs does not need it.

1. `pnpm run content:setup-comfy`
2. `pnpm run content:models`
3. Put `LTXV_API_KEY` in `content-pipeline/.env` (copy from `content-pipeline/.env.example`)
4. Leave `pnpm run content:comfy` running at `http://127.0.0.1:8188`
5. In that UI, Load `qwen_image_edit.json`, `qwen_image_edit_spatial.json`, and `ltx_gemma_api.json`, and fix missing nodes or files

`pnpm run content:asset-deps` installs trimesh into that same Python so downloaded glTF/OBJ files can be read. Optional catalog keys: `SKETCHFAB_TOKEN`, `SMITHSONIAN_API_KEY`.

Catalog terms and licenses: `docs/ASSETS.md`. Attribution file: `docs/CREDITS.md` (`pnpm run content:assets -- --credits`).
