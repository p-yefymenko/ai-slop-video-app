# Reelshort

Vertical episodes are authored as one `ShowScript` JSON file and rendered on a local GPU. Every command lives in the root `package.json`. Run `pnpm run` with no arguments to list them. Do not run raw `wrangler` or `python` commands.

`PLAN.md` is the setup and deploy contract. This file is daily usage.

## Render an episode

`content:render` still does the full pass. It runs four stages in order and forwards the same flags to each. It does **not** pause for review between stills and clips.

| Stage | Command | Writes |
| --- | --- | --- |
| Location plates | `content:plates` | `output/plates/<show>/<location>/plate.png` |
| Location meshes | `content:assets` | `output/assets/<show>/<location>/model.glb` |
| Landmark coordinates | `content:landmarks` | writes `position` into `shows/<id>/script.json`; review images in `output/landmarks/<show>/<location>/` |
| Clay previs | `content:previs` | `output/previs/<show>/<episode>/` — per-scene `blockout.mp4`, plus one episode `blockout.mp4` |
| Qwen stills | `content:frames` | `output/frames/<show>/characters/` and `output/frames/<show>/<episode>/` |
| LTX clips | `content:generate` | `output/generate/<show>/<episode>/` |

Clay previs is **only** `content:previs` (or that stage inside `content:render`). `content:frames` reads those clay frames; it does not create them. `content:previs` does not need ComfyUI. `content:plates` and `content:assets` do. People and cameras stay on the script marks. An empty location is one mesh. A location with people is one mesh per landmark, fitted to the size in the script. The same appearance and size are built once and placed at each position.

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
pnpm run content:plates -- --show the-iron-bride
pnpm run content:assets -- --show the-iron-bride
pnpm run content:landmarks -- --show the-iron-bride
pnpm run content:previs -- --show the-iron-bride --episode 1
pnpm run view -- --show the-iron-bride
pnpm run content:frames -- --show the-iron-bride --episode 1
pnpm run content:generate -- --show the-iron-bride --episode 1
```

Use `pnpm run view`, not `pnpm view` (that is pnpm’s package lookup). The viewer is `http://127.0.0.1:5174` and needs `--show`.

Useful flags (all stages accept them; plates and assets ignore `--episode` / `--scene` / `--seed`):

| Flag | Meaning |
| --- | --- |
| `--show <id>` | One show |
| `--episode N` | One episode |
| `--scene N` | One scene (needs `--episode`) |
| `--force` | Rebuild that selected scene (needs `--scene`) |
| `--offline` | Plates and assets: do not generate. A missing file fails the build |

Existing files are skipped unless `--force` is set on a selected scene. After a structural script rewrite, archive first:

```bash
pnpm run content:archive -- the-iron-bride
```

## Where files live

| Path | What it is |
| --- | --- |
| `content-pipeline/shows/<id>/script.json` | Authored show. Folder name must match `id`. |
| `content-pipeline/output/plates/<id>/` | Location pictures |
| `content-pipeline/output/assets/<id>/` | Location meshes |
| `content-pipeline/output/landmarks/<id>/` | Shaded views used to place landmark coordinates |
| `content-pipeline/output/previs/<id>/` | Clay scenes |
| `content-pipeline/output/frames/<id>/` | Character portraits and scene stills |
| `content-pipeline/output/generate/<id>/` | Scene clips and the episode cut |

Per episode, generated stages sort as:

```text
output/previs/<show>/<episode>/blockout.mp4
output/previs/<show>/<episode>/scene_XX/{blockout.mp4, start.png, scene.json, guides/}
output/frames/<show>/characters/<id>.png
output/frames/<show>/<episode>/scene_XX_start.png
output/generate/<show>/<episode>/scene_XX.mp4
output/generate/<show>/<episode>/episode.mp4
```

Location plates live at `output/plates/<show>/<locationId>/plate.png`. Location meshes live at `output/assets/<show>/<locationId>/model.glb`.

Write the script, then `pnpm run content:validate`. Field semantics are in `packages/shared/src/script.ts`. Screenwriting guidance is `.cursor/rules/Short-reel-scripts-writer.mdc`.

## Viewer

`pnpm run view -- --show <show-id>` lists that show’s locations and scenes. Deep links: `/location/<locationId>`, `/scene/<episode>/<scene>`.

`content:plates` has Qwen draw and stop, so the picture can be reviewed. An empty location is one picture of the place. A location with people is one picture per landmark. The same appearance and size are drawn once. `content:assets` then has TRELLIS.2 build one mesh from each reviewed picture, fitted uniformly into the size in the script. The same appearance and size are meshed once. The default polygon limit is 300,000. `pnpm run content:assets -- --triangles 300000` sets another limit. `content:landmarks` writes positions only for an empty location's mesh. A location with people already has those positions. It is not part of `content:render`. A position already in the script is kept unless `--force` is set.

God camera (OrbitControls):

- Left-drag: orbit
- Right-drag, middle-drag, or Ctrl/Cmd + left-drag: pan
- Scroll: zoom

**Scene camera** is the authored lens; orbit/pan/zoom are off. Play / Pause and the timeline slider only change time.

## GPU setup (once)

Needed for `content:frames` and `content:generate`. Clay previs does not need it.

1. `pnpm run content:setup-comfy`
2. `pnpm run content:models`
3. Put `LTXV_API_KEY` in `content-pipeline/.env` (copy from `content-pipeline/.env.example`)
4. Leave `pnpm run content:comfy` running at `http://127.0.0.1:8188`
5. In that UI, Load `qwen_image_edit.json`, `qwen_image_edit_spatial.json`, and `ltx_gemma_api.json`, and fix missing nodes or files

`pnpm run content:asset-deps` installs trimesh, moderngl, transformers, timm, and einops into that same Python. trimesh reads glTF and OBJ files. moderngl draws the clay previs on the GPU. transformers, timm, and einops load Florence-2 for `content:landmarks`. `content:plates` needs ComfyUI running and the Qwen still stack. `content:assets` needs ComfyUI running and the TRELLIS.2 weights from `pnpm run content:models`. `content:landmarks` does not need ComfyUI running. If ComfyUI is holding the GPU, Florence-2 can run out of memory.

Catalog terms and licenses: `docs/ASSETS.md`. Attribution file: `docs/CREDITS.md` (`pnpm run content:assets -- --credits`).
