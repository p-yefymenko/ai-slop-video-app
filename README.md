# Reelshort

Vertical episodes are authored as one show script and rendered on a local GPU. The command menu is `pnpm run`. `PLAN.md` is the contract for setup, deploy, and the content pipeline.

## Content pipeline

`content-pipeline/shows/<id>/script.json` is the authored show. Hand-placed models go in `shows/<id>/assets/`. Everything built from the script — fallback meshes, sets, clay previs, stills, and clips — goes in `content-pipeline/output/<id>/`. `content-pipeline/library/` is the shared prefab cache.

```
pnpm run content:validate
pnpm run content:assets
pnpm run content:previs
pnpm run content:render
pnpm run view
```

`content:render` builds prefabs, the clay blockout, the Qwen stills, and the LTX clips. `pnpm run view` opens the Y-up stage viewer at `http://127.0.0.1:5174`. Catalog terms, licenses, and the credit file are in `docs/ASSETS.md` and `docs/CREDITS.md`.
