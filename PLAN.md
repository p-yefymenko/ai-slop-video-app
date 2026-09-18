# ReelShort Clone — Build Plan for Cursor

## Goal

Build a vertical short-drama video app (ReelShort clone) as a single monorepo. Android launch first via native Play Store app (React Native/Expo), architected so iOS can be added later with minimal extra work. Content is AI-generated locally (LTX) and uploaded to cloud storage; users unlock episodes via coins/subscription.

## 🧍 Human-only steps — in order, start to Play Store launch

> The agent should never check these off itself — only the human does, after actually completing the action outside Cursor.

1. [x] Register a Cloudflare account (free)
2. [x] Register the $25 Google Play Developer (personal) account — do this early; identity verification can take several days, and this account is what your 14-day closed-testing clock later runs against
3. [x] Run `pnpm run login:cloudflare` (one-time browser auth — the only Cloudflare step that can't be scripted)
4. [x] Run `pnpm run setup` to create the R2 bucket and D1 database, then paste the printed IDs into `server/wrangler.toml`
5. [ ] Obtain Google Play Billing service account credentials (Play Console → API access) and add them to `.env`/secrets
6. [x] Get an LTX API key: sign up at the LTX Developer Console (docs.ltx.video) and generate a key there. Text encoding via this API is free — you're only using it to offload the text-encoder step, not for paid video generation, since video generation itself runs locally on your GPU. Add it to `content-pipeline/.env` as `LTXV_API_KEY` (the agent's ComfyUI workflow config should reference this env var, not a hardcoded key) — this is what the `GemmaAPITextEncode` node in the LTX workflow uses to authenticate.
7. [ ] **Install ComfyUI + models locally on the GPU machine** — run `pnpm run content:setup-comfy` once (clones ComfyUI and the LTX/GGUF/VHS custom nodes into `content-pipeline/.comfyui` and installs CUDA PyTorch). If you already ran setup before CUDA torch existed, run `pnpm run content:comfy-torch`. Then `pnpm run content:models` (downloads the ~14GB LTX-2.3 distilled-1.1 Q4_K_M GGUF + matching video VAE + audio VAE + Gemma API stub, and the Qwen-Image-Edit-2511 still stack: ~13GB Q4_K_M GGUF, 9.4GB Qwen2.5-VL encoder, VAE, and 4-step Lightning LoRA). Leave `pnpm run content:comfy` running so the API is at `http://127.0.0.1:8188`. Open that URL, Load `qwen_image_edit.json` and `ltx_gemma_api.json`, and fix any missing-node / missing-file errors before generating.
8. [ ] **Create character refs, then write or source episode scripts.** For each named character, generate one reference image and write a fixed description block that includes wardrobe (see **Character consistency**). Put rooms in the episode JSON `locations` object (`promptBlock` + `preserve`, keyed by `locationId`) — text only, no location photos. `promptBlock` is the set **and** where the character stands in it (feet, scale vs furniture, camera). Every scene has `imagePrompt`: Qwen first builds a location+character plate, then edits that plate into the scene still so props and pose are already in frame 0. Prefer one scene per visit (`Cut.` for 2–4 shots in the same room). Follow **Episode script authoring**. Original writing, translated/licensed material reworked into scene prompts, or AI-assisted drafting are all fair game. Paste that authoring block plus the JSON schema into a chat and save the output with no reformatting. Drop files into `content-pipeline/characters/` and `content-pipeline/scripts_input/`.
9. [ ] Run `pnpm run content:frames`. Review `plate_*.png` then `scene_*_start.png` in `content-pipeline/output/`. Keep, replace, or delete any file you do not like (delete a plate and its scene stills to regenerate them). Then run `pnpm run content:generate` and `pnpm run content:upload` on your own machine (the RTX 5070 Ti). ComfyUI must already be running from the previous step.
10. [ ] (Optional) Buy a domain and point it at the deployed Worker — needed for the privacy policy URL Play Store requires
11. [ ] Trigger the production build via `eas build` (Expo's build service)
12. [ ] Upload the build to a Closed Testing track in Play Console
13. [ ] Recruit 12 testers and wait 14 continuous days with them opted in
14. [ ] Apply for production access once the closed test requirements are met
15. [ ] Publish to production

Everything not listed above — repo setup, code, schema, screens, pipeline scripts, deployment config, and even the R2/D1 creation commands themselves — should be built/scripted by the AI agent.

### Expected costs at low usage

- **Workers + D1: $0/month** on the free tier (100K requests/day, 5M D1 row reads/day, 100K row writes/day — comfortably covers early testing and thousands of daily active users before any payment is needed)
- **R2: $0/month** while the video library stays under 10GB (roughly the first 500-1000+ short episodes depending on compression), then $0.015/GB-month beyond that — egress is always free regardless of scale
- **The only guaranteed cost pre-launch is the one-time $25 Google Play Developer fee.**
- **Post-launch, not part of the sequence above:** once real traffic approaches the free-tier limits, upgrade to Workers Paid ($5/mo) in the Cloudflare dashboard — this also raises D1's limits since it's the same subscription. There's no need to do this preemptively; the app will simply keep working on the free tier until you're actually close to the ceiling.

---



## Command Interface — the ONE place to look for what you can run

> Rule for the agent: every command a human needs to run — setup, dev, migrations, deploy, everything — must be added as a named script in the **root** `package.json`. Never tell the human to run a raw `wrangler ...` or `python ...` command directly; wrap it in a script here first. This file is the single source of truth for "what can I run," so the human never has to go back to `PLAN.md` to find a command.

Running `pnpm run` with no arguments lists every available script — that's the menu. Keep names short, grouped by prefix (`setup:`, `dev:`, `db:`, `content:`, `deploy:`), and keep this list in the root `package.json` in sync with what's below as new scripts are added:

```json
{
  "scripts": {
    "login:cloudflare": "pnpm --filter server exec wrangler login",
    "setup": "bash scripts/setup-cloudflare.sh",
    "secret:admin": "pnpm --filter server exec wrangler secret put ADMIN_SECRET",
    "secret:media": "pnpm --filter server exec wrangler secret put MEDIA_SIGNING_SECRET",
    "secret:play": "pnpm --filter server exec wrangler secret put GOOGLE_PLAY_SERVICE_ACCOUNT",
    "dev:mobile": "pnpm --filter mobile start",
    "dev:server": "pnpm --filter server exec wrangler dev",
    "db:generate": "pnpm --filter server exec drizzle-kit generate",
    "db:migrate:local": "pnpm --filter server exec wrangler d1 migrations apply reelshort-db --local",
    "db:migrate:remote": "pnpm --filter server exec wrangler d1 migrations apply reelshort-db --remote",
    "db:seed": "pnpm --filter server exec tsx src/db/seed.ts",
    "content:setup-comfy": "node scripts/setup-comfyui.cjs",
    "content:comfy-torch": "node scripts/install-comfy-torch.cjs",
    "content:models": "node scripts/download-ltx-models.cjs",
    "content:comfy": "node scripts/start-comfyui.cjs",
    "content:frames": "node scripts/run-python.cjs content-pipeline/scripts/generate_batch.py --stage frames",
    "content:generate": "node scripts/run-python.cjs content-pipeline/scripts/generate_batch.py --stage video",
    "content:render": "pnpm run content:frames && pnpm run content:generate",
    "content:upload": "node scripts/run-python.cjs content-pipeline/scripts/upload_to_r2.py",
    "deploy": "bash scripts/deploy.sh",
    "build:android": "pnpm --filter mobile exec eas build --platform android --profile production"
  }
}
```

Whenever the agent adds a new manual step anywhere in the build process (a migration, a one-off admin action, a new pipeline step), it should add a corresponding `pnpm run <name>` script here rather than leaving it as a raw shell command described only in prose.

## Deployment (Cloudflare Workers)

No droplet, no Docker, no SSH — deployment is a single command (`pnpm run deploy`) via Wrangler (Cloudflare's Workers CLI) under the hood. Resource creation (R2 bucket, D1 database) is also CLI-driven, not dashboard clicking — wrapped in `pnpm run setup`:

`scripts/setup-cloudflare.sh` (agent should write this; human runs it once via `pnpm run setup`):

```bash
#!/bin/bash
# One-time setup. Requires `wrangler login` to have been run already.
wrangler r2 bucket create reelshort-videos
wrangler d1 create reelshort-db
# ^ copy the printed bucket_name and database_id into server/wrangler.toml
```

`server/wrangler.toml` should bind the Worker to its D1 database and R2 bucket:

```toml
name = "reelshort-api"
main = "src/index.ts"
compatibility_date = "2026-01-01"

[[d1_databases]]
binding = "DB"
database_name = "reelshort-db"
database_id = "<created via `pnpm run setup`>"

[[r2_buckets]]
binding = "VIDEO_BUCKET"
bucket_name = "<the R2 bucket created by `pnpm run setup`>"
```

- Local dev: `pnpm run dev:server` runs the Worker locally with a local D1 emulation (SQLite file) — no need for docker-compose or a local Postgres instance at all.
- Deploy: `pnpm run deploy` pushes the Worker live. This is the entire deploy process — no server to provision, no reverse proxy, no Certbot. Cloudflare handles HTTPS automatically on the `*.workers.dev` subdomain or a custom domain.
- Migrations: `pnpm run db:generate` to create migration files after schema changes, then `pnpm run db:migrate:local` / `pnpm run db:migrate:remote` to apply them.
- Custom domain (optional, once the human buys one): add a Worker route in `wrangler.toml` to map `api.yourdomain.com` to this Worker.



## Final Tech Stack

- **Mobile app:** React Native + Expo (managed workflow), TypeScript
- **Backend:** Cloudflare Workers (TypeScript) + Hono (lightweight router for Workers, since NestJS doesn't run on the Workers runtime) — serverless, scale-to-zero, no server to manage
- **Database:** Cloudflare D1 (serverless SQLite) with Drizzle ORM (Prisma doesn't support D1's runtime; Drizzle is the standard ORM for D1)
- **Storage/CDN:** Cloudflare R2 (video files, thumbnails) — zero egress cost
- **Hosting:** None to manage — Workers, D1, and R2 are all serverless/managed by Cloudflare on the same account. No droplet, no Docker, no SSH.
- **Payments:** Google Play Billing (server-side receipt verification inside a Worker)
- **Content generation (offline, not part of the live app):** ComfyUI running locally on the GPU machine. Start stills come from **Qwen-Image-Edit-2511** (Q4_K_M GGUF + Lightning 4-step LoRA): first a location+character plate from the character PNG, then a per-scene edit of that plate from `imagePrompt` so pose and props are already in frame 0. Those stills are then animated with **LTX-2.3 distilled-1.1** (Q4_K_M GGUF) image-to-video, with Gemma API used for LTX text-encoder conditioning to stay within 16GB VRAM. Qwen and LTX are not meant to stay loaded together; `content:frames` and `content:generate` unload idle models between stages. Output MP4s are uploaded to R2 via a script, not generated at runtime.
- **Monorepo tooling:** pnpm workspaces

> **Cost model:** Workers + D1 usage is free up to 100K requests/day and 5M D1 row reads/day; R2 is free up to 10GB storage with egress always free. Realistically $0/month until real user traction, then a flat $5/month (Workers Paid, which also raises D1 limits) covers a large jump in headroom. See cost breakdown in the Human-only steps section above.



## Repo Structure

```
reelshort-clone/
├── apps/
│   └── mobile/                  # Expo RN app
│       ├── app/                 # screens (expo-router)
│       ├── components/
│       ├── stores/              # state management (zustand)
│       └── api/                 # typed API client, generated from shared types
├── server/                      # Cloudflare Worker backend (Hono)
│   ├── src/
│   │   ├── routes/
│   │   │   ├── series.ts
│   │   │   ├── episodes.ts
│   │   │   ├── purchases.ts       # coin purchases, Play Billing verification
│   │   │   ├── wallet.ts           # coin balance logic
│   │   │   └── users.ts
│   │   ├── db/
│   │   │   └── schema.ts          # Drizzle schema (D1)
│   │   ├── storage.ts              # R2 upload/signed URL helpers
│   │   └── index.ts                 # Hono app entrypoint
│   ├── wrangler.toml                # Cloudflare Worker config (bindings for D1 + R2)
│   └── drizzle/                      # generated migrations
├── content-pipeline/
│   ├── workflows/                # ComfyUI graphs: qwen_image_edit.json (stills) + ltx_gemma_api.json (video)
│   ├── scripts/
│   │   ├── generate_batch.py     # calls ComfyUI API to render a batch of episodes from a script/prompt list
│   │   └── upload_to_r2.py       # pushes finished MP4s + thumbnails to R2, registers them via the backend API
│   ├── characters/                # one JSON + reference image per named character, for identity consistency across scenes
│   └── scripts_input/            # episode scripts live here (JSON: locations + scenes, see schema below)
├── packages/
│   └── shared/                   # shared TypeScript types (Episode, Series, User, Purchase) used by both mobile and server
├── pnpm-workspace.yaml
└── package.json
```



## Database Schema (Drizzle, targeting D1) — core models to implement first

- `User`: id, googlePlayAccountId (or device id for now), coinBalance, createdAt
- `Series`: id, title, description, coverImageUrl, isPublished
- `Episode`: id, seriesId, order, videoUrl, thumbnailUrl, coinCost, isFree (first N episodes free per series)
- `UnlockedEpisode`: userId, episodeId, unlockedAt (records which users unlocked which paid episodes)
- `Purchase`: id, userId, playOrderId, coinsGranted, amountPaid, verifiedAt, status
- `Subscription` (if hybrid model is chosen later): userId, tier, renewsAt, status

> Note: D1 is SQLite-based (single-writer model), not Postgres. This schema has no complex joins or high write-concurrency needs, so it's a good fit — don't add features that assume Postgres-specific behavior.



## Backend API Endpoints (Hono routes on Workers) — v1 scope

- `GET /series` — list published series with cover art
- `GET /series/:id/episodes` — episode list for a series, with lock status per requesting user
- `GET /episodes/:id/play` — returns a signed/temporary R2 URL for playback (only if free or unlocked)
- `POST /purchases/verify` — receives a Google Play purchase token, verifies server-side with Play Developer API, credits coins
- `POST /episodes/:id/unlock` — spends coins to unlock an episode, creates `UnlockedEpisode`
- `GET /users/me/wallet` — coin balance



## Mobile App Screens — v1 scope

- **Feed/Discover** — vertical swipeable list of series (TikTok-style)
- **Series detail** — episode list with lock icons on paid episodes
- **Player** — vertical video player, swipe up for next episode, paywall overlay triggers when hitting a locked episode
- **Paywall/Coin store** — coin packages, triggers Google Play Billing purchase flow
- **Profile** — coin balance, unlocked history



## Content Pipeline — v1 scope

- `generate_batch.py`: reads episode JSON from `scripts_input/`, resolves each scene's `characterIds` against `content-pipeline/characters/*.json` and `locationId` against that episode’s `locations` object, and prepends those `promptBlock`s. For each location+character pair, **Qwen-Image-Edit-2511** first builds a plate (`plate_<locationId>_<characterId>.png`) from the character PNG + location text. Every scene then has `imagePrompt`: Qwen edits that plate (plate as Picture 1, character PNG as Picture 2) into `scene_XX_start.png` so pose and props already exist in frame 0. After a human review of plates and scene stills, `pnpm run content:generate` animates each scene's start PNG with `videoPrompt` via LTX image-to-video (picture and native soundtrack in the same MP4). Finished MP4s stay local until `upload_to_r2.py` runs. Existing `plate_*.png` / `scene_*_start.png` / `scene_*.mp4` files are skipped; delete a file to regenerate it. Deleting a plate does not refresh scene stills — delete those too.
- `upload_to_r2.py`: uploads generated MP4s + auto-generated thumbnails to the R2 bucket, then calls the backend Worker's admin route to create the corresponding `Episode` record
- Simple admin script or Worker admin route to create/publish a `Series` and attach uploaded episodes to it in order

### Character consistency

Diffusion video models have no memory between separate generations — every scene is an independent roll of the dice, so the same text description produces a *different* person each time unless you deliberately lock identity. v1 does that in two places:

1. **A fixed character prompt block** — one detailed description (hair, face, age, wardrobe) reused word-for-word in every scene that character appears in.
2. **A reference still**, generated once per character and kept next to the JSON. `content:frames` feeds this PNG into **Qwen-Image-Edit-2511** as Picture 1 when building the location+character plate, and as Picture 2 when editing that plate into a scene still. Extra character PNGs (up to 3 total images including the plate) are additional identities. Do **not** feed the headshot into LTX as frame 0 — that would open every clip on the portrait. LTX only sees the reviewed `scene_XX_start.png`.

Qwen-Image-Edit-2511 is the still model because it is instruction-based editing from a photo (identity lock), Apache 2.0 (fine for a Play Store app), and the community default on 16GB cards as Unsloth’s Q4_K_M GGUF (~13GB) plus the 4-step Lightning LoRA. FLUX.1 Kontext is what some hosted LTX tools use for reference stills, but FLUX.1 [dev] is non-commercial. Z-Image Turbo is fast text-to-image and does not lock a real reference photo. Full BF16/FP8 Qwen-Edit checkpoints do not fit 16GB VRAM without heavy offload.

For recurring lead characters appearing across many episodes, consider training a small character LoRA later for tighter identity lock — out of scope for v1, but the `characters/` registry below is structured so a `lora` field can be added later without reshaping anything else.

`content-pipeline/characters/<character-slug>.json`:
```json
{
  "id": "elena-heiress",
  "referenceImage": "elena-heiress-ref.png",
  "promptBlock": "Elena, a woman in her late 20s with long loosely waved dark brown hair falling over her shoulders, warm tanned skin, blue-grey eyes, full lips, wearing a dark navy wool coat over a metallic bronze wrap top."
}
```

- `referenceImage` lives alongside the JSON file in the same `characters/` folder — generate it once (a single still image, any text-to-image tool) before writing scenes that use this character. A face crop is fine. **`promptBlock` must include wardrobe**; Qwen will copy a shirtless or coat-less ref photo unless clothes are named here.
- `generate_batch.py` looks up each scene's `characterIds`, prepends the matching `promptBlock`(s) to both `imagePrompt` and `videoPrompt`, and confirms the `referenceImage` file exists. `content:frames` sends that PNG into Qwen-Image-Edit as a **face** lock (Picture 1 on the plate, Picture 2 on the scene edit) and dresses the body from `promptBlock`; `content:generate` never uses it as the LTX first frame.

### Location consistency

Qwen-Edit is bad at compositing a person onto a second empty-room photo, so v1 does **not** feed empty plates as Picture 2. The room is locked by **one location+character plate** per episode, then each scene still is an **edit of that plate**.

LTX I2V never leaves that picture’s set ([I2V workflow](https://ltx.io/blog/ltx-2-image-to-video-text-to-video-workflow)): the start image is the location; the prompt describes what happens next. A single generation can hold character, setting, and voice across 2–4 shots joined by `Cut.` Prefer that for a continuous visit. Returning to a room uses the same `locationId` (same plate) with a new `imagePrompt` so props and pose are already in frame 0 — do not copy the previous scene still, or objects appear from nowhere.

Rooms live on the episode JSON as `locations`, keyed by `locationId` (`promptBlock` + `preserve`, optional `platePrompt`). No separate location files or photos. **`promptBlock` includes the set and where the character stands in it** (feet on the floor, scale vs furniture, camera). That is the plate. `imagePrompt` only changes pose and named props.

`content:frames` does two Qwen passes:

1. **Plate** (`plate_<locationId>_<characterId>.png`). Character PNG as a face lock (Picture 1) + location `promptBlock` (set + default stance). Dress from the character `promptBlock`; do not copy a shirtless ref. Optional `platePrompt` overrides the fallback framing.
2. **Scene still** (`scene_XX_start.png`). Plate as Picture 1, character PNG as a face lock (Picture 2), `imagePrompt` as the edit: pose, eyeline, hands, named props. Every scene has `imagePrompt`. Keep wardrobe and scale from the plate.

```json
"locations": {
  "mansion-foyer": {
    "promptBlock": "Grand foyer: cream paneled walls, white marble fireplace, large gold-framed oil portrait of four people hanging above the mantel, black-and-white diamond marble floor, warm wall sconces. Interior, no windows in frame. The woman stands in front of the mantel, both feet on the marble, full body at correct adult scale, the fireplace behind her not through her. Camera medium toward the mantel.",
    "preserve": "the same cream paneled foyer, white fireplace, gold-framed family portrait, black-and-white marble floor, warm sconces, the woman at the mantel at correct scale, fully clothed"
  }
}
```

- Define every room the episode uses in `locations`. Do not add `referenceImage`, `views`, or `locationView`.
- Every scene **must** have `locationId` matching a key in that object and an `imagePrompt`.
- `generate_batch.py` prepends location `promptBlock` when building the plate, `preserve` when editing a scene still, and `Preserve: {preserve}` on every `videoPrompt`.
- Plates are per episode (`output/<series>/<episode>/`). A later episode defines its own `locations` and generates new plates.

### Local ComfyUI (required before `content:frames` / `content:generate`)

`pnpm run content:frames` and `pnpm run content:generate` only talk to a ComfyUI HTTP API at `http://127.0.0.1:8188`. They do not install ComfyUI, download weights, or start the server. Do this on the GPU machine, in order:

1. `pnpm run content:setup-comfy` — clones [ComfyUI](https://github.com/comfyanonymous/ComfyUI), [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo) (includes `GemmaAPITextEncode`), [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF), and [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) into `content-pipeline/.comfyui`. Copies `content-pipeline/workflows/*.json` into ComfyUI’s user workflow folder and installs the small `reelshort_ltx` helper node so Gemma API embeddings (already projected to 6144) match the local Q4_K_M GGUF. Installs CUDA 12.8 PyTorch into the ComfyUI venv (required for RTX 50-series / Blackwell).
2. `pnpm run content:comfy-torch` — only if setup ran before CUDA torch was wired, or if `content:comfy` still says “Torch not compiled with CUDA enabled”.
3. `pnpm run content:models` — downloads:
   - `ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf` (~14GB) into `content-pipeline/.comfyui/models/diffusion_models/`, the matching LTX-2.3 video VAE into `models/vae/`, the matching audio VAE into `models/vae/` (copied into `models/checkpoints/` because `LTXVAudioVAELoader` lists that folder), and a tiny official LTX-2.3 distilled-1.1 safetensors stub into `models/checkpoints/` so `GemmaAPITextEncode` can read a `model_id` the LTX prompt-embedding API actually serves. The same stub is also written as `ltx-2-19b-distilled-api-id.safetensors` so an old graph still open in the ComfyUI UI does not show a missing-model error. The GGUF family must match that API model (LTX-2 19B is rejected by the current endpoint).
   - Qwen-Image-Edit-2511 Q4_K_M GGUF (`qwen-image-edit-2511-Q4_K_M.gguf`, ~13GB) into `models/diffusion_models/`, `qwen_2.5_vl_7b_fp8_scaled.safetensors` into `models/text_encoders/`, `qwen_image_vae.safetensors` into `models/vae/`, and the 4-step Lightning LoRA into `models/loras/`. This is the 16GB-card still stack (Apache 2.0). Skip files that are already present.
4. `pnpm run content:comfy` — starts ComfyUI on `127.0.0.1:8188` using the ComfyUI venv (not system Python). Leave this process running in its own terminal.
5. Open `http://127.0.0.1:8188` → **Load** → `qwen_image_edit.json`, then `ltx_gemma_api.json`. If ComfyUI reports missing nodes, the custom-node clone did not finish; rerun setup. If it reports a missing model/VAE/LoRA file, the filename in the workflow does not match a file on disk — point the loader node at the downloaded file.
6. Confirm `content-pipeline/.env` has `LTXV_API_KEY=...`. `content:generate` injects that key into the `GemmaAPITextEncode` node; do not hardcode it in the workflow JSON. `content:frames` does not need the LTX API key.
7. Then `pnpm run content:frames`. Review `content-pipeline/output/<series>/<episode>/scene_*_start.png`. Then `pnpm run content:generate`. `pnpm run content:render` runs those two in sequence with no still-review pause.

`content:frames` / `content:generate` post the matching workflow graph to ComfyUI’s `/prompt` API. The UI load step is only so you can see missing nodes/files before a long batch run. Both stages unload idle models first so the 16GB card is not holding Qwen and LTX at once.



### `scripts_input/` file format

One JSON file per episode, named `<series-slug>/<episode-number>.json`. This schema is designed to be generated directly by an AI chat — paste the schema below (plus the relevant character IDs / `promptBlock`s from `characters/`) into a chat and ask for episodes in this exact shape, then save the output as-is with no reformatting:

```json
{
  "series": "midnight-heiress",
  "episodeNumber": 1,
  "title": "The Return",
  "isFree": true,
  "coinCost": 0,
  "locations": {
    "mansion-gates": {
      "promptBlock": "Grey stone mansion with a steep slate roof and chimneys, seen through a pair of tall black wrought-iron gates that stand open on stone gateposts. Dusk sky, orange-gold sun on the facade, gravel drive, clipped hedges. The woman stands between the open gates, both feet on the gravel, full body at correct adult scale, gateposts beside her not through her. Camera outside the gates looking in toward her and the house.",
      "preserve": "the same grey stone mansion, open black iron gates, dusk orange light, gravel drive, the woman standing between the gates at correct scale, fully clothed"
    },
    "mansion-foyer": {
      "promptBlock": "Grand foyer: cream paneled walls, white marble fireplace, large gold-framed oil portrait of four people hanging above the mantel, black-and-white diamond marble floor, warm wall sconces. Interior, no windows in frame. The woman stands in front of the mantel, both feet on the marble, full body at correct adult scale, the fireplace behind her not through her. Camera medium toward the mantel.",
      "preserve": "the same cream paneled foyer, white fireplace, gold-framed family portrait, black-and-white marble floor, warm sconces, the woman at the mantel at correct scale, fully clothed"
    }
  },
  "scenes": [
    {
      "sceneNumber": 1,
      "characterIds": ["elena-heiress"],
      "locationId": "mansion-gates",
      "imagePrompt": "Elena stands between the open gates, body facing the house, chin lifted, eyes on the upper facade, hands at her sides, lips slightly parted, not looking at camera.",
      "videoPrompt": "Elena looks up slowly toward the mansion filling the upper frame. She says, \"Ten years.\" She pauses, then continues, \"Time to finish this.\" Camera static. Outdoor wind, a distant gate creak, no music.",
      "durationSeconds": 6
    },
    {
      "sceneNumber": 2,
      "characterIds": ["elena-heiress"],
      "locationId": "mansion-foyer",
      "imagePrompt": "Medium shot. Elena stands in front of the mantel, three-quarter to camera, chin lifted, eyes on the painted faces in the portrait, hands at her sides, lips slightly parted, not looking at camera. The gold-framed portrait hangs above her.",
      "videoPrompt": "Elena tilts her head slightly toward the portrait. She says, \"You should have told me.\" Cut. She pulls a sealed cream envelope from behind the lower-right of the gold frame and glances at it. She murmurs, \"What is this.\" Cut. She unfolds the letter and reads. She says, \"He sold the house.\" She pauses, then continues, \"They all knew.\" Camera slow push-in. Room echo, paper sliding, no music.",
      "durationSeconds": 9
    },
    {
      "sceneNumber": 3,
      "characterIds": ["elena-heiress"],
      "locationId": "mansion-foyer",
      "imagePrompt": "Medium shot. Elena stands in front of the mantel, the unfolded letter already in both hands, eyes on the page, then lifting toward camera, lips slightly parted. The gold-framed portrait hangs above her.",
      "videoPrompt": "Elena looks at the camera and says, \"I can keep secrets too.\" Camera static. Quiet room echo, no music.",
      "durationSeconds": 5
    }
  ]
}
```

- `scenes` is an ordered list. Every scene has `characterIds`, a required `locationId`, and a required `imagePrompt`. `content:frames` builds one Qwen plate per location+character pair, then edits that plate into each scene's start still; `content:generate` animates each start PNG with `videoPrompt`. Keep `imagePrompt` as framing + eyeline / hands / props / lips for **this** clip's frame 0. Keep `videoPrompt` as a cinematographer shot list (what happens next; `Cut.` for 2–4 connected shots; static or slow push; short quoted speech; matching foley). Character `promptBlock`s are prepended to both; location `promptBlock` is used on the plate; location `preserve` is prepended to scene-still edits and to `videoPrompt`. A legacy `prompt` field is still accepted as a fallback for `videoPrompt`.
- `durationSeconds` is the target **video** clip length. `content:generate` converts it to an LTX frame count at 24 fps using `8n+1` lengths (2s → 49 frames, 5s → 121, 6s → 145, 9s → 217). Start stills are 768×1152 Qwen edits (vertical 9:16, ~1MP); LTX I2V resizes them to the video latent (512×768 smoke size). `content:generate` forces Gemma `enhance_prompt` off so the shot list is not rewritten.
- `isFree`/`coinCost` map directly onto the `Episode` schema, so decide monetization per-episode right in the script file rather than as a separate step.
- When asking a chat AI to draft episodes, paste **Episode script authoring** below plus this JSON schema (and the relevant IDs / `promptBlock`s from `characters/`) and ask for only valid JSON (no prose, no markdown fences) so it can be saved as the `.json` file. The draft must include a `locations` object for every `locationId` the scenes use.

### Episode script authoring (distilled LTX + Qwen-Image-Edit)

Write shots these models can actually land. Pipeline: Qwen-Image-Edit builds a location+character plate from the character PNG, then edits that plate with `imagePrompt` into the scene still; LTX-2.3 **distilled** I2V animates the still from `videoPrompt` (picture and sound together). Distilled is weaker at prompt-following than the full LTX-2.3 dev checkpoint — do not write beats that need precision the distilled model does not have.

Prompting follows LTX’s own guides, not freeform style notes: [How to improve LTX-2.3 prompt adherence](https://ltx.io/blog/how-to-improve-ltx-2-3-prompt-adherence) (cinematographer shot list; main action first; chronological; one main action per 2–3 seconds; keep under 200 words; `enhance_prompt` off for control), [Directing dialogue and acting](https://ltx.io/blog/directing-dialogue-and-acting) (phrase + one cue + next phrase; eyeline / pause / voice / physical beat; audio last), [Character consistency](https://ltx.io/blog/how-to-maintain-character-consistency-in-ai-video) (fixed prompt block at the start of every prompt — the pipeline prepends `promptBlock`), [I2V workflow](https://ltx.io/blog/ltx-2-image-to-video-text-to-video-workflow) (in image-to-video, prompt what happens, not what the picture already shows; 2–4 connected shots joined by `Cut.` in one generation), [LTX 2.3 prompt template](https://ltx23.github.io/ltx-2-3-prompt-template/) (subject → action → camera → look). Left/right are **screen-space** (audience left/right of the frame), not the character’s own left/right.

**Output**
- One file: `content-pipeline/scripts_input/<series-slug>/<episode-number>.json`. JSON object only (no markdown fences).
- Use only existing `characterIds` from `content-pipeline/characters/`. `promptBlock` must include wardrobe. Define every room in this episode’s top-level `locations` object (`promptBlock` + `preserve`); `promptBlock` is the set plus default stance and scale. Scenes use those keys as `locationId`. Appearance must match that character `promptBlock`. The set is whatever the plate shows; do not invent a new room in `imagePrompt` or `videoPrompt`.
- Qwen-Edit takes at most **3** images (plate + character refs) on a scene still. Prefer **one speaker on camera** per scene.
- Every scene requires `imagePrompt`. Same `locationId` shares the plate; each scene still is a new edit of it. Prefer one scene per visit with `Cut.` (2–4 shots). Splitting or returning is fine; put this clip’s starting pose and props in `imagePrompt`.

**Do not write (unreliable on distilled)**
- Monologues or long unbroken quoted speeches (LTX-2.3 rushes or slurs them; segment instead).
- Two people talking in one shot, off-screen voices, or language other than English.
- Music / score.
- Plot that depends on readable on-screen text, logos, or signage.
- Crowds, fast fight choreography, morphs, 360 orbits, or a new location/wardrobe/face mid-clip.
- Five actions in one clip with no cuts. One main action per 2–3 seconds of `durationSeconds`. At most four `Cut.`-joined shots per scene.

**`imagePrompt`**
- Required on **every** scene. Qwen edits the location+character plate into this clip’s LTX frame 0. Framing of the actor plus blocking: body angle, **eyeline**, which hand, where the prop sits in **screen-space**, lips, whether they look at camera. No motion, no speech.
- `left` / `right` mean the audience’s frame (`screen-left`, `lower-right of the gold frame`). Use `her right hand` only for which hand, then place it in the frame.
- Hands must reach what they touch **in this still**. If the beat is later in the clip (she pulls the envelope after a `Cut.`), do not pre-pose it here — I2V does that from frame 0. If this **is** the starting beat (laptop already open, clipping already in hand), put that object in the still.
- If they speak this clip: **lips slightly parted**. Do not write jaw set / clenched / sealed lips / closed eyes. If they are not addressing the lens, write **not looking at camera**.
- Do not restate hair, wardrobe, or architecture. That lives on `promptBlock` / the plate.

**`videoPrompt`**
- I2V already has the still. Write what happens next, literally and in time order: main action, then gesture, then speech beats, then camera, then audio. Do not re-describe the picture.
- Same room, several beats: join 2–4 shots with `Cut.` so LTX holds character, set, and voice across them. Or split into extra scenes with the same `locationId` and a new `imagePrompt` — each clip starts from that edited still.
- Speech is LTX-2.3 segmented acting: short quoted phrase, **one** cue (eyeline, pause, voice quality, or physical beat), next phrase. Example: `She says, "Ten years." She pauses, then continues, "Time to finish this."` Not one unbroken paragraph of dialogue. One cue per beat — more than that reads twitchy.
- Always name the camera (`Camera static` or `Camera slow push-in`). Slow moves that underline a beat. No wide horizontal pans during dialogue.
- Audio last: acoustic space + matching foley, **no music**. Off-screen voices often replace the line.
- Do not contradict frame 0 (no new wardrobe or room). A new prop after `Cut.` is allowed if the action produces it. A gaze cue between phrases is valid; do not replace the spoken line with “looks past the camera.”

**Plot**
- Hook / turn / button is fine. Each scene is one I2V clip from its own start still. Do not pad with silent walks, rummages, or “a memory surfaces.”
- Recurring rooms reuse `locationId` (same plate, new `imagePrompt`). Do not invent a new mansion in prose.



## Deployment

See the **Command Interface** section above — `pnpm run setup` provisions R2/D1, `pnpm run dev:server` runs the Worker locally, `pnpm run deploy` ships it. No droplet, no Docker, no SSH.

## 🤖 AI Agent Build Order (check off as completed, in sequence)

> The agent should check these off itself as each is genuinely finished and working, not just started.

- [ ] 1. Scaffold the pnpm monorepo, `shared` package with core types, and empty `mobile`/`server` apps (`server` as a Cloudflare Worker using Hono) that build and run locally via `wrangler dev`.
- [ ] 2. Set up the Drizzle schema and generate D1 migrations; apply them to a local D1 instance via `wrangler d1 migrations apply --local`.
- [ ] 3. Build `series`/`episodes` read routes in the Worker, seed the local D1 DB with a few dummy series/episodes pointing at placeholder video URLs.
- [ ] 4. Build the mobile Feed → Series detail → Player flow against those dummy endpoints, using Expo's video component for vertical playback.
- [ ] 5. Add coin wallet + unlock logic (backend routes) and paywall UI (mobile) using dummy coin balances (no real payment yet).
- [ ] 6. Integrate Google Play Billing purchase flow in mobile + server-side verification route in the Worker.
- [x] 7. Build the content-pipeline scripts (`generate_batch.py`, `upload_to_r2.py`) to consume the `scripts_input/` JSON schema (episode `locations` object + scenes) and the `characters/` registry defined in the Content Pipeline section (character refs, location+character plates, per-scene still edits, no empty-room compositing), and the R2 binding/upload logic in the Worker.
- [ ] 8. Wire everything together: real generated episodes flowing from the pipeline into R2 into the app.
- [ ] 9. Write `scripts/setup-cloudflare.sh` and `scripts/deploy.sh`, wire them into root `package.json` as `setup` and `deploy` scripts, finalize `wrangler.toml` bindings with placeholder IDs, and document the first real deploy in `DEPLOY.md` — the human only needs to run `wrangler login`, then `pnpm run setup` and paste the printed IDs into `wrangler.toml`.
- [ ] 10. Add a minimal privacy policy static page and any other Play Store listing requirements (app description, screenshots).
- [ ] 11. Prepare a production build (`eas build` for Expo) for Google Play upload — hand off to the human here for account setup and closed testing.



## Explicitly out of scope for v1 (revisit later)

- iOS build (architecture should not block this, but do not implement Apple-specific code yet)
- Subscription tier (start coins-only, add subscription model later if desired)
- Recommendation algorithm (v1 feed can be simple reverse-chronological or manually curated order)

