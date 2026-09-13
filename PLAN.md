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
8. [ ] **Create character reference images and prompt blocks first, then write or source episode scripts.** For each named character, generate one reference image and write a fixed description block (see **Character consistency** in the Content Pipeline section below) — do this before writing scenes, since scenes reference these by `characterId`. Then write the story as `imagePrompt` + `videoPrompt` following **Episode script authoring** (distilled LTX / Qwen-Edit limits). Original writing, translated/licensed material reworked into scene prompts, or AI-assisted drafting are all fair game. Paste that authoring block plus the JSON schema into a chat and save the output with no reformatting. None of this has a code dependency — do it whenever, then drop files into `content-pipeline/characters/` and `content-pipeline/scripts_input/`.
9. [ ] Run `pnpm run content:frames`, look at the `scene_*_start.png` files in `content-pipeline/output/`, and keep, replace, or delete any you do not like (delete + rerun `content:frames` regenerates only the missing stills). Then run `pnpm run content:generate` and `pnpm run content:upload` on your own machine (the RTX 5070 Ti). ComfyUI must already be running from the previous step.
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
- **Content generation (offline, not part of the live app):** ComfyUI running locally on the GPU machine. Start stills come from **Qwen-Image-Edit-2511** (Q4_K_M GGUF + Lightning 4-step LoRA) using each character’s reference PNG as identity. Those stills are then animated with **LTX-2.3 distilled-1.1** (Q4_K_M GGUF) image-to-video, with Gemma API used for LTX text-encoder conditioning to stay within 16GB VRAM. Qwen and LTX are not meant to stay loaded together; `content:frames` and `content:generate` unload idle models between stages. Output MP4s are uploaded to R2 via a script, not generated at runtime.
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
│   └── scripts_input/            # episode scripts live here (JSON, see schema below)
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

- `generate_batch.py`: reads episode JSON from `scripts_input/`, resolves each scene's `characterIds` against `content-pipeline/characters/*.json`, and prepends character `promptBlock`s. `pnpm run content:frames` runs **Qwen-Image-Edit-2511** on the character reference PNG(s) plus `imagePrompt`, writing `scene_XX_start.png`. After a human review of those PNGs, `pnpm run content:generate` animates each approved still with `videoPrompt` via LTX image-to-video (picture and native soundtrack in the same MP4). Finished MP4s stay local until `upload_to_r2.py` runs.
- `upload_to_r2.py`: uploads generated MP4s + auto-generated thumbnails to the R2 bucket, then calls the backend Worker's admin route to create the corresponding `Episode` record
- Simple admin script or Worker admin route to create/publish a `Series` and attach uploaded episodes to it in order

### Character consistency

Diffusion video models have no memory between separate generations — every scene is an independent roll of the dice, so the same text description produces a *different* person each time unless you deliberately lock identity. v1 does that in two places:

1. **A fixed character prompt block** — one detailed description (hair, face, age, wardrobe) reused word-for-word in every scene that character appears in.
2. **A reference still**, generated once per character and kept next to the JSON. `content:frames` feeds this PNG into **Qwen-Image-Edit-2511** as Picture 1 (up to three refs) so the scene still keeps that face while following `imagePrompt` (wide shot, close-up, new wardrobe/location). Do **not** feed the headshot into LTX as frame 0 — that would open every clip on the portrait. LTX only sees the reviewed `scene_XX_start.png`.

Qwen-Image-Edit-2511 is the still model because it is instruction-based editing from a photo (identity lock), Apache 2.0 (fine for a Play Store app), and the community default on 16GB cards as Unsloth’s Q4_K_M GGUF (~13GB) plus the 4-step Lightning LoRA. FLUX.1 Kontext is what some hosted LTX tools use for reference stills, but FLUX.1 [dev] is non-commercial. Z-Image Turbo is fast text-to-image and does not lock a real reference photo. Full BF16/FP8 Qwen-Edit checkpoints do not fit 16GB VRAM without heavy offload.

For recurring lead characters appearing across many episodes, consider training a small character LoRA later for tighter identity lock — out of scope for v1, but the `characters/` registry below is structured so a `lora` field can be added later without reshaping anything else.

`content-pipeline/characters/<character-slug>.json`:
```json
{
  "id": "elena-heiress",
  "referenceImage": "elena-heiress-ref.png",
  "promptBlock": "Elena, a woman in her late 20s with long loosely waved dark brown hair falling over her shoulders, warm tanned skin, blue-grey eyes, full lips, wearing a metallic bronze wrap top. Cinematic lighting, photorealistic."
}
```

- `referenceImage` lives alongside the JSON file in the same `characters/` folder — generate it once (a single still image, any text-to-image tool) before writing scenes that use this character.
- `generate_batch.py` looks up each scene's `characterIds`, prepends the matching `promptBlock`(s) to both `imagePrompt` and `videoPrompt`, and confirms the `referenceImage` file exists. `content:frames` sends that PNG into Qwen-Image-Edit; `content:generate` never uses it as the LTX first frame.

### Local ComfyUI (required before `content:frames` / `content:generate`)

`pnpm run content:frames` and `pnpm run content:generate` only talk to a ComfyUI HTTP API at `http://127.0.0.1:8188`. They do not install ComfyUI, download weights, or start the server. Do this on the GPU machine, in order:

1. `pnpm run content:setup-comfy` — clones [ComfyUI](https://github.com/comfyanonymous/ComfyUI), [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo) (includes `GemmaAPITextEncode`), [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF), and [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) into `content-pipeline/.comfyui`. Copies `content-pipeline/workflows/*.json` into ComfyUI’s user workflow folder and installs the small `reelshort_ltx` helper node so Gemma API embeddings (already projected to 6144) match the local Q4_K_M GGUF. Installs CUDA 12.8 PyTorch into the ComfyUI venv (required for RTX 50-series / Blackwell).
2. `pnpm run content:comfy-torch` — only if setup ran before CUDA torch was wired, or if `content:comfy` still says “Torch not compiled with CUDA enabled”.
3. `pnpm run content:models` — downloads:
   - `ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf` (~14GB) into `content-pipeline/.comfyui/models/diffusion_models/`, the matching LTX-2.3 video VAE into `models/vae/`, the matching audio VAE into `models/vae/` (copied into `models/checkpoints/` because `LTXVAudioVAELoader` lists that folder), and a tiny official LTX-2.3 distilled-1.1 safetensors stub into `models/checkpoints/` so `GemmaAPITextEncode` can read a `model_id` the LTX prompt-embedding API actually serves. The GGUF family must match that API model (LTX-2 19B is rejected by the current endpoint).
   - Qwen-Image-Edit-2511 Q4_K_M GGUF (`qwen-image-edit-2511-Q4_K_M.gguf`, ~13GB) into `models/diffusion_models/`, `qwen_2.5_vl_7b_fp8_scaled.safetensors` into `models/text_encoders/`, `qwen_image_vae.safetensors` into `models/vae/`, and the 4-step Lightning LoRA into `models/loras/`. This is the 16GB-card still stack (Apache 2.0). Skip files that are already present.
4. `pnpm run content:comfy` — starts ComfyUI on `127.0.0.1:8188` using the ComfyUI venv (not system Python). Leave this process running in its own terminal.
5. Open `http://127.0.0.1:8188` → **Load** → `qwen_image_edit.json`, then `ltx_gemma_api.json`. If ComfyUI reports missing nodes, the custom-node clone did not finish; rerun setup. If it reports a missing model/VAE/LoRA file, the filename in the workflow does not match a file on disk — point the loader node at the downloaded file.
6. Confirm `content-pipeline/.env` has `LTXV_API_KEY=...`. `content:generate` injects that key into the `GemmaAPITextEncode` node; do not hardcode it in the workflow JSON. `content:frames` does not need the LTX API key.
7. Then `pnpm run content:frames`. Review `content-pipeline/output/<series>/<episode>/scene_*_start.png`. Then `pnpm run content:generate`. `pnpm run content:render` runs those two in sequence with no still-review pause.

`content:frames` / `content:generate` post the matching workflow graph to ComfyUI’s `/prompt` API. The UI load step is only so you can see missing nodes/files before a long batch run. Both stages unload idle models first so the 16GB card is not holding Qwen and LTX at once. On Windows, `content:comfy`, `content:frames`, `content:generate`, and `content:render` tell the OS to skip idle sleep while they run (GPU load alone does not). Closing the lid can still sleep the machine.



### `scripts_input/` file format

One JSON file per episode, named `<series-slug>/<episode-number>.json`. This schema is designed to be generated directly by an AI chat — paste the schema below (plus the relevant character IDs from your `characters/` registry) into a chat and ask for episodes in this exact shape, then save the output as-is with no reformatting:

```json
{
  "series": "midnight-heiress",
  "episodeNumber": 1,
  "title": "The Return",
  "isFree": true,
  "coinCost": 0,
  "scenes": [
    {
      "sceneNumber": 1,
      "characterIds": ["elena-heiress"],
      "imagePrompt": "Cinematic still photograph, vertical 9:16 frame. Elena stands beside an open black limousine in front of a glass skyscraper at dusk, neon city lights reflecting on wet pavement, wide shot, dramatic lighting, photorealistic, sharp, no motion blur.",
      "videoPrompt": "Elena steps out of the limousine and walks forward, her hair moving in the wind, camera slowly pushes in. She says, \"I'm home.\" Distant traffic and a car door thud. No music.",
      "durationSeconds": 2
    },
    {
      "sceneNumber": 2,
      "characterIds": ["elena-heiress"],
      "imagePrompt": "Cinematic still photograph, vertical 9:16 close-up of Elena's face, determined expression, hair slightly lifted, dusk city lights blurred behind her, photorealistic, sharp, no motion blur.",
      "videoPrompt": "Close-up on her face, determined expression, wind blowing her hair, camera slowly pushes in. She says, \"Lock the door.\" Night wind and distant traffic. No music.",
      "durationSeconds": 2
    }
  ]
}
```

- `scenes` is an ordered list. `content:frames` turns each `imagePrompt` plus the scene’s character reference PNG(s) into `scene_XX_start.png` with Qwen-Image-Edit; `content:generate` then animates that PNG with `videoPrompt` (pixels and sound together). Keep `imagePrompt` as a locked composition (subject, wardrobe, camera, lighting, still photograph). Keep `videoPrompt` as motion from that still (action, camera move, short quoted speech, matching foley). Both get the character `promptBlock` prepended automatically. A legacy `prompt` field is still accepted as a fallback for either.
- `durationSeconds` is the target **video** clip length. `content:generate` converts it to an LTX frame count at 24 fps using `8n+1` lengths (2s → 49 frames, 5s → 121, 6s → 145). Start stills are 768×1152 Qwen edits (vertical 9:16, ~1MP); LTX I2V resizes them to the video latent (512×768 smoke size).
- `isFree`/`coinCost` map directly onto the `Episode` schema, so decide monetization per-episode right in the script file rather than as a separate step.
- When asking a chat AI to draft episodes, paste **Episode script authoring** below plus this JSON schema (and the relevant character IDs / `promptBlock`s from `characters/`) and ask for only valid JSON (no prose, no markdown fences) so it can be saved as the `.json` file.

### Episode script authoring (distilled LTX + Qwen-Image-Edit)

Write shots these models can actually land. Pipeline: Qwen-Image-Edit builds a 9:16 start still from the character PNG + `imagePrompt`; LTX-2.3 **distilled** I2V animates that still from `videoPrompt` (picture and sound together). Distilled is weaker at prompt-following than the full LTX-2.3 dev checkpoint — do not write beats that need precision the distilled model does not have.

**Output**
- One file: `content-pipeline/scripts_input/<series-slug>/<episode-number>.json`. JSON object only (no markdown fences).
- Use only existing `characterIds` from `content-pipeline/characters/`. Appearance must match that `promptBlock` and reference PNG.
- Qwen-Edit takes at most **3** character refs; prefer **one speaker on camera** per shot.

**Do not write (unreliable on distilled)**
- Monologues or long unbroken quoted speeches (it rushes, skips, or slurs).
- Two people talking in one shot, off-screen voices, or language other than English.
- Music / score.
- Plot that depends on readable on-screen text, logos, or signage.
- Crowds, fast fight choreography, morphs, 360 orbits, or a new location/wardrobe/face mid-clip.

**`imagePrompt`**
- Vertical 9:16 still photograph: who, wardrobe, place, camera distance, light. No motion, no speech, no camera move.
- Same person as the reference PNG. Change the *shot* (wide vs close-up), not their identity.
- If they speak this clip: **lips slightly parted**, medium or medium close-up, and any prop they handle already in frame. Do not write jaw set / clenched / sealed lips / closed eyes, or a face-only CU that hides the letter or hands. Wardrobe used later in the episode (a coat, a bag) belongs in every still.

**`videoPrompt`**
- Animate **that** still: one action + one slow camera move. Do not contradict the still’s framing or invent objects that are out of frame.
- Every clip has speech: short quoted phrases with a beat between them (`She says, "I'm home." She pauses, then, "Lock the door."`). Size `durationSeconds` to that line + action — not a paragraph of speech in one clip. Prefer plain words over long idioms and stacked contractions.
- After the action, name matching foley and space (door thud, wet pavement, distant traffic). **No music.** Off-screen voices or extra foley often replace the line. Do not write “looks past the camera” on a talking close-up.

**Plot**
- Hook / turn / button is fine. Each beat is one I2V clip from a still, with a short spoken line, not a theatre scene. Do not pad with silent walks, rummages, or “a memory surfaces.”



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
- [x] 7. Build the content-pipeline scripts (`generate_batch.py`, `upload_to_r2.py`) to consume the `scripts_input/` JSON schema and the `characters/` registry defined in the Content Pipeline section (character prompt block + reference image resolution per scene), and the R2 binding/upload logic in the Worker.
- [ ] 8. Wire everything together: real generated episodes flowing from the pipeline into R2 into the app.
- [ ] 9. Write `scripts/setup-cloudflare.sh` and `scripts/deploy.sh`, wire them into root `package.json` as `setup` and `deploy` scripts, finalize `wrangler.toml` bindings with placeholder IDs, and document the first real deploy in `DEPLOY.md` — the human only needs to run `wrangler login`, then `pnpm run setup` and paste the printed IDs into `wrangler.toml`.
- [ ] 10. Add a minimal privacy policy static page and any other Play Store listing requirements (app description, screenshots).
- [ ] 11. Prepare a production build (`eas build` for Expo) for Google Play upload — hand off to the human here for account setup and closed testing.



## Explicitly out of scope for v1 (revisit later)

- iOS build (architecture should not block this, but do not implement Apple-specific code yet)
- Subscription tier (start coins-only, add subscription model later if desired)
- Recommendation algorithm (v1 feed can be simple reverse-chronological or manually curated order)

