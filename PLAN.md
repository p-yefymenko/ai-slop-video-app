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
8. [ ] **Write a show JSON.** One file per show at `content-pipeline/scripts_input/<id>.json` (`ShowScript` in `packages/shared/src/script.ts`). Put characters, locations (text only), every prompt template, and episodes in that file. `pnpm run content:frames` generates character stills from it — no handmade PNGs in `characters/`.
9. [ ] Run `pnpm run content:frames`. Review `characters/` then `scene_*_start.png` in `content-pipeline/output/<show>/`. Keep, replace, or delete any file you do not like. Then run `pnpm run content:generate` and `pnpm run content:upload` on your own machine (the RTX 5070 Ti). ComfyUI must already be running from the previous step.
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
    "content:previs": "node scripts/run-python.cjs content-pipeline/scripts/spatial_previs.py",
    "content:test-spatial": "node scripts/run-python.cjs -m unittest discover -s content-pipeline/tests -p test_spatial_*.py",
    "content:frames": "node scripts/run-python.cjs content-pipeline/scripts/generate_batch.py --stage frames",
    "content:frame": "node scripts/run-python.cjs content-pipeline/scripts/generate_batch.py --stage frames",
    "content:generate": "node scripts/run-python.cjs content-pipeline/scripts/generate_batch.py --stage video",
    "content:clip": "node scripts/run-python.cjs content-pipeline/scripts/generate_batch.py --stage video",
    "content:render": "node scripts/render-content.cjs",
    "content:archive": "node scripts/archive-content-output.cjs",
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
- **Content generation (offline, not part of the live app):** Each episode first defines a deterministic 3D blocking timeline: measured sets, character/prop keyframes, and physical cameras. `content:previs` projects that state into proxy frames and a contact sheet. **Qwen-Image-Edit-2511** receives an eyeline-matched profile identity plus the proxy; dialogue singles receive only an abstracted, person-free continuity plate so reference people and set landmarks cannot be duplicated. Prop inserts derive their subject from an earlier photorealistic master. Generated stills pass a duplicate-face gate before video. Establishing shots, inserts, reactions, and silent anchors default to stable supersampled camera motion. Dialogue/action shots use **LTX-2.3 distilled-1.1**; dialogue additionally uses audio-video modality guidance and explicit lip-sync conditioning. Gemma API supplies text conditioning within 16GB VRAM.
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
│   ├── scripts_input/            # one ShowScript JSON per show (`<id>.json`)
├── packages/
│   └── shared/                   # shared TypeScript types (Episode, Series, User, Purchase, ShowScript) used by mobile, server, and the content pipeline
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

- `generate_batch.py`: reads one `ShowScript` JSON per show from `scripts_input/<id>.json`. `pnpm run content:frames` generates missing character stills, then each scene still from those portraits plus location text. `pnpm run content:generate` animates each scene still with LTX I2V. Every instruction string comes from that JSON’s `prompts` templates — nothing is hardcoded in the Python. Existing PNGs/MP4s are skipped unless one selected scene uses `--force`.
- `upload_to_r2.py`: uploads generated MP4s + auto-generated thumbnails to the R2 bucket, then calls the backend Worker's admin route to create the corresponding `Episode` record
- Simple admin script or Worker admin route to create/publish a `Series` and attach uploaded episodes to it in order

`content:frames` writes:

1. Character stills — `output/<show>/characters/<id>.png` (human review, then Qwen Picture 1 / Picture 2 on scene stills)
2. Spatial previs — `output/<show>/<episode>/previs/` (proxy start/end frames and contact sheet)
3. Scene stills — `scene_XX_start.png`, plus `scene_XX_end.png` when `endGuideFrame` is enabled

`content:generate` writes `scene_XX.mp4` and concatenates `episode.mp4`. LTX only sees the reviewed scene still, never the character portrait.

`content:render` runs `content:previs`, `content:frames`, and `content:generate` in that order and forwards the same selection arguments to all three stages.

The vertical render profile is 768x1360 for Qwen stills and 448x800 for LTX clips. Both are near 9:16; the LTX size is divisible by 32 and uses fewer pixels than the old 512x768 2:3 profile, which is necessary on the 16GB card.

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
7. Then `pnpm run content:frames`. Review `content-pipeline/output/<show>/characters/` and each episode’s `scene_*_start.png`. Then `pnpm run content:generate`. `pnpm run content:render` runs those two in sequence with no still-review pause.

`content:frames` / `content:generate` post the matching workflow graph to ComfyUI’s `/prompt` API. The UI load step is only so you can see missing nodes/files before a long batch run. Both stages unload idle models first so the 16GB card is not holding Qwen and LTX at once.



### `scripts_input/` file format

One JSON file per show: `content-pipeline/scripts_input/<id>.json`. Shape is `ShowScript` in `packages/shared/src/script.ts`. Filename stem must match `id`.

```json
{
  "id": "midnight-heiress",
  "title": "Midnight Heiress",
  "characters": {
    "elena-heiress": {
      "promptBlock": "Elena, late 20s, dark hair, navy wool coat over a metallic bronze wrap top."
    }
  },
  "locations": {
    "mansion-gates": {
      "promptBlock": "Grey stone mansion, open iron gates, dusk, gravel drive."
    }
  },
  "prompts": {
    "characterImage": "Photorealistic vertical 9:16 identity reference, waist-up, exactly one person facing camera, neutral closed-mouth expression, hands out of frame, plain fitted crew-neck shirt, plain warm-grey studio backdrop, soft even light, natural skin, sharp eyes. No text, props, jewelry, costume, or other people. Ignore the attached blank image and create a new person from this description: {characterPromptBlock}",
    "sceneStill": "The attached identity pictures map exactly as follows: {referenceMap} Transform those people into one new photorealistic vertical 9:16 scene. The finished scene contains exactly {characterCount} visible people: {characterIds}. Each appears once only. No duplicates, twins, background people, portraits, paintings, mirrors, or reflections. Preserve each referenced face, hair, apparent age, and skin tone, but do not copy the reference backdrop, shirt, pose, or gaze. This is a dramatic film frame, not a frontal identity portrait; obey the stated eyeline and placement. Do not visualize an off-frame or absent addressee. {locationPromptBlock} {imagePrompt}",
    "coverageStill": "The attached pictures map exactly as follows: {referenceMap}; Picture {coverageReferencePictureNumber} = the rendered coverage master. Reframe the same cinematic moment as a new vertical 9:16 shot, not a collage. Preserve the identity from the identity picture(s), and inherit the set architecture, wardrobe, lighting direction, color, axis, and screen geography from the coverage master. The finished frame contains exactly {characterCount} visible people: {characterIds}. Do not retain, duplicate, or invent any other person from the master. {imagePrompt}",
    "environmentStill": "Create a new photorealistic cinematic vertical 9:16 frame from the attached blank canvas. The finished frame contains no people, faces, portraits, statues shaped like people, mirrors, reflections, or readable text. {locationPromptBlock} {imagePrompt}",
    "sceneVideo": "Continue directly from this image as the exact first frame. Preserve its people, wardrobe, props, set, composition, and lighting. Use one continuous take. Animate only the motion, performance, camera, dialogue, and sound described here: {videoPrompt}"
  },
  "episodes": [
    {
      "episodeNumber": 1,
      "title": "The Return",
      "isFree": true,
      "coinCost": 0,
      "logline": "Elena reaches the estate before her enemy can destroy her inheritance.",
      "dramaticQuestion": "Will Elena enter before the gates close?",
      "hook": "Elena stands outside the closing gates.",
      "reversal": "She recognizes the person ordering the gates shut.",
      "cliffhanger": "Elena crosses the threshold and the gates lock behind her.",
      "nextEpisodeOpening": "Elena faces the enemy waiting inside the locked gates.",
      "screenDirection": "At the gates, Elena is screen left and looks frame right toward the estate.",
      "scenes": [
        {
          "sceneNumber": 1,
          "beatType": "hook",
          "coverageRole": "closeup",
          "locationId": "mansion-gates",
          "storyBeat": "Elena returns to the estate and chooses to confront her past.",
          "continuityIn": "Elena has just arrived alone outside the open gates.",
          "continuityOut": "Elena commits to entering the estate.",
          "shotType": "single",
          "characterIds": ["elena-heiress"],
          "speakerId": "elena-heiress",
          "addresseeId": null,
          "imagePrompt": "Chest-up single of Elena in a navy wool coat, facing the estate off-frame right with guarded resolve, never looking at camera.",
          "videoPrompt": "Elena speaks immediately with no silent pause, her low voice fully audible and weighted with dread: \"Ten years, and this place still knows how to frighten me.\" Her resolve hardens afterward. Camera locked. Outdoor wind stirs.",
          "durationSeconds": 6
        }
      ]
    }
  ]
}
```

- `prompts` is the only place instruction text lives. `{placeholders}` are filled from the matching fields. Do not put lock/blocking copy in Python.
- `locations` is a short environment clause (where they are), reused verbatim. Not a camera. Scenes point at it with `locationId`.
- Screenwriting guidance lives in `.cursor/rules/episode-scripts.mdc`; renderer field semantics live in `packages/shared/src/script.ts`. Draft the 0–60 second hook/pressure/reversal/cliffhanger skeleton before prompts.
- `spatialTimeline` is the physical source of truth. Every location has measured geometry and every shot—including establishing shots and inserts—has `timeRangeSeconds` plus a physical camera. Prompt-only scenes are invalid. Character tracks define timed position, body yaw, eye target, stance, and hand targets; props have one timed position or owner.
- `endGuideFrame` is reserved for action shots whose final blocking differs from the start. Static dialogue stays on one mark and camera; pinning an identical LTX guide adds substantial render cost without adding spatial information.
- `motionMode` is `cameraOnly` or `generative`. Silent anchors, inserts, reactions, and establishing shots default to `cameraOnly`; opt into generative motion only for one simple visible action.
- Spatial singles use left/right profile identity references plus an abstracted continuity plate. The plate must contain no recognizable person or landmark copy. In-set prop inserts require `coverageReferenceSceneNumber` and `focusTargetId`.
- `shotType` is `establishing`, `insert`, `single`, `reaction`, or `twoShot`. Use 2–3 second silent two-shots as spatial anchors, then set each related single's `coverageReferenceSceneNumber` to that anchor. This makes the coverage share a rendered set and axis instead of resembling unrelated portraits. For three people, establish pairwise anchors when the active pair changes.
- `establishing` and `insert` shots use empty `characterIds` and `prompts.environmentStill`; the pipeline supplies a blank canvas instead of identity references. Use these shots for geography, architecture, creatures, weather, and hero props that create visual scale.
- `screenDirection` fixes the 180-degree axis per recurring location before shots are written. Every scene derives left/right eyelines from it; never choose eyelines independently per prompt.
- `speakerId` and `addresseeId` disambiguate dialogue and eyelines. A reaction shot may use an off-screen `speakerId`; other shot types require a visible speaker. One quoted line maximum, 16 words maximum.
- `imagePrompt` is the exact first frame: wardrobe, prop state, shot size, placement, and gaze. Identity is the PNG.
- `imagePrompt` may name only characters in `characterIds`. Off-frame eyelines use empty left/right space without naming the absent addressee, preventing Qwen from inventing an unreferenced extra person.
- Dialogue singles use tight head-and-shoulders stills so identity detail survives animation.
- A `videoPrompt` uses compact `VISUAL:`, `DIALOGUE:`, `CAMERA:`, and optional `AUDIO:` clauses. Include only literal motion and essential sound; LTX may visualize decorative foley or metaphoric directions.
- `durationSeconds` is flexible and converted to `8n+1` frames at 24 fps. Episode scenes should total about 60 seconds.
- The Python loader validates only render-critical structure such as required fields, known location/visible-character IDs, at most two Qwen character references, sequential output numbers, and positive duration. It does not reject scripts for creative guidance such as pacing, dialogue length, shot semantics, or prompt wording.
- Generated files are skipped when present. After a structural script rewrite, use `pnpm run content:archive -- <show-id>` before generating fresh frames. It archives old episode assets while retaining the reviewed character identity PNGs in the active output folder.
- Character portraits, scene stills, and clips use stable per-shot seeds by default, so an unchanged shot reproduces instead of changing randomly between full renders.
- Iterate on one shot with `pnpm run content:frame -- --show <show-id> --episode <n> --scene <n> --force`, then test only its video with the same filters through `pnpm run content:clip`. To intentionally reroll a bad still or clip, add `--seed <0-4294967295>` to that selected `--force` command. A partial clip render never overwrites `episode.mp4`.


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
- [x] 7. Build the content-pipeline scripts (`generate_batch.py`, `upload_to_r2.py`) to consume one `ShowScript` JSON per show (`scripts_input/<id>.json`: characters, locations, prompt templates, episodes) and the R2 binding/upload logic in the Worker.
- [ ] 8. Wire everything together: real generated episodes flowing from the pipeline into R2 into the app.
- [ ] 9. Write `scripts/setup-cloudflare.sh` and `scripts/deploy.sh`, wire them into root `package.json` as `setup` and `deploy` scripts, finalize `wrangler.toml` bindings with placeholder IDs, and document the first real deploy in `DEPLOY.md` — the human only needs to run `wrangler login`, then `pnpm run setup` and paste the printed IDs into `wrangler.toml`.
- [ ] 10. Add a minimal privacy policy static page and any other Play Store listing requirements (app description, screenshots).
- [ ] 11. Prepare a production build (`eas build` for Expo) for Google Play upload — hand off to the human here for account setup and closed testing.



## Explicitly out of scope for v1 (revisit later)

- iOS build (architecture should not block this, but do not implement Apple-specific code yet)
- Subscription tier (start coins-only, add subscription model later if desired)
- Recommendation algorithm (v1 feed can be simple reverse-chronological or manually curated order)

