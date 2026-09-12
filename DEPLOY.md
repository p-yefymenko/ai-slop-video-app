# First Worker deploy

The human-only Cloudflare steps are: create a Cloudflare account, run `wrangler login` once, then run the scripts below.

1. `pnpm run setup` — creates the R2 bucket `reelshort-videos` and D1 database `reelshort-db`. Paste the printed `database_id` into `server/wrangler.toml` (`[[d1_databases]].database_id`). Keep `bucket_name = "reelshort-videos"`.
2. `pnpm --filter server exec wrangler secret put ADMIN_SECRET`
3. `pnpm --filter server exec wrangler secret put MEDIA_SIGNING_SECRET`
4. `pnpm --filter server exec wrangler secret put GOOGLE_PLAY_SERVICE_ACCOUNT` (paste the Play service-account JSON as the secret value)
5. `pnpm run db:migrate:remote`
6. `pnpm run deploy`

Local loop (no Cloudflare account needed after code is installed):

1. `pnpm run db:migrate:local`
2. `pnpm run db:seed`
3. `pnpm run dev:server`
4. `pnpm run dev:mobile`

Set `EXPO_PUBLIC_API_URL` if the phone is not using the Android emulator (`http://10.0.2.2:8787`) or iOS simulator (`http://127.0.0.1:8787`).

Privacy policy URL after deploy: `https://reelshort-api.<your-subdomain>.workers.dev/privacy`

Android production build (after Expo/EAS login and `eas init` inside `apps/mobile`):

`pnpm run build:android`
