#!/bin/bash
# One-time setup. Requires `wrangler login` to have been run already.
set -euo pipefail

echo "Creating R2 bucket reelshort-videos..."
pnpm exec wrangler r2 bucket create reelshort-videos || true

echo "Creating D1 database reelshort-db..."
pnpm exec wrangler d1 create reelshort-db

echo
echo "Copy the printed database_id into server/wrangler.toml under [[d1_databases]]."
echo "Set bucket_name to reelshort-videos under [[r2_buckets]] if it is not already."
echo "Then run: pnpm run db:migrate:remote"
echo "Optional secrets:"
echo "  pnpm --filter server exec wrangler secret put ADMIN_SECRET"
echo "  pnpm --filter server exec wrangler secret put MEDIA_SIGNING_SECRET"
echo "  pnpm --filter server exec wrangler secret put GOOGLE_PLAY_SERVICE_ACCOUNT"
