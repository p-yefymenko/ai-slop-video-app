#!/bin/bash
# One-time setup. Requires `pnpm run login:cloudflare` to have been run already.
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
echo "  pnpm run secret:admin"
echo "  pnpm run secret:media"
echo "  pnpm run secret:play"
