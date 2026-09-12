#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../server"
pnpm exec wrangler deploy
