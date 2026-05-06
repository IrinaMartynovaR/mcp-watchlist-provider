#!/usr/bin/env bash
set -euo pipefail

mkdir -p vendor/langfuse
curl -fsSL https://raw.githubusercontent.com/langfuse/langfuse/main/docker-compose.yml \
  -o vendor/langfuse/docker-compose.yml

cat <<'MSG'
Downloaded official Langfuse docker-compose.yml to vendor/langfuse/docker-compose.yml

Next steps:
1. Open vendor/langfuse/docker-compose.yml
2. Replace all default secrets/passwords for real use
3. Start Langfuse:
   docker compose -f vendor/langfuse/docker-compose.yml up -d
4. Open http://localhost:3000 and create project keys
5. Put keys into .env:
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
MSG
