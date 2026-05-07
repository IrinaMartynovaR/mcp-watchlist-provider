#!/usr/bin/env bash
set -euo pipefail

LANGFUSE_VERSION="${1:-4.5.1}"
TARGET_DIR="vendor/langfuse"
TARGET_FILE="${TARGET_DIR}/docker-compose.yml"
SOURCE_URL="https://raw.githubusercontent.com/langfuse/langfuse/main/docker-compose.yml"

mkdir -p "${TARGET_DIR}"
curl -fsSL "${SOURCE_URL}" -o "${TARGET_FILE}"

python - <<'PY' "${TARGET_FILE}" "${LANGFUSE_VERSION}"
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
version = sys.argv[2]
text = path.read_text()
text = re.sub(r"(image:\s*langfuse/langfuse:)[^\s]+", rf"\g<1>{version}", text)
path.write_text(text)
PY

cat <<MSG
Downloaded official Langfuse developer docker compose stack to ${TARGET_FILE}
Pinned langfuse/langfuse image to version ${LANGFUSE_VERSION}

Next steps:
1. Review ${TARGET_FILE} and adjust secrets for non-local use.
2. Start Langfuse:
   docker compose -f ${TARGET_FILE} up -d
3. Open http://localhost:3000 and create project keys.
4. Put keys into .env:
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
MSG
