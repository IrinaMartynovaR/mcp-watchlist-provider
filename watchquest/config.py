from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def data_dir() -> Path:
    path = Path(os.getenv("WATCHQUEST_DATA_DIR", "./data")).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


DATA_DIR = data_dir()
PROFILE_FILE = DATA_DIR / "profile.json"
WATCHLIST_FILE = DATA_DIR / "watchlist.json"
SOURCES_FILE = DATA_DIR / "sources.json"
CACHE_FILE = DATA_DIR / "cache.json"

FEEDLY_ACCESS_TOKEN = os.getenv("FEEDLY_ACCESS_TOKEN", "").strip()
FEEDLY_STREAM_IDS = [
    item.strip()
    for item in os.getenv("FEEDLY_STREAM_IDS", "").split(",")
    if item.strip()
]

ZAI_API_KEY = os.getenv("ZAI_API_KEY", "").strip()
ZAI_BASE_URL = os.getenv("ZAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4").strip().rstrip("/")
ZAI_MODEL = os.getenv("ZAI_MODEL", "glm-4.7-flash").strip()
ZAI_TIMEOUT_SECONDS = float(os.getenv("ZAI_TIMEOUT_SECONDS", "90"))
