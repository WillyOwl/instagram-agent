"""config.py — Loads and validates all environment configuration for the agent.

All other modules import from here. This is the single source of truth for
settings so that .env changes never require edits in multiple files.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    """Return the value of an env var, raising ValueError if it is missing."""
    value = os.getenv(key)
    if not value:
        raise ValueError(
            f"Required environment variable '{key}' is not set. "
            "Copy .env.example to .env and fill in your credentials."
        )
    return value


# ── Instagram ────────────────────────────────────────────────────────────────
INSTAGRAM_USERNAME: str = _require("INSTAGRAM_USERNAME")
INSTAGRAM_PASSWORD: str = _require("INSTAGRAM_PASSWORD")

# ── Ollama ───────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


# ── Polling ──────────────────────────────────────────────────────────────────
POLL_INTERVAL_SECONDS: int = int(os.getenv("POLL_INTERVAL_SECONDS", "45"))

# ── Data ─────────────────────────────────────────────────────────────────────
HISTORY_PATH: str = os.getenv("HISTORY_PATH", "data/instagram_export/messages/")

# ── Whitelist ────────────────────────────────────────────────────────────────
# Comma-separated Instagram usernames that the agent is allowed to reply to.
# An empty/missing value means NO replies will be sent (fail-safe default).
_raw_whitelist = os.getenv("SENDER_WHITELIST", "")
SENDER_WHITELIST: frozenset[str] = frozenset(
    u.strip().lower() for u in _raw_whitelist.split(",") if u.strip()
)
