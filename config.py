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


# ── Fixed-Model pathway ──────────────────────────────────────────────────────
# Selects the fixed-model backend. Currently only 'ollama' is implemented.
# Add new backends here as a future extension point.
FIXED_MODEL_PROVIDER: str = os.getenv("FIXED_MODEL_PROVIDER", "ollama")

# ── Configurable-Model pathway (LiteLLM) ─────────────────────────────────────
ACTIVE_MODEL: str | None = os.getenv("ACTIVE_MODEL") or None
ACTIVE_EMBED_MODEL: str | None = os.getenv("ACTIVE_EMBED_MODEL") or None
ACTIVE_API_BASE: str | None = os.getenv("ACTIVE_API_BASE") or None


# Maps model name prefixes → the env var that must hold the provider's API key.
# Used by agent.py for startup validation.
# Providers requiring no key (e.g. local Ollama) are intentionally omitted.
_PROVIDER_KEY_MAP: dict[str, str] = {
    "gpt":            "OPENAI_API_KEY",
    "openai/":        "OPENAI_API_KEY",
    "text-embedding-": "OPENAI_API_KEY",
    "anthropic/":     "ANTHROPIC_API_KEY",
    "gemini/":        "GEMINI_API_KEY",
    "xiaomi_mimo/":   "XIAOMI_MIMO_API_KEY",
    # Add more provider prefixes here as new providers are onboarded.
}


def get_required_key_for_model(model: str) -> str | None:
    """Return the env var name required for a given model string, or None."""
    for prefix, key_var in _PROVIDER_KEY_MAP.items():
        if model.startswith(prefix):
            return key_var
    return None

