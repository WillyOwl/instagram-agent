"""history_parser.py — Parses Instagram JSON chat history into LlamaIndex Documents.

Instagram's "Download Your Information" export puts each thread in:
    messages/inbox/<thread_name>/message_1.json  (and message_2.json if large)

Each file has the structure:
    {
      "participants": [{"name": "Alice"}, {"name": "You"}],
      "messages": [
        {
          "sender_name": "Alice",
          "timestamp_ms": 1700000000000,
          "content": "Hey!",          # text messages only
          "is_unsent": false,
          "reactions": [...]           # optional
        },
        ...
      ]
    }

Instagram encodes text with latin-1-escaped UTF-8, so non-ASCII characters
(emoji, CJK, Arabic, accented letters, etc.) must be re-encoded.  This module
handles that automatically, giving correct multi-language output.

Output format for each exchange stored in a Document:
    User: <incoming message>
    Me: <your reply>
"""

from __future__ import annotations

import glob
import json
import logging
import os
from typing import Any

from llama_index.core import Document

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fix_encoding(text: str) -> str:
    """Re-encode Instagram's latin-1-escaped UTF-8 text to proper Unicode.

    Instagram serialises text as UTF-8 bytes escaped as latin-1 code points.
    Decoding the latin-1 bytes back to UTF-8 recovers the original characters,
    including emoji, CJK, Arabic, Cyrillic, accented characters, etc.
    """
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        # Already valid unicode — return as-is.
        return text


def _is_text_message(msg: dict[str, Any]) -> bool:
    """Return True only for plain-text, non-unsent, non-reaction-only messages."""
    if msg.get("is_unsent", False):
        return False
    if "content" not in msg:
        # Media-only, sticker, audio, video share — skip.
        return False
    return True


def _get_my_username(data: dict[str, Any], your_ig_name: str) -> str:
    """Determine which participant name corresponds to 'you'.

    Instagram exports use the *display name* (not the @username) as sender_name.
    We match against the Instagram username set in config, falling back to the
    first participant not found as a message sender if the name doesn't match.
    """
    participants = {p["name"] for p in data.get("participants", [])}
    # Direct match — Instagram sometimes stores just the first name.
    for p in participants:
        if your_ig_name.lower() in p.lower() or p.lower() in your_ig_name.lower():
            return p
    # Fallback: the participant who sent the most messages is most likely you
    # (since you're extracting your *own* history).
    sender_counts: dict[str, int] = {}
    for msg in data.get("messages", []):
        name = msg.get("sender_name", "")
        sender_counts[name] = sender_counts.get(name, 0) + 1
    if sender_counts:
        return max(sender_counts, key=lambda k: sender_counts[k])
    return ""


# ── Main parser ───────────────────────────────────────────────────────────────

def parse_thread_file(
    filepath: str,
    your_ig_name: str,
) -> list[tuple[str, str]]:
    """Parse a single message_N.json file into (user_message, my_reply) pairs.

    Messages arrive newest-first in the JSON, so we reverse them to get
    chronological order before pairing.

    Args:
        filepath: Absolute or relative path to message_N.json.
        your_ig_name: Your Instagram username (used to identify your messages).

    Returns:
        List of (incoming_message, your_reply) string tuples.
    """
    with open(filepath, encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)

    my_name = _get_my_username(data, your_ig_name)
    if not my_name:
        logger.warning("Could not identify your messages in %s — skipping.", filepath)
        return []

    # Instagram stores messages newest-first; reverse for chronological order.
    messages = list(reversed(data.get("messages", [])))

    pairs: list[tuple[str, str]] = []
    i = 0
    while i < len(messages) - 1:
        msg = messages[i]
        if not _is_text_message(msg):
            i += 1
            continue

        sender = _fix_encoding(msg.get("sender_name", ""))
        content = _fix_encoding(msg.get("content", "")).strip()

        if sender == my_name or not content:
            i += 1
            continue

        # Look ahead for your reply (next text message you sent).
        j = i + 1
        while j < len(messages):
            next_msg = messages[j]
            if not _is_text_message(next_msg):
                j += 1
                continue
            next_sender = _fix_encoding(next_msg.get("sender_name", ""))
            next_content = _fix_encoding(next_msg.get("content", "")).strip()
            if next_sender == my_name and next_content:
                pairs.append((content, next_content))
                i = j + 1  # advance past the reply we just consumed
                break
            # Another message from the same person or unknown — keep looking.
            j += 1
        else:
            # No reply found for this incoming message.
            i += 1

    return pairs


def load_history(history_path: str, your_ig_name: str) -> list[Document]:
    """Walk all thread directories under history_path and build LlamaIndex Documents.

    Each Document contains one (user_message, your_reply) exchange formatted as:
        User: <message>
        Me: <reply>

    Args:
        history_path: Path to the messages/ folder from the Instagram export.
        your_ig_name: Your Instagram username.

    Returns:
        List of LlamaIndex Document objects ready for VectorStoreIndex.
    """
    pattern = os.path.join(history_path, "inbox", "**", "message_*.json")
    files = glob.glob(pattern, recursive=True)

    if not files:
        logger.warning(
            "No message JSON files found at '%s'. "
            "Make sure HISTORY_PATH points to the messages/ directory.",
            history_path,
        )
        return []

    documents: list[Document] = []
    total_pairs = 0

    for filepath in sorted(files):
        try:
            pairs = parse_thread_file(filepath, your_ig_name)
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            logger.error("Failed to parse %s: %s", filepath, exc)
            continue

        for user_msg, my_reply in pairs:
            text = f"User: {user_msg}\nMe: {my_reply}"
            documents.append(
                Document(
                    text=text,
                    metadata={"source_file": os.path.basename(filepath)},
                )
            )
        total_pairs += len(pairs)

    logger.info(
        "Loaded %d exchange pairs from %d files in '%s'.",
        total_pairs,
        len(files),
        history_path,
    )
    return documents
