"""tests/test_history_parser.py — Unit tests for history_parser.py.

All tests are fully offline: no files are read from disk (fixtures are
constructed in-memory) and no LlamaIndex or Ollama calls are made.
"""

from __future__ import annotations

import json
import os

from history_parser import _fix_encoding, _is_text_message, load_history, parse_thread_file


# ── _fix_encoding ─────────────────────────────────────────────────────────────

class TestFixEncoding:
    def test_ascii_unchanged(self) -> None:
        assert _fix_encoding("Hello world") == "Hello world"

    def test_emoji_corrected(self) -> None:
        # Instagram encodes 😂 as latin-1-escaped UTF-8 bytes.
        raw = "\xf0\x9f\x98\x82"  # 😂 as latin-1 escape
        assert _fix_encoding(raw) == "😂"

    def test_korean_corrected(self) -> None:
        # 안녕 in latin-1-escaped UTF-8
        raw = "\xec\x95\x88\xeb\x85\x95"
        assert _fix_encoding(raw) == "안녕"

    def test_arabic_corrected(self) -> None:
        raw = "\xd9\x85\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7"
        assert _fix_encoding(raw) == "مرحبا"

    def test_already_unicode_passthrough(self) -> None:
        text = "日本語テスト"
        assert _fix_encoding(text) == text

    def test_invalid_sequence_passthrough(self) -> None:
        # Should not raise — returns original string.
        assert _fix_encoding("\xff\xfe") == "\xff\xfe"


# ── _is_text_message ──────────────────────────────────────────────────────────

class TestIsTextMessage:
    def test_valid_text_message(self) -> None:
        msg = {"sender_name": "Alice", "content": "Hey!", "is_unsent": False}
        assert _is_text_message(msg) is True

    def test_unsent_message_excluded(self) -> None:
        msg = {"sender_name": "Alice", "content": "oops", "is_unsent": True}
        assert _is_text_message(msg) is False

    def test_media_only_excluded(self) -> None:
        msg = {"sender_name": "Alice", "share": {"link": "https://example.com"}}
        assert _is_text_message(msg) is False

    def test_reaction_only_excluded(self) -> None:
        msg = {"sender_name": "Alice", "reactions": [{"reaction": "❤️"}]}
        assert _is_text_message(msg) is False

    def test_content_key_absent_excluded(self) -> None:
        assert _is_text_message({"sender_name": "Alice"}) is False


# ── parse_thread_file ─────────────────────────────────────────────────────────

def _write_thread_file(directory: str, messages: list[dict]) -> str:
    """Write a minimal Instagram message_1.json fixture."""
    data = {
        "participants": [{"name": "Me"}, {"name": "Alice"}],
        "messages": messages,
    }
    path = os.path.join(directory, "message_1.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    return path


class TestParseThreadFile:
    def test_basic_exchange(self, tmp_path: object) -> None:
        # Messages stored newest-first as Instagram does.
        msgs = [
            {"sender_name": "Me", "content": "Sure!", "is_unsent": False, "timestamp_ms": 2000},
            {"sender_name": "Alice", "content": "You free?", "is_unsent": False, "timestamp_ms": 1000},
        ]
        path = _write_thread_file(str(tmp_path), msgs)
        pairs = parse_thread_file(path, your_ig_name="Me")
        assert pairs == [("You free?", "Sure!")]

    def test_media_message_skipped(self, tmp_path: object) -> None:
        msgs = [
            {"sender_name": "Me", "content": "Haha", "is_unsent": False, "timestamp_ms": 3000},
            {"sender_name": "Alice", "timestamp_ms": 2000},   # media — no content
            {"sender_name": "Alice", "content": "Check this out", "is_unsent": False, "timestamp_ms": 1000},
        ]
        path = _write_thread_file(str(tmp_path), msgs)
        pairs = parse_thread_file(path, your_ig_name="Me")
        assert ("Check this out", "Haha") in pairs

    def test_unsent_message_skipped(self, tmp_path: object) -> None:
        msgs = [
            {"sender_name": "Me", "content": "Hey", "is_unsent": False, "timestamp_ms": 2000},
            {"sender_name": "Alice", "content": "deleted msg", "is_unsent": True, "timestamp_ms": 1000},
        ]
        path = _write_thread_file(str(tmp_path), msgs)
        pairs = parse_thread_file(path, your_ig_name="Me")
        assert pairs == []  # unsent incoming — no valid pair

    def test_multi_language_exchange(self, tmp_path: object) -> None:
        # Korean message — content stored with latin-1-escaped UTF-8.
        korean_raw = "\xec\x95\x88\xeb\x85\x95"  # 안녕
        reply_raw = "\xec\xa2\x8b\xec\x95\x84"   # 좋아
        msgs = [
            {"sender_name": "Me", "content": reply_raw, "is_unsent": False, "timestamp_ms": 2000},
            {"sender_name": "Alice", "content": korean_raw, "is_unsent": False, "timestamp_ms": 1000},
        ]
        path = _write_thread_file(str(tmp_path), msgs)
        pairs = parse_thread_file(path, your_ig_name="Me")
        assert pairs == [("안녕", "좋아")]

    def test_no_reply_produces_no_pair(self, tmp_path: object) -> None:
        msgs = [
            {"sender_name": "Alice", "content": "Hello?", "is_unsent": False, "timestamp_ms": 1000},
        ]
        path = _write_thread_file(str(tmp_path), msgs)
        pairs = parse_thread_file(path, your_ig_name="Me")
        assert pairs == []

    def test_invalid_json_returns_empty(self, tmp_path: object) -> None:
        bad_path = os.path.join(str(tmp_path), "bad.json")
        with open(bad_path, "w") as fh:
            fh.write("not json {{{")
        # load_history wraps parse_thread_file and catches JSONDecodeError.
        docs = load_history(str(tmp_path), "Me")
        assert docs == []


# ── load_history ──────────────────────────────────────────────────────────────

class TestLoadHistory:
    def test_empty_directory_returns_empty(self, tmp_path: object) -> None:
        docs = load_history(str(tmp_path), "Me")
        assert docs == []

    def test_documents_created_correctly(self, tmp_path: object) -> None:
        inbox = os.path.join(str(tmp_path), "inbox", "thread1")
        os.makedirs(inbox)
        msgs = [
            {"sender_name": "Me", "content": "Yep", "is_unsent": False, "timestamp_ms": 2000},
            {"sender_name": "Alice", "content": "Coming?", "is_unsent": False, "timestamp_ms": 1000},
        ]
        _write_thread_file(inbox, msgs)

        docs = load_history(str(tmp_path), "Me")
        assert len(docs) == 1
        assert "User: Coming?" in docs[0].text
        assert "Me: Yep" in docs[0].text
