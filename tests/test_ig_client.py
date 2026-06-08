"""tests/test_ig_client.py — Unit tests for ig_client.py.

All instagrapi calls and time.sleep are mocked so tests run offline
and deterministically without touching Instagram's servers.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import config


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_thread(thread_id: str, sender_id: str, text: str) -> MagicMock:
    """Build a minimal fake instagrapi DirectThread object."""
    msg = MagicMock()
    msg.user_id = sender_id
    msg.text = text

    thread = MagicMock()
    thread.id = thread_id
    thread.messages = [msg]
    return thread


# ── IGClient.login ────────────────────────────────────────────────────────────

class TestLogin:
    @patch("ig_client.SESSION_FILE")
    @patch("ig_client.Client")
    def test_fresh_login_dumps_session(self, mock_client_cls: MagicMock, mock_session_file: MagicMock) -> None:
        mock_session_file.exists.return_value = False
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        from ig_client import IGClient
        client = IGClient()
        client.login()

        mock_client.login.assert_called_once_with(
            config.INSTAGRAM_USERNAME, config.INSTAGRAM_PASSWORD
        )
        mock_client.dump_settings.assert_called_once()

    @patch("ig_client.SESSION_FILE")
    @patch("ig_client.Client")
    def test_cached_session_loaded(self, mock_client_cls: MagicMock, mock_session_file: MagicMock) -> None:
        mock_session_file.exists.return_value = True
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        from ig_client import IGClient
        client = IGClient()
        client.login()

        mock_client.load_settings.assert_called_once()
        # dump_settings should NOT be called when session is reused.
        mock_client.dump_settings.assert_not_called()


# ── IGClient.get_unread_messages ──────────────────────────────────────────────

class TestGetUnreadMessages:
    @patch.object(config, "SENDER_WHITELIST", frozenset({"alice"}))
    @patch("ig_client.Client")
    def test_whitelisted_sender_returned(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        thread = _make_thread("t1", "uid_alice", "Hey!")
        mock_client.direct_threads.return_value = [thread]
        mock_client.user_info.return_value = MagicMock(username="alice")

        from ig_client import IGClient
        client = IGClient()
        results = client.get_unread_messages()

        assert len(results) == 1
        assert results[0].username == "alice"
        assert results[0].text == "Hey!"
        assert results[0].thread_id == "t1"

    @patch.object(config, "SENDER_WHITELIST", frozenset({"alice"}))
    @patch("ig_client.Client")
    def test_non_whitelisted_sender_excluded(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        thread = _make_thread("t1", "uid_bob", "Hello")
        mock_client.direct_threads.return_value = [thread]
        mock_client.user_info.return_value = MagicMock(username="bob")

        from ig_client import IGClient
        client = IGClient()
        results = client.get_unread_messages()

        assert results == []

    @patch.object(config, "SENDER_WHITELIST", frozenset())
    @patch("ig_client.Client")
    def test_empty_whitelist_returns_empty(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        from ig_client import IGClient
        client = IGClient()
        results = client.get_unread_messages()

        # Should return early without even calling direct_threads.
        mock_client.direct_threads.assert_not_called()
        assert results == []

    @patch.object(config, "SENDER_WHITELIST", frozenset({"alice"}))
    @patch("ig_client.Client")
    def test_media_only_message_skipped(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        thread = _make_thread("t1", "uid_alice", "")  # empty text = media
        thread.messages[0].text = None
        mock_client.direct_threads.return_value = [thread]
        mock_client.user_info.return_value = MagicMock(username="alice")

        from ig_client import IGClient
        client = IGClient()
        results = client.get_unread_messages()
        assert results == []


# ── IGClient.send_reply ───────────────────────────────────────────────────────

class TestSendReply:
    @patch("ig_client.time.sleep")
    @patch("ig_client.random.uniform", return_value=2.0)
    @patch("ig_client.Client")
    def test_send_reply_calls_direct_send(
        self,
        mock_client_cls: MagicMock,
        mock_uniform: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        from ig_client import IGClient
        client = IGClient()
        client.send_reply("thread_123", "Hello there!")

        mock_sleep.assert_called_once_with(2.0)
        mock_client.direct_send.assert_called_once_with(
            "Hello there!", thread_ids=["thread_123"]
        )

    @patch("ig_client.time.sleep")
    @patch("ig_client.Client")
    def test_send_reply_propagates_client_error(
        self, mock_client_cls: MagicMock, mock_sleep: MagicMock
    ) -> None:
        from instagrapi.exceptions import ClientError

        mock_client = MagicMock()
        mock_client.direct_send.side_effect = ClientError("Rate limited")
        mock_client_cls.return_value = mock_client

        from ig_client import IGClient
        client = IGClient()

        with pytest.raises(ClientError):
            client.send_reply("thread_123", "Hi")
