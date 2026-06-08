"""ig_client.py — Instagram connector using instagrapi.

Wraps instagrapi.Client with:
  - Session caching (avoids repeated logins that can trigger flags)
  - Whitelist enforcement (only interacts with approved senders)
  - Human-like send delay jitter (random 1 - 3 s before each reply)
  - Structured return types so the rest of the codebase never imports instagrapi directly

Public API:
    client = IGClient()
    client.login()
    for item in client.get_unread_messages():
        # item.thread_id, item.user_id, item.username, item.text
        client.send_reply(item.thread_id, "Hello!")
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

from instagrapi import Client
from instagrapi.exceptions import ClientError


import config

logger = logging.getLogger(__name__)

SESSION_FILE = Path(".instagram_session.json")


@dataclass(frozen=True)
class IncomingMessage:
    """A single unread DM eligible for a reply."""

    thread_id: str
    user_id: str
    username: str  # lowercased @username of the sender
    text: str


class IGClient:
    """Thin wrapper around instagrapi.Client with safety and whitelist logic."""

    def __init__(self) -> None:
        self._client = Client()
        # Set a realistic request delay to mimic human usage.
        self._client.delay_range = [1, 3]

    # ── Auth ──────────────────────────────────────────────────────────────────

    def login(self) -> None:
        """Authenticate with Instagram, reusing a cached session if possible.

        Session caching significantly reduces the number of login events, which
        is the main trigger for Instagram's automated account flag system.
        """
        if SESSION_FILE.exists():
            logger.info("Loading cached session from %s", SESSION_FILE)
            try:
                self._client.load_settings(str(SESSION_FILE))
                # Verify that the loaded session is still valid
                self._client.get_timeline_feed(amount=1)
                logger.info("Session restored successfully.")
                return
            except Exception as exc:
                logger.warning("Cached session invalid (%s) — logging in fresh.", exc)
                SESSION_FILE.unlink(missing_ok=True)

        logger.info("Logging in as %s …", config.INSTAGRAM_USERNAME)
        try:
            self._client.login(config.INSTAGRAM_USERNAME, config.INSTAGRAM_PASSWORD)
        except ClientError as exc:
            # Sometimes post-login flow steps (like reels_tray) fail with 467/400,
            # but the actual login succeeded and self._client.user_id is populated.
            if self._client.user_id:
                logger.warning(
                    "Login flow encountered a non-fatal step error (%s) "
                    "but session is active. Proceeding...",
                    exc,
                )
            else:
                raise

        self._client.dump_settings(str(SESSION_FILE))
        logger.info("Login successful. Session saved to %s", SESSION_FILE)

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_unread_messages(self) -> list[IncomingMessage]:
        """Return all unread DMs from whitelisted senders.

        Threads are fetched from the pending and primary inbox.  Only the most
        recent unread message per thread is returned (to avoid double-replying
        to a burst of messages in rapid succession).

        Returns:
            List of IncomingMessage dataclasses, one per eligible thread.
        """
        if not config.SENDER_WHITELIST:
            logger.warning(
                "SENDER_WHITELIST is empty — no replies will be sent. "
                "Add at least one username to .env SENDER_WHITELIST."
            )
            return []

        results: list[IncomingMessage] = []

        try:
            threads = self._client.direct_threads(amount=20, selected_filter="unread")
        except ClientError as exc:
            logger.error("Failed to fetch inbox: %s", exc)
            return []

        for thread in threads:
            if not thread.messages:
                continue

            # The most recent message is index 0.
            latest = thread.messages[0]
            if not latest.text:
                # Media / sticker — skip.
                continue

            # Resolve the sender's username.
            try:
                sender_info = self._client.user_info(latest.user_id)
                sender_username = sender_info.username.lower()
            except ClientError as exc:
                logger.warning("Could not resolve user %s: %s", latest.user_id, exc)
                continue

            if sender_username not in config.SENDER_WHITELIST:
                logger.debug(
                    "Skipping message from @%s (not in whitelist).", sender_username
                )
                continue

            results.append(
                IncomingMessage(
                    thread_id=str(thread.id),
                    user_id=str(latest.user_id),
                    username=sender_username,
                    text=latest.text,
                )
            )

        logger.info(
            "Found %d unread message(s) from whitelisted senders.", len(results)
        )
        return results

    # ── Write ─────────────────────────────────────────────────────────────────

    def send_reply(self, thread_id: str, text: str) -> None:
        """Send a text reply to a DM thread with a human-like delay.

        Args:
            thread_id: The Instagram thread ID to reply to.
            text: The reply text generated by the agent.
        """
        # Human-like jitter: wait 1–3 seconds before sending.
        delay = random.uniform(1.0, 3.0)
        logger.debug("Waiting %.1f s before sending reply (jitter).", delay)
        time.sleep(delay)

        try:
            self._client.direct_send(text, thread_ids=[thread_id])
            logger.info("Reply sent to thread %s.", thread_id)
        except ClientError as exc:
            logger.error("Failed to send reply to thread %s: %s", thread_id, exc)
            raise
