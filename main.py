"""main.py — Entry point for the Instagram auto-reply agent.

Run with:
    python main.py

The loop:
  1. Initialises the agent (indexes your chat history once at startup).
  2. Logs in to Instagram (reuses cached session if available).
  3. Every POLL_INTERVAL_SECONDS (± random jitter) checks for new DMs from
     whitelisted senders.
  4. For each eligible message, generates a reply and sends it back.

Press Ctrl+C to stop gracefully.
"""

from __future__ import annotations

import logging
import random
import time

import config  # validates .env on import — fails fast if credentials missing
from agent import Agent
from ig_client import IGClient

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:  # noqa: C901
    logger.info("=== Instagram Auto-Reply Agent starting ===")
    logger.info("Model     : %s @ %s", config.OLLAMA_MODEL, config.OLLAMA_BASE_URL)
    logger.info("Whitelist : %s", sorted(config.SENDER_WHITELIST) or "(empty — no replies will be sent)")
    logger.info("Poll interval: %d s ± 5 s", config.POLL_INTERVAL_SECONDS)

    # Initialise agent (builds RAG index — may take a moment on first run).
    agent = Agent()

    # Initialise Instagram client.
    client = IGClient()
    client.login()

    logger.info("Entering polling loop. Press Ctrl+C to stop.")

    while True:
        try:
            messages = client.get_unread_messages()

            for msg in messages:
                logger.info(
                    "Processing DM from @%s: %r", msg.username, msg.text[:60]
                )

                reply = agent.generate_reply(
                    incoming_message=msg.text,
                    conversation_history=[],  # extend later with thread history
                )

                if reply is None:
                    logger.warning(
                        "No reply generated for @%s — skipping.", msg.username
                    )
                    continue

                client.send_reply(msg.thread_id, reply)

        except KeyboardInterrupt:
            logger.info("Shutdown requested. Exiting …")
            break
        except Exception as exc:  # noqa: BLE001
            # Catch-all so the loop never dies on a transient network error.
            logger.error("Unexpected error in polling loop: %s", exc, exc_info=True)

        # Sleep with ±5 s jitter to avoid predictable request patterns.
        sleep_duration = config.POLL_INTERVAL_SECONDS + random.uniform(-5.0, 5.0)
        logger.debug("Sleeping %.1f s until next poll.", sleep_duration)
        time.sleep(max(sleep_duration, 30.0))  # never sleep less than 30 s


if __name__ == "__main__":
    main()
