"""agent.py — RAG-powered reply generator using LlamaIndex + Ollama (llama3).

Workflow:
  1. On init, parse your Instagram history and index it with LlamaIndex.
  2. On each call to generate_reply(), retrieve the top-5 most semantically
     similar past exchanges and inject them as style examples into the prompt.
  3. Call Ollama (llama3) with the enriched prompt and return the reply text.

Includes a retry wrapper: if Ollama times out or errors once, it retries
after a short pause before giving up and returning None.
"""

from __future__ import annotations

import os
import logging
import time
from typing import Optional, cast

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama


import config
from history_parser import load_history

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are acting as the user. Your job is to reply to Instagram DMs exactly as \
the user would — matching their tone, vocabulary, language, and response length.

Rules:
- Keep replies short and conversational unless the user's history shows longer replies.
- If the incoming message is in a different language (e.g. Korean, Spanish, \
  Japanese, Chinese, Arabic), reply in THAT same language, matching the style \
  shown in the examples.
- Do not be overly formal. Match the casual register of the examples.
- Never reveal that you are an AI. Reply as if you are the user.
- Do not add greetings or sign-offs unless the examples show them.
- Output ONLY the reply text — no metadata, no labels, no quotes.
"""

_RETRY_DELAY_SECONDS = 3
_TOP_K_EXAMPLES = 5


class Agent:
    """RAG agent that generates Instagram DM replies in the user's style."""

    def __init__(self) -> None:
        logger.info("Initialising agent — loading chat history …")

        # Use Ollama's embedding model so indexing is also fully local.
        embed_model = OllamaEmbedding(
            model_name=config.OLLAMA_EMBED_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )

        self._llm = Ollama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            request_timeout=60.0,
        )

        from llama_index.core import StorageContext, load_index_from_storage

        persist_dir = "./data/index_storage"
        
        is_testing = os.getenv("TESTING") == "true"
        if not is_testing and os.path.exists(persist_dir) and os.listdir(persist_dir):
            logger.info("Loading existing RAG index from local cache '%s'...", persist_dir)
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            self._index = cast(
                VectorStoreIndex,
                load_index_from_storage(
                    storage_context,
                    embed_model=embed_model,
                ),
            )

            logger.info("Agent ready (index loaded from cache).")
        else:
            logger.info("No cached index found. Building RAG index from scratch...")
            # Parse history and build in-memory vector index.
            documents = load_history(config.HISTORY_PATH, config.INSTAGRAM_USERNAME)

            if not documents:
                logger.warning(
                    "No history documents loaded. The agent will still work but "
                    "replies will not be style-matched to your past conversations."
                )

            self._index = VectorStoreIndex.from_documents(
                documents,
                embed_model=embed_model,
                show_progress=True,
            )
            
            logger.info("Persisting RAG index to local cache '%s'...", persist_dir)
            os.makedirs(persist_dir, exist_ok=True)
            self._index.storage_context.persist(persist_dir=persist_dir)
            logger.info("Agent ready. Indexed %d exchange pairs.", len(documents))

        self._retriever = VectorIndexRetriever(
            index=self._index,
            similarity_top_k=_TOP_K_EXAMPLES,
            embed_model=embed_model,
        )


    # ── Public API ────────────────────────────────────────────────────────────

    def generate_reply(
        self,
        incoming_message: str,
        conversation_history: Optional[list[tuple[str, str]]] = None,
    ) -> Optional[str]:
        """Generate a reply to an incoming Instagram DM.

        Args:
            incoming_message: The text of the DM to reply to.
            conversation_history: Optional list of (sender, text) tuples
                representing the recent thread context before this message.

        Returns:
            The generated reply string, or None if generation failed after retry.
        """
        prompt = self._build_prompt(incoming_message, conversation_history or [])

        for attempt in (1, 2):
            try:
                response = self._llm.complete(prompt)
                reply = str(response).strip()
                if reply:
                    logger.info(
                        "Reply generated (attempt %d): %r", attempt, reply[:80]
                    )
                    return reply
                logger.warning("Empty response from Ollama on attempt %d.", attempt)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Ollama call failed on attempt %d: %s", attempt, exc
                )

            if attempt == 1:
                logger.info("Retrying in %d s …", _RETRY_DELAY_SECONDS)
                time.sleep(_RETRY_DELAY_SECONDS)

        logger.error("Failed to generate a reply after 2 attempts.")
        return None

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_prompt(
        self,
        incoming_message: str,
        conversation_history: list[tuple[str, str]],
    ) -> str:
        """Assemble the full prompt with system instructions, RAG examples, and context."""

        # 1. Retrieve similar past exchanges from the vector index.
        retrieved_nodes = self._retriever.retrieve(incoming_message)
        style_examples = "\n\n".join(node.text for node in retrieved_nodes)

        # 2. Format the recent thread context (if provided).
        thread_context = ""
        if conversation_history:
            lines = []
            for sender, text in conversation_history[-6:]:  # last 6 turns max
                label = "Them" if sender != config.INSTAGRAM_USERNAME else "Me"
                lines.append(f"{label}: {text}")
            thread_context = "\n".join(lines)

        # 3. Assemble prompt sections.
        sections: list[str] = [_SYSTEM_PROMPT.strip()]

        if style_examples:
            sections.append(
                "--- STYLE EXAMPLES FROM YOUR PAST CONVERSATIONS ---\n"
                + style_examples
            )

        if thread_context:
            sections.append(
                "--- RECENT CONVERSATION THREAD ---\n" + thread_context
            )

        sections.append(
            "--- NEW MESSAGE TO REPLY TO ---\n"
            f"Them: {incoming_message}\nMe:"
        )

        return "\n\n".join(sections)
