"""tests/test_agent.py — Unit tests for agent.py.

All LlamaIndex index construction, retrieval, and Ollama LLM calls are mocked
so tests run offline and instantaneously.
"""

from __future__ import annotations

import os
import pytest
from unittest.mock import MagicMock, patch

from agent import Agent


@pytest.fixture(autouse=True)
def reset_config_for_testing():
    with (
        patch("config.ACTIVE_MODEL", None),
        patch("config.ACTIVE_EMBED_MODEL", None),
        patch("config.ACTIVE_API_BASE", None),
        patch("config.FIXED_MODEL_PROVIDER", "ollama"),
    ):
        yield


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_node(text: str) -> MagicMock:
    node = MagicMock()
    node.text = text
    return node


# ── Agent.__init__ ────────────────────────────────────────────────────────────

class TestAgentInit:
    @patch("agent.load_history", return_value=[])
    @patch("agent.VectorStoreIndex")
    @patch("agent.VectorIndexRetriever")
    @patch("agent.OllamaEmbedding")
    @patch("agent.Ollama")
    def test_init_with_empty_history(
        self,
        mock_ollama: MagicMock,
        mock_embed: MagicMock,
        mock_retriever_cls: MagicMock,
        mock_index_cls: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        """Agent should initialise without error even when history is empty."""
        from agent import Agent
        _ = Agent()
        mock_load.assert_called_once()
        mock_index_cls.from_documents.assert_called_once()

    @patch("agent.load_history")
    @patch("agent.VectorStoreIndex")
    @patch("agent.VectorIndexRetriever")
    @patch("agent.OllamaEmbedding")
    @patch("agent.Ollama")
    def test_init_indexes_all_documents(
        self,
        mock_ollama: MagicMock,
        mock_embed: MagicMock,
        mock_retriever_cls: MagicMock,
        mock_index_cls: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        fake_docs = [MagicMock(), MagicMock(), MagicMock()]
        mock_load.return_value = fake_docs

        from agent import Agent
        _ = Agent()

        args, kwargs = mock_index_cls.from_documents.call_args
        assert args[0] == fake_docs


# ── Agent.generate_reply ──────────────────────────────────────────────────────

class TestGenerateReply:
    def _build_agent(
        self,
        retrieved_nodes: list,
        llm_response: str = "Sure!",
        llm_raises: Exception | None = None,
    ) -> "Agent":  # noqa: F821
        """Construct an Agent with all external dependencies mocked."""
        with (
            patch("agent.os.path.exists", return_value=False),
            patch("agent.load_history", return_value=[]),
            patch("agent.VectorStoreIndex"),
            patch("agent.VectorIndexRetriever") as mock_retriever_cls,
            patch("agent.OllamaEmbedding"),
            patch("agent.Ollama") as mock_ollama_cls,
        ):
            mock_retriever = MagicMock()
            mock_retriever.retrieve.return_value = retrieved_nodes
            mock_retriever_cls.return_value = mock_retriever

            mock_llm = MagicMock()
            if llm_raises:
                mock_llm.complete.side_effect = [llm_raises, llm_response]
            else:
                mock_llm.complete.return_value = llm_response
            mock_ollama_cls.return_value = mock_llm

            from agent import Agent
            agent = Agent()
            # Inject mocks directly so they persist outside the `with` block.
            agent._retriever = mock_retriever
            agent._llm = mock_llm
            return agent

    def test_successful_reply(self) -> None:
        nodes = [_make_node("User: Hey\nMe: What's up")]
        agent = self._build_agent(nodes, llm_response="All good!")
        result = agent.generate_reply("Hey")
        assert result == "All good!"

    def test_reply_strips_whitespace(self) -> None:
        nodes: list[MagicMock] = []
        agent = self._build_agent(nodes, llm_response="  Yep.  \n")
        result = agent.generate_reply("You free?")
        assert result == "Yep."

    def test_retry_on_first_failure_succeeds(self) -> None:
        """If the first Ollama call fails, the agent should retry once."""
        with (
            patch("agent.load_history", return_value=[]),
            patch("agent.VectorStoreIndex"),
            patch("agent.VectorIndexRetriever") as mock_retriever_cls,
            patch("agent.OllamaEmbedding"),
            patch("agent.Ollama") as mock_ollama_cls,
            patch("agent.time.sleep"),  # speed up the test
        ):
            mock_retriever = MagicMock()
            mock_retriever.retrieve.return_value = []
            mock_retriever_cls.return_value = mock_retriever

            mock_llm = MagicMock()
            mock_llm.complete.side_effect = [RuntimeError("timeout"), "Retry reply"]
            mock_ollama_cls.return_value = mock_llm

            from agent import Agent
            agent = Agent()
            agent._retriever = mock_retriever
            agent._llm = mock_llm

            result = agent.generate_reply("Hi")
            assert result == "Retry reply"
            assert mock_llm.complete.call_count == 2

    def test_two_failures_returns_none(self) -> None:
        """If both attempts fail, generate_reply should return None."""
        with (
            patch("agent.load_history", return_value=[]),
            patch("agent.VectorStoreIndex"),
            patch("agent.VectorIndexRetriever") as mock_retriever_cls,
            patch("agent.OllamaEmbedding"),
            patch("agent.Ollama") as mock_ollama_cls,
            patch("agent.time.sleep"),
        ):
            mock_retriever = MagicMock()
            mock_retriever.retrieve.return_value = []
            mock_retriever_cls.return_value = mock_retriever

            mock_llm = MagicMock()
            mock_llm.complete.side_effect = RuntimeError("persistent error")
            mock_ollama_cls.return_value = mock_llm

            from agent import Agent
            agent = Agent()
            agent._retriever = mock_retriever
            agent._llm = mock_llm

            result = agent.generate_reply("Hey")
            assert result is None

    def test_rag_examples_included_in_prompt(self) -> None:
        """Retrieved node text should appear in the prompt sent to the LLM."""
        example_text = "User: Yo\nMe: What's good"
        nodes = [_make_node(example_text)]

        with (
            patch("agent.load_history", return_value=[]),
            patch("agent.VectorStoreIndex"),
            patch("agent.VectorIndexRetriever") as mock_retriever_cls,
            patch("agent.OllamaEmbedding"),
            patch("agent.Ollama") as mock_ollama_cls,
        ):
            mock_retriever = MagicMock()
            mock_retriever.retrieve.return_value = nodes
            mock_retriever_cls.return_value = mock_retriever

            mock_llm = MagicMock()
            mock_llm.complete.return_value = "Cool"
            mock_ollama_cls.return_value = mock_llm

            from agent import Agent
            agent = Agent()
            agent._retriever = mock_retriever
            agent._llm = mock_llm

            agent.generate_reply("Yo")

            prompt_arg = mock_llm.complete.call_args[0][0]
            assert example_text in prompt_arg

    def test_incoming_language_preserved_in_prompt(self) -> None:
        """Non-English incoming message should appear verbatim in the prompt."""
        nodes: list[MagicMock] = []
        agent = self._build_agent(nodes, llm_response="응")
        with patch.object(agent._llm, "complete", return_value="응") as mock_complete:
            agent.generate_reply("안녕하세요")
            prompt = mock_complete.call_args[0][0]
            assert "안녕하세요" in prompt


# ── Configurable-Model & Provider Tests ──────────────────────────────────────

class TestAgentInitConfigurableModel:
    @patch("agent.load_history", return_value=[])
    @patch("agent.VectorStoreIndex")
    @patch("agent.VectorIndexRetriever")
    @patch("agent.OllamaEmbedding")
    def test_raises_if_api_key_missing(
        self,
        mock_embed: MagicMock,
        mock_retriever_cls: MagicMock,
        mock_index_cls: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        """EnvironmentError should be raised if the required API key for ACTIVE_MODEL is missing."""
        temp_env = os.environ.copy()
        temp_env.pop("ANTHROPIC_API_KEY", None)
        with (
            patch("config.ACTIVE_MODEL", "anthropic/claude-3-5-sonnet"),
            patch.dict(os.environ, temp_env, clear=True),
        ):
            from agent import Agent
            with pytest.raises(EnvironmentError) as exc_info:
                Agent()
            assert "Required API key 'ANTHROPIC_API_KEY'" in str(exc_info.value)

    @patch("agent.load_history", return_value=[])
    @patch("agent.VectorStoreIndex")
    @patch("agent.VectorIndexRetriever")
    @patch("agent.OllamaEmbedding")
    def test_init_succeeds_with_key_set(
        self,
        mock_embed: MagicMock,
        mock_retriever_cls: MagicMock,
        mock_index_cls: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        with (
            patch("config.ACTIVE_MODEL", "anthropic/claude-3-5-sonnet"),
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake-key"}),
        ):
            from agent import Agent
            agent = Agent()
            assert agent._active_model == "anthropic/claude-3-5-sonnet"
            assert agent._llm is None

    @patch("agent.load_history", return_value=[])
    @patch("agent.VectorStoreIndex")
    @patch("agent.VectorIndexRetriever")
    @patch("agent.OllamaEmbedding")
    def test_xiaomi_raises_if_key_missing(
        self,
        mock_embed: MagicMock,
        mock_retriever_cls: MagicMock,
        mock_index_cls: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        temp_env = os.environ.copy()
        temp_env.pop("XIAOMI_MIMO_API_KEY", None)
        with (
            patch("config.ACTIVE_MODEL", "xiaomi_mimo/mimo-v2.5"),
            patch.dict(os.environ, temp_env, clear=True),
        ):
            from agent import Agent
            with pytest.raises(EnvironmentError) as exc_info:
                Agent()
            assert "Required API key 'XIAOMI_MIMO_API_KEY'" in str(exc_info.value)

    @patch("agent.load_history", return_value=[])
    @patch("agent.VectorStoreIndex")
    @patch("agent.VectorIndexRetriever")
    @patch("agent.OllamaEmbedding")
    @patch("llama_index.embeddings.litellm.LiteLLMEmbedding")
    def test_init_succeeds_with_configurable_embed(
        self,
        mock_litellm_embed: MagicMock,
        mock_embed: MagicMock,
        mock_retriever_cls: MagicMock,
        mock_index_cls: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        with (
            patch("config.ACTIVE_MODEL", "gpt-4"),
            patch("config.ACTIVE_EMBED_MODEL", "text-embedding-3-small"),
            patch.dict(os.environ, {"OPENAI_API_KEY": "fake-openai-key"}),
        ):
            from agent import Agent
            agent = Agent()
            assert agent._active_model == "gpt-4"
            mock_litellm_embed.assert_called_once_with(model_name="text-embedding-3-small")

    @patch("agent.load_history", return_value=[])
    @patch("agent.VectorStoreIndex")
    @patch("agent.VectorIndexRetriever")
    @patch("agent.OllamaEmbedding")
    def test_init_raises_if_embed_key_missing(
        self,
        mock_embed: MagicMock,
        mock_retriever_cls: MagicMock,
        mock_index_cls: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        temp_env = os.environ.copy()
        temp_env.pop("OPENAI_API_KEY", None)
        temp_env["GEMINI_API_KEY"] = "fake-gemini-key"
        with (
            patch("config.ACTIVE_MODEL", "gemini/gemini-pro"),
            patch("config.ACTIVE_EMBED_MODEL", "text-embedding-3-small"),
            patch.dict(os.environ, temp_env, clear=True),
        ):
            from agent import Agent
            with pytest.raises(EnvironmentError) as exc_info:
                Agent()
            assert "Required API key 'OPENAI_API_KEY' for embedding model" in str(exc_info.value)


class TestGenerateReplyConfigurableModel:
    def _build_agent(
        self,
        retrieved_nodes: list,
    ) -> "Agent":
        with (
            patch("agent.os.path.exists", return_value=False),
            patch("agent.load_history", return_value=[]),
            patch("agent.VectorStoreIndex"),
            patch("agent.VectorIndexRetriever") as mock_retriever_cls,
            patch("agent.OllamaEmbedding"),
        ):
            mock_retriever = MagicMock()
            mock_retriever.retrieve.return_value = retrieved_nodes
            mock_retriever_cls.return_value = mock_retriever

            with (
                patch("config.ACTIVE_MODEL", "xiaomi_mimo/mimo-v2.5"),
                patch.dict(os.environ, {"XIAOMI_MIMO_API_KEY": "fake-key"}),
            ):
                from agent import Agent
                agent = Agent()
                agent._retriever = mock_retriever
                return agent

    @patch("litellm.completion")
    def test_litellm_called_not_ollama(
        self,
        mock_completion: MagicMock,
    ) -> None:
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Xiaomi reply!"
        mock_completion.return_value = mock_response

        agent = self._build_agent([])
        result = agent.generate_reply("Hi")
        assert result == "Xiaomi reply!"
        mock_completion.assert_called_once()
        assert agent._llm is None

    @patch("agent.time.sleep")
    @patch("litellm.completion")
    def test_retry_on_litellm_failure(
        self,
        mock_completion: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Retry reply!"
        mock_completion.side_effect = [RuntimeError("LiteLLM timeout"), mock_response]

        agent = self._build_agent([])
        result = agent.generate_reply("Hi")
        assert result == "Retry reply!"
        assert mock_completion.call_count == 2
        mock_sleep.assert_called_once()

    @patch("agent.time.sleep")
    @patch("litellm.completion")
    def test_two_litellm_failures_returns_none(
        self,
        mock_completion: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        mock_completion.side_effect = RuntimeError("persistent LiteLLM error")

        agent = self._build_agent([])
        result = agent.generate_reply("Hi")
        assert result is None
        assert mock_completion.call_count == 2


class TestAgentInitFixedModelProvider:
    @patch("agent.load_history", return_value=[])
    @patch("agent.VectorStoreIndex")
    @patch("agent.VectorIndexRetriever")
    @patch("agent.OllamaEmbedding")
    @patch("agent.Ollama")
    def test_unknown_provider_raises(
        self,
        mock_ollama: MagicMock,
        mock_embed: MagicMock,
        mock_retriever_cls: MagicMock,
        mock_index_cls: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        with (
            patch("config.ACTIVE_MODEL", None),
            patch("config.FIXED_MODEL_PROVIDER", "unknown-provider"),
        ):
            from agent import Agent
            with pytest.raises(NotImplementedError) as exc_info:
                Agent()
            assert "Fixed-model provider 'unknown-provider' is not yet implemented" in str(exc_info.value)

    @patch("agent.load_history", return_value=[])
    @patch("agent.VectorStoreIndex")
    @patch("agent.VectorIndexRetriever")
    @patch("agent.OllamaEmbedding")
    @patch("agent.Ollama")
    def test_ollama_provider_is_default(
        self,
        mock_ollama: MagicMock,
        mock_embed: MagicMock,
        mock_retriever_cls: MagicMock,
        mock_index_cls: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        with (
            patch("config.ACTIVE_MODEL", None),
            patch("config.FIXED_MODEL_PROVIDER", "ollama"),
        ):
            from agent import Agent
            agent = Agent()
            assert agent._active_model is None
            mock_ollama.assert_called_once()

