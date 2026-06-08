"""tests/test_agent.py — Unit tests for agent.py.

All LlamaIndex index construction, retrieval, and Ollama LLM calls are mocked
so tests run offline and instantaneously.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agent import Agent



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
