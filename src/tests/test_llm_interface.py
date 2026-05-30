"""
Tests for Module 7 — LLMInterface (src/generation/llm_interface.py)

All tests are pure unit tests — no live OpenAI or Ollama calls are made.
OpenAI and Ollama clients are replaced with MagicMock objects injected
directly onto the instance after construction (avoids __init__ side-effects
from env var loading).

Compatibility notes (real models.py)
--------------------------------------
- RetrievalMode members: GRAPH, VECTOR, HYBRID, WEB  (UPPERCASE)
- Citation.url is Optional[str]
- MedicalAnswer.confidence is float, ge=0, le=1
- Chunk.score is float (non-optional, default 0.0)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from src.generation.llm_interface import (
    LLMInterface,
    _FALLBACK_ANSWER_TEXT,
    _FALLBACK_DISCLAIMER,
)
from src.utils.models import (
    Citation,
    Chunk,
    GraphEdge,
    GraphNode,
    GraphSubgraph,
    MedicalAnswer,
    NodeType,
    EdgeType,
    RetrievalMode,
)


# ---------------------------------------------------------------------------
# Dynamic enum resolution (robust to any member naming convention)
# ---------------------------------------------------------------------------

def _find_node_type(*values: str) -> NodeType:
    lower_vals = {v.lower() for v in values}
    for m in NodeType:
        if m.value.lower() in lower_vals:
            return m
    return list(NodeType)[0]


def _find_edge_type(*values: str) -> EdgeType:
    lower_vals = {v.lower() for v in values}
    for m in EdgeType:
        if m.value.lower() in lower_vals:
            return m
    return list(EdgeType)[0]


_NT_DRUG = _find_node_type("Drug")
_NT_DISEASE = _find_node_type("Disease")
_ET_TREATS = _find_edge_type("TREATS")


# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

GOOD_JSON = {
    "answer": "Metformin is the first-line treatment for Type 2 Diabetes.",
    "citations": [
        {
            "citation_id": "c1",
            "title": "Metformin treatment study",
            "url": "https://pubmed.ncbi.nlm.nih.gov/12345/",
            "source": "pubmed",
            "year": 2023,
            "authors": ["Smith J", "Jones A"],
            "relevance_score": 0.91,
        }
    ],
    "graph_path": ["Type2Diabetes", "Metformin"],
    "confidence": 0.75,
}


def _make_mock_openai_response(content: str) -> MagicMock:
    """Return a mock that mimics openai.ChatCompletion response."""
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = content
    return mock_resp


def _make_subgraph(path_confidence: float = 0.72) -> GraphSubgraph:
    drug = GraphNode(
        id="C001",
        name="Metformin",
        node_type=_NT_DRUG,
        last_updated=datetime.now(tz=timezone.utc),
        confidence_score=0.9,
    )
    disease = GraphNode(
        id="C002",
        name="Type 2 Diabetes",
        node_type=_NT_DISEASE,
        last_updated=datetime.now(tz=timezone.utc),
        confidence_score=0.95,
    )
    edge = GraphEdge(
        source_id="C001",
        target_id="C002",
        edge_type=_ET_TREATS,
        confidence=path_confidence,
    )
    return GraphSubgraph(
        nodes=[drug, disease],
        edges=[edge],
        query_node_ids=["C002"],
        path_confidence=path_confidence,
    )


class _FakeRetrievalResult:
    """Lightweight stand-in for RetrievalResult (avoids importing orchestrator)."""

    def __init__(
        self,
        retrieval_mode: RetrievalMode = RetrievalMode.HYBRID,
        graph_subgraph: Optional[GraphSubgraph] = None,
    ) -> None:
        self.retrieval_mode = retrieval_mode
        self.graph_subgraph = graph_subgraph


@pytest.fixture()
def llm() -> LLMInterface:
    """LLMInterface with no real API keys — tests inject mock clients."""
    return LLMInterface(openai_api_key="", openai_model="gpt-4o")


@pytest.fixture()
def llm_with_key() -> LLMInterface:
    """LLMInterface with a fake-but-non-empty API key."""
    return LLMInterface(openai_api_key="sk-test-key")


# ---------------------------------------------------------------------------
# 1. Constructor
# ---------------------------------------------------------------------------

class TestInit:
    def test_stores_openai_model(self, llm: LLMInterface) -> None:
        assert llm.openai_model == "gpt-4o"

    def test_stores_empty_key(self) -> None:
        interface = LLMInterface(openai_api_key="")
        assert interface.openai_api_key == ""

    def test_default_ollama_model(self) -> None:
        interface = LLMInterface()
        assert interface.ollama_model == "llama3:8b"

    def test_default_temperature(self) -> None:
        interface = LLMInterface()
        assert interface.temperature == 0.1

    def test_custom_max_tokens(self) -> None:
        interface = LLMInterface(max_tokens=500)
        assert interface.max_tokens == 500

    def test_clients_start_as_none(self, llm: LLMInterface) -> None:
        assert llm._openai_client is None
        assert llm._ollama_client is None

    def test_ollama_base_url_strips_trailing_slash(self) -> None:
        interface = LLMInterface(ollama_base_url="http://localhost:11434/")
        assert not interface.ollama_base_url.endswith("/")


# ---------------------------------------------------------------------------
# 2. _to_messages
# ---------------------------------------------------------------------------

class TestToMessages:
    def test_string_wrapped_as_user(self, llm: LLMInterface) -> None:
        messages = llm._to_messages("What is metformin?")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "metformin" in messages[0]["content"]

    def test_list_returned_unchanged(self, llm: LLMInterface) -> None:
        msgs = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "What is diabetes?"},
        ]
        assert llm._to_messages(msgs) is msgs

    def test_empty_string_still_wraps(self, llm: LLMInterface) -> None:
        messages = llm._to_messages("")
        assert messages[0]["role"] == "user"

    def test_empty_list_returned_unchanged(self, llm: LLMInterface) -> None:
        assert llm._to_messages([]) == []


# ---------------------------------------------------------------------------
# 3. _extract_json
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_clean_json_returned_as_is(self, llm: LLMInterface) -> None:
        raw = '{"answer": "yes", "confidence": 0.9}'
        result = llm._extract_json(raw)
        assert json.loads(result)["answer"] == "yes"

    def test_strips_json_fenced(self, llm: LLMInterface) -> None:
        raw = '```json\n{"answer": "ok"}\n```'
        result = llm._extract_json(raw)
        assert json.loads(result)["answer"] == "ok"

    def test_strips_plain_fenced(self, llm: LLMInterface) -> None:
        raw = '```\n{"answer": "ok"}\n```'
        result = llm._extract_json(raw)
        assert json.loads(result)["answer"] == "ok"

    def test_extracts_json_from_prose(self, llm: LLMInterface) -> None:
        raw = 'Here is the answer: {"answer": "Metformin"} That is all.'
        result = llm._extract_json(raw)
        assert json.loads(result)["answer"] == "Metformin"

    def test_nested_json_preserved(self, llm: LLMInterface) -> None:
        raw = '{"a": {"b": "c"}, "d": [1, 2]}'
        result = llm._extract_json(raw)
        parsed = json.loads(result)
        assert parsed["a"]["b"] == "c"
        assert parsed["d"] == [1, 2]

    def test_no_braces_returns_text(self, llm: LLMInterface) -> None:
        raw = "No JSON here"
        result = llm._extract_json(raw)
        assert result == "No JSON here"

    def test_real_llm_output_pattern(self, llm: LLMInterface) -> None:
        raw = f"```json\n{json.dumps(GOOD_JSON)}\n```"
        result = llm._extract_json(raw)
        parsed = json.loads(result)
        assert parsed["answer"] == GOOD_JSON["answer"]


# ---------------------------------------------------------------------------
# 4. _build_confidence
# ---------------------------------------------------------------------------

class TestBuildConfidence:
    def test_uses_graph_path_confidence(self, llm: LLMInterface) -> None:
        rr = _FakeRetrievalResult(graph_subgraph=_make_subgraph(0.72))
        conf = llm._build_confidence(llm_confidence=0.5, retrieval_result=rr)
        assert conf == pytest.approx(0.72)

    def test_falls_back_to_llm_when_no_subgraph(self, llm: LLMInterface) -> None:
        rr = _FakeRetrievalResult(graph_subgraph=None)
        conf = llm._build_confidence(llm_confidence=0.65, retrieval_result=rr)
        assert conf == pytest.approx(0.65)

    def test_falls_back_to_llm_when_no_retrieval_result(self, llm: LLMInterface) -> None:
        conf = llm._build_confidence(llm_confidence=0.5, retrieval_result=None)
        assert conf == pytest.approx(0.5)

    def test_clamped_to_zero_minimum(self, llm: LLMInterface) -> None:
        conf = llm._build_confidence(llm_confidence=-0.5, retrieval_result=None)
        assert conf == 0.0

    def test_clamped_to_one_maximum(self, llm: LLMInterface) -> None:
        conf = llm._build_confidence(llm_confidence=1.5, retrieval_result=None)
        assert conf == 1.0

    def test_zero_path_confidence_falls_back_to_llm(self, llm: LLMInterface) -> None:
        # path_confidence=0.0 means no graph path traversed → fall back
        rr = _FakeRetrievalResult(graph_subgraph=_make_subgraph(0.0))
        conf = llm._build_confidence(llm_confidence=0.4, retrieval_result=rr)
        assert conf == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# 5. _parse_response
# ---------------------------------------------------------------------------

class TestParseResponse:
    def test_parses_good_json(self, llm: LLMInterface) -> None:
        raw = json.dumps(GOOD_JSON)
        answer = llm._parse_response(raw, retrieval_result=None)
        assert isinstance(answer, MedicalAnswer)
        assert "Metformin" in answer.answer

    def test_parses_citations(self, llm: LLMInterface) -> None:
        raw = json.dumps(GOOD_JSON)
        answer = llm._parse_response(raw, retrieval_result=None)
        assert len(answer.citations) == 1
        assert answer.citations[0].citation_id == "c1"
        assert answer.citations[0].year == 2023

    def test_parses_graph_path(self, llm: LLMInterface) -> None:
        raw = json.dumps(GOOD_JSON)
        answer = llm._parse_response(raw, retrieval_result=None)
        assert answer.graph_path == ["Type2Diabetes", "Metformin"]

    def test_uses_graph_confidence(self, llm: LLMInterface) -> None:
        rr = _FakeRetrievalResult(graph_subgraph=_make_subgraph(0.6))
        raw = json.dumps(GOOD_JSON)
        answer = llm._parse_response(raw, retrieval_result=rr)
        assert answer.confidence == pytest.approx(0.6)

    def test_uses_llm_confidence_without_graph(self, llm: LLMInterface) -> None:
        raw = json.dumps(GOOD_JSON)
        answer = llm._parse_response(raw, retrieval_result=None)
        assert answer.confidence == pytest.approx(0.75)

    def test_propagates_retrieval_mode(self, llm: LLMInterface) -> None:
        rr = _FakeRetrievalResult(retrieval_mode=RetrievalMode.VECTOR)
        raw = json.dumps(GOOD_JSON)
        answer = llm._parse_response(raw, retrieval_result=rr)
        assert answer.retrieval_mode == RetrievalMode.VECTOR

    def test_default_retrieval_mode_is_hybrid(self, llm: LLMInterface) -> None:
        raw = json.dumps(GOOD_JSON)
        answer = llm._parse_response(raw, retrieval_result=None)
        assert answer.retrieval_mode == RetrievalMode.HYBRID

    def test_attaches_raw_subgraph(self, llm: LLMInterface) -> None:
        sg = _make_subgraph()
        rr = _FakeRetrievalResult(graph_subgraph=sg)
        raw = json.dumps(GOOD_JSON)
        answer = llm._parse_response(raw, retrieval_result=rr)
        assert answer.raw_graph_subgraph is sg

    def test_handles_missing_answer_field(self, llm: LLMInterface) -> None:
        raw = json.dumps({"citations": [], "graph_path": [], "confidence": 0.5})
        answer = llm._parse_response(raw, retrieval_result=None)
        assert answer.answer == _FALLBACK_ANSWER_TEXT

    def test_handles_missing_citations(self, llm: LLMInterface) -> None:
        raw = json.dumps({"answer": "Some answer", "graph_path": [], "confidence": 0.3})
        answer = llm._parse_response(raw, retrieval_result=None)
        assert answer.citations == []

    def test_handles_empty_graph_path(self, llm: LLMInterface) -> None:
        raw = json.dumps({"answer": "ans", "citations": [], "graph_path": [], "confidence": 0.0})
        answer = llm._parse_response(raw, retrieval_result=None)
        assert answer.graph_path == []

    def test_handles_fenced_json(self, llm: LLMInterface) -> None:
        raw = f"```json\n{json.dumps(GOOD_JSON)}\n```"
        answer = llm._parse_response(raw, retrieval_result=None)
        assert "Metformin" in answer.answer

    def test_raises_on_unparseable_json(self, llm: LLMInterface) -> None:
        with pytest.raises(ValueError, match="JSON decode failed"):
            llm._parse_response("not json at all >>>", retrieval_result=None)

    def test_skips_malformed_citation(self, llm: LLMInterface) -> None:
        data = {
            "answer": "ans",
            "citations": [{"citation_id": "c1", "title": "ok", "source": "pubmed"},
                          "not-a-dict"],
            "graph_path": [],
            "confidence": 0.5,
        }
        answer = llm._parse_response(json.dumps(data), retrieval_result=None)
        # Only the valid dict citation is kept
        assert len(answer.citations) == 1

    def test_citation_url_is_optional(self, llm: LLMInterface) -> None:
        data = dict(GOOD_JSON)
        data["citations"] = [{"citation_id": "c1", "title": "no url",
                               "source": "pubmed", "url": None}]
        answer = llm._parse_response(json.dumps(data), retrieval_result=None)
        assert answer.citations[0].url is None


# ---------------------------------------------------------------------------
# 6. _fallback_answer
# ---------------------------------------------------------------------------

class TestFallbackAnswer:
    def test_returns_medical_answer(self, llm: LLMInterface) -> None:
        result = llm._fallback_answer("test error")
        assert isinstance(result, MedicalAnswer)

    def test_answer_text_is_fallback(self, llm: LLMInterface) -> None:
        result = llm._fallback_answer("err")
        assert result.answer == _FALLBACK_ANSWER_TEXT

    def test_confidence_is_zero(self, llm: LLMInterface) -> None:
        result = llm._fallback_answer("err")
        assert result.confidence == 0.0

    def test_citations_empty(self, llm: LLMInterface) -> None:
        result = llm._fallback_answer("err")
        assert result.citations == []

    def test_graph_path_empty(self, llm: LLMInterface) -> None:
        result = llm._fallback_answer("err")
        assert result.graph_path == []

    def test_disclaimer_populated(self, llm: LLMInterface) -> None:
        result = llm._fallback_answer("err")
        assert result.disclaimer  # non-empty string


# ---------------------------------------------------------------------------
# 7. _call_openai (mocked client)
# ---------------------------------------------------------------------------

class TestCallOpenAI:
    def test_calls_chat_completions(self, llm_with_key: LLMInterface) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            _make_mock_openai_response(json.dumps(GOOD_JSON))
        )
        llm_with_key._openai_client = mock_client

        result = llm_with_key._call_openai([{"role": "user", "content": "q"}])
        mock_client.chat.completions.create.assert_called_once()
        assert "Metformin" in result

    def test_passes_model_name(self, llm_with_key: LLMInterface) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            _make_mock_openai_response("{}")
        )
        llm_with_key._openai_client = mock_client
        llm_with_key._call_openai([{"role": "user", "content": "q"}])

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"

    def test_uses_json_response_format(self, llm_with_key: LLMInterface) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            _make_mock_openai_response("{}")
        )
        llm_with_key._openai_client = mock_client
        llm_with_key._call_openai([{"role": "user", "content": "q"}])

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs.get("response_format") == {"type": "json_object"}

    def test_raises_when_no_client(self, llm: LLMInterface) -> None:
        """Empty API key → RuntimeError when _ensure_openai_client returns None."""
        with pytest.raises(RuntimeError):
            llm._call_openai([{"role": "user", "content": "q"}])

    def test_raises_on_api_error(self, llm_with_key: LLMInterface) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("rate limit")
        llm_with_key._openai_client = mock_client

        with pytest.raises(Exception, match="rate limit"):
            llm_with_key._call_openai([{"role": "user", "content": "q"}])


# ---------------------------------------------------------------------------
# 8. _call_ollama (mocked client)
# ---------------------------------------------------------------------------

class TestCallOllama:
    def test_calls_chat_completions(self, llm: LLMInterface) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            _make_mock_openai_response(json.dumps(GOOD_JSON))
        )
        llm._ollama_client = mock_client

        result = llm._call_ollama([{"role": "user", "content": "q"}])
        assert "Metformin" in result

    def test_falls_back_when_json_mode_fails(self, llm: LLMInterface) -> None:
        mock_client = MagicMock()
        # First call (with response_format) raises; second call succeeds
        mock_client.chat.completions.create.side_effect = [
            Exception("json mode not supported"),
            _make_mock_openai_response(json.dumps(GOOD_JSON)),
        ]
        llm._ollama_client = mock_client

        result = llm._call_ollama([{"role": "user", "content": "q"}])
        assert "Metformin" in result
        assert mock_client.chat.completions.create.call_count == 2

    def test_raises_when_client_unavailable(self, llm: LLMInterface) -> None:
        """When openai package is unavailable _ensure_ollama_client returns None → RuntimeError."""
        with patch("src.generation.llm_interface.OPENAI_AVAILABLE", False):
            # Reset cached client so the patch takes effect
            llm._ollama_client = None
            with pytest.raises(RuntimeError):
                llm._call_ollama([{"role": "user", "content": "q"}])


# ---------------------------------------------------------------------------
# 9. call_llm end-to-end (mocked providers)
# ---------------------------------------------------------------------------

class TestCallLlm:
    def test_returns_medical_answer(self, llm_with_key: LLMInterface) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            _make_mock_openai_response(json.dumps(GOOD_JSON))
        )
        llm_with_key._openai_client = mock_client

        result = llm_with_key.call_llm("What treats diabetes?")
        assert isinstance(result, MedicalAnswer)

    def test_answer_text_populated(self, llm_with_key: LLMInterface) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            _make_mock_openai_response(json.dumps(GOOD_JSON))
        )
        llm_with_key._openai_client = mock_client

        result = llm_with_key.call_llm("query")
        assert result.answer == GOOD_JSON["answer"]

    def test_accepts_messages_list(self, llm_with_key: LLMInterface) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            _make_mock_openai_response(json.dumps(GOOD_JSON))
        )
        llm_with_key._openai_client = mock_client
        messages = [{"role": "system", "content": "..."}, {"role": "user", "content": "q"}]

        result = llm_with_key.call_llm(messages)
        assert isinstance(result, MedicalAnswer)

    def test_openai_failure_triggers_ollama_fallback(self, llm_with_key: LLMInterface) -> None:
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.side_effect = Exception("OpenAI down")
        llm_with_key._openai_client = mock_openai

        mock_ollama = MagicMock()
        mock_ollama.chat.completions.create.return_value = (
            _make_mock_openai_response(json.dumps(GOOD_JSON))
        )
        llm_with_key._ollama_client = mock_ollama

        result = llm_with_key.call_llm("query")
        assert isinstance(result, MedicalAnswer)
        mock_ollama.chat.completions.create.assert_called()

    def test_both_fail_returns_fallback_answer(self, llm_with_key: LLMInterface) -> None:
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.side_effect = Exception("OpenAI down")
        llm_with_key._openai_client = mock_openai

        mock_ollama = MagicMock()
        # Both attempts in _call_ollama raise
        mock_ollama.chat.completions.create.side_effect = Exception("Ollama down")
        llm_with_key._ollama_client = mock_ollama

        result = llm_with_key.call_llm("query")
        assert isinstance(result, MedicalAnswer)
        assert result.answer == _FALLBACK_ANSWER_TEXT
        assert result.confidence == 0.0

    def test_propagates_graph_confidence(self, llm_with_key: LLMInterface) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            _make_mock_openai_response(json.dumps(GOOD_JSON))
        )
        llm_with_key._openai_client = mock_client

        rr = _FakeRetrievalResult(graph_subgraph=_make_subgraph(0.48))
        result = llm_with_key.call_llm("query", retrieval_result=rr)
        assert result.confidence == pytest.approx(0.48)

    def test_propagates_retrieval_mode(self, llm_with_key: LLMInterface) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            _make_mock_openai_response(json.dumps(GOOD_JSON))
        )
        llm_with_key._openai_client = mock_client

        rr = _FakeRetrievalResult(retrieval_mode=RetrievalMode.GRAPH)
        result = llm_with_key.call_llm("query", retrieval_result=rr)
        assert result.retrieval_mode == RetrievalMode.GRAPH

    def test_disclaimer_always_present(self, llm_with_key: LLMInterface) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            _make_mock_openai_response(json.dumps(GOOD_JSON))
        )
        llm_with_key._openai_client = mock_client

        result = llm_with_key.call_llm("query")
        assert result.disclaimer  # non-empty

    def test_bad_json_returns_fallback(self, llm_with_key: LLMInterface) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            _make_mock_openai_response("This is not JSON at all, sorry!")
        )
        llm_with_key._openai_client = mock_client

        result = llm_with_key.call_llm("query")
        assert isinstance(result, MedicalAnswer)
        assert result.answer == _FALLBACK_ANSWER_TEXT

    def test_no_openai_key_skips_to_ollama(self) -> None:
        """When openai_api_key is empty, OpenAI is skipped entirely."""
        interface = LLMInterface(openai_api_key="")

        mock_ollama = MagicMock()
        mock_ollama.chat.completions.create.return_value = (
            _make_mock_openai_response(json.dumps(GOOD_JSON))
        )
        interface._ollama_client = mock_ollama

        result = interface.call_llm("query")
        assert isinstance(result, MedicalAnswer)
        # OpenAI client should never have been created
        assert interface._openai_client is None


# ---------------------------------------------------------------------------
# 10. Lazy client initialisation
# ---------------------------------------------------------------------------

class TestClientInit:
    def test_ensure_openai_returns_none_without_key(self) -> None:
        interface = LLMInterface(openai_api_key="")
        client = interface._ensure_openai_client()
        assert client is None

    @patch("src.generation.llm_interface.openai_lib")
    def test_ensure_openai_creates_client_with_key(self, mock_openai_mod) -> None:
        mock_openai_mod.OpenAI.return_value = MagicMock()
        interface = LLMInterface(openai_api_key="sk-real")
        client = interface._ensure_openai_client()
        assert client is not None
        mock_openai_mod.OpenAI.assert_called_once_with(api_key="sk-real")

    @patch("src.generation.llm_interface.openai_lib")
    def test_ensure_openai_cached_on_second_call(self, mock_openai_mod) -> None:
        mock_openai_mod.OpenAI.return_value = MagicMock()
        interface = LLMInterface(openai_api_key="sk-real")
        c1 = interface._ensure_openai_client()
        c2 = interface._ensure_openai_client()
        assert c1 is c2
        assert mock_openai_mod.OpenAI.call_count == 1

    @patch("src.generation.llm_interface.openai_lib")
    def test_ensure_ollama_uses_v1_endpoint(self, mock_openai_mod) -> None:
        mock_openai_mod.OpenAI.return_value = MagicMock()
        interface = LLMInterface(ollama_base_url="http://localhost:11434")
        interface._ensure_ollama_client()
        call_kwargs = mock_openai_mod.OpenAI.call_args[1]
        assert call_kwargs["base_url"] == "http://localhost:11434/v1"

    @patch("src.generation.llm_interface.openai_lib")
    def test_ensure_ollama_uses_dummy_key(self, mock_openai_mod) -> None:
        mock_openai_mod.OpenAI.return_value = MagicMock()
        interface = LLMInterface()
        interface._ensure_ollama_client()
        call_kwargs = mock_openai_mod.OpenAI.call_args[1]
        assert call_kwargs["api_key"] == "ollama"
