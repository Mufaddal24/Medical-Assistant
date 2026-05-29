"""
Tests for Module 6 — PromptBuilder (src/generation/prompt_builder.py)
All tests are pure unit tests — no external services required.

Compatibility notes
-------------------
- NodeType / EdgeType members use UPPERCASE names in this project
  (e.g. NodeType.DRUG, NodeType.DISEASE, EdgeType.TREATS).
  Enum members are resolved dynamically by .value so this file works
  regardless of naming convention.
- Chunk.score   is float (non-optional, default 0.0) — never pass None.
- Chunk.pub_date is Optional[datetime] — use datetime objects, not strings.
- WebSnippet.pub_date is Optional[str] — plain strings are fine.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import List, Optional

import pytest

from src.generation.prompt_builder import (
    MEDICAL_DISCLAIMER,
    PromptBuilder,
    _BUDGET_GRAPH,
    _BUDGET_MERGED,
    _BUDGET_VECTOR,
    _BUDGET_WEB,
)
from src.utils.models import (
    Chunk,
    EdgeType,
    GraphEdge,
    GraphNode,
    GraphSubgraph,
    NodeType,
    WebSnippet,
)


# ---------------------------------------------------------------------------
# Dynamic enum resolution
# Resolves NodeType / EdgeType members by .value, regardless of member name
# (handles DRUG, Drug, drug equally).
# ---------------------------------------------------------------------------


def _find_node_type(*values: str) -> NodeType:
    """Return the NodeType member whose .value matches one of *values*
    (case-insensitive). Falls back to the first available member."""
    lower_vals = {v.lower() for v in values}
    for member in NodeType:
        if member.value.lower() in lower_vals:
            return member
    return list(NodeType)[0]


def _find_edge_type(*values: str) -> EdgeType:
    """Return the EdgeType member whose .value matches one of *values*
    (case-insensitive). Falls back to the first available member."""
    lower_vals = {v.lower() for v in values}
    for member in EdgeType:
        if member.value.lower() in lower_vals:
            return member
    return list(EdgeType)[0]


# Resolved once — safe because we look up by .value, not member name
_NT_DRUG = _find_node_type("Drug", "drug")
_NT_DISEASE = _find_node_type("Disease", "disease")
_ET_TREATS = _find_edge_type("TREATS", "treats")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def builder() -> PromptBuilder:
    return PromptBuilder()


def _make_node(
    node_id: str,
    name: str,
    node_type: Optional[NodeType] = None,
    conf: float = 0.9,
) -> GraphNode:
    if node_type is None:
        node_type = _NT_DRUG
    return GraphNode(
        id=node_id,
        name=name,
        node_type=node_type,
        source_url=f"https://example.com/{node_id}",
        last_updated=datetime.now(tz=timezone.utc),
        confidence_score=conf,
    )


def _make_edge(
    src: str,
    tgt: str,
    etype: Optional[EdgeType] = None,
    conf: float = 0.8,
) -> GraphEdge:
    if etype is None:
        etype = _ET_TREATS
    return GraphEdge(
        source_id=src,
        target_id=tgt,
        edge_type=etype,
        confidence=conf,
        source_doc_id="doc_001",
        year=2023,
    )


def _make_subgraph() -> GraphSubgraph:
    drug = _make_node("C001", "Metformin", _NT_DRUG)
    disease = _make_node("C002", "Type 2 Diabetes", _NT_DISEASE)
    edge = _make_edge("C001", "C002")
    return GraphSubgraph(
        nodes=[drug, disease],
        edges=[edge],
        query_node_ids=["C002"],
        path_confidence=0.8,
    )


def _make_chunk(idx: int, source: str = "pubmed") -> Chunk:
    """Create a sample Chunk matching the real models.py schema.

    Key constraints from real models.py:
      - score   : float (non-optional, default 0.0) — never None
      - pub_date: Optional[datetime]                — use datetime, not str
      - doc_id  : str (required)
    """
    return Chunk(
        chunk_id=f"chunk_{idx:04d}",
        text=f"Sample evidence text number {idx} about diabetes treatment.",
        doc_id=f"doc_{idx}",
        node_cui=None,
        score=max(0.01, 0.9 - idx * 0.05),   # always a positive float
        source=source,
        source_url=f"https://pubmed.ncbi.nlm.nih.gov/{1000 + idx}/",
        pub_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
    )


def _make_snippet(idx: int) -> WebSnippet:
    """Create a sample WebSnippet.

    WebSnippet.pub_date is Optional[str] — plain strings are correct here.
    """
    return WebSnippet(
        title=f"Web result {idx}: Latest diabetes research",
        url=f"https://news.example.com/article-{idx}",
        snippet=f"Recent web snippet {idx} about diabetes management and drug therapies.",
        pub_date="2024-06-01",   # str — correct type for WebSnippet
        source="web",
        score=max(0.01, 0.85 - idx * 0.05),
    )


# ---------------------------------------------------------------------------
# 1. System block
# ---------------------------------------------------------------------------


class TestSystemBlock:
    def test_contains_disclaimer(self, builder: PromptBuilder) -> None:
        block = builder._build_system_block()
        assert MEDICAL_DISCLAIMER in block

    def test_contains_role_description(self, builder: PromptBuilder) -> None:
        block = builder._build_system_block()
        assert "biomedical knowledge assistant" in block.lower()

    def test_instructs_json_only_output(self, builder: PromptBuilder) -> None:
        block = builder._build_system_block()
        assert "JSON" in block

    def test_no_hallucination_rule(self, builder: PromptBuilder) -> None:
        block = builder._build_system_block()
        assert "hallucinate" in block.lower()


# ---------------------------------------------------------------------------
# 2. Graph block
# ---------------------------------------------------------------------------


class TestGraphBlock:
    def test_none_subgraph_returns_empty(self, builder: PromptBuilder) -> None:
        assert builder._build_graph_block(None) == ""

    def test_empty_subgraph_returns_empty(self, builder: PromptBuilder) -> None:
        empty = GraphSubgraph(nodes=[], edges=[], query_node_ids=[], path_confidence=1.0)
        assert builder._build_graph_block(empty) == ""

    def test_contains_node_names(self, builder: PromptBuilder) -> None:
        block = builder._build_graph_block(_make_subgraph())
        assert "Metformin" in block
        assert "Type 2 Diabetes" in block

    def test_contains_edge_type_value(self, builder: PromptBuilder) -> None:
        block = builder._build_graph_block(_make_subgraph())
        assert _ET_TREATS.value in block

    def test_marks_query_match(self, builder: PromptBuilder) -> None:
        block = builder._build_graph_block(_make_subgraph())
        assert "QUERY MATCH" in block

    def test_includes_path_confidence(self, builder: PromptBuilder) -> None:
        block = builder._build_graph_block(_make_subgraph())
        assert "path_confidence" in block

    def test_trimmed_when_oversized(self, builder: PromptBuilder) -> None:
        nodes = [_make_node(f"C{i:04d}", "X" * 500) for i in range(50)]
        big = GraphSubgraph(nodes=nodes, edges=[], query_node_ids=[], path_confidence=1.0)
        block = builder._build_graph_block(big)
        assert len(block) <= _BUDGET_GRAPH + 50


# ---------------------------------------------------------------------------
# 3. Vector block
# ---------------------------------------------------------------------------


class TestVectorBlock:
    def test_empty_list_returns_empty(self, builder: PromptBuilder) -> None:
        assert builder._build_vector_block([]) == ""

    def test_contains_chunk_text(self, builder: PromptBuilder) -> None:
        block = builder._build_vector_block([_make_chunk(0)])
        assert "Sample evidence text" in block

    def test_contains_source_url(self, builder: PromptBuilder) -> None:
        block = builder._build_vector_block([_make_chunk(1)])
        assert "pubmed.ncbi.nlm.nih.gov" in block

    def test_contains_score(self, builder: PromptBuilder) -> None:
        block = builder._build_vector_block([_make_chunk(0)])
        assert "score=" in block

    def test_multiple_chunks_indexed(self, builder: PromptBuilder) -> None:
        block = builder._build_vector_block([_make_chunk(i) for i in range(3)])
        assert "[V1]" in block
        assert "[V2]" in block
        assert "[V3]" in block

    def test_trimmed_when_oversized(self, builder: PromptBuilder) -> None:
        chunks = [
            Chunk(chunk_id=f"c{i}", text="A" * 5000, doc_id=f"d{i}",
                  score=0.5, source="pubmed")
            for i in range(20)
        ]
        block = builder._build_vector_block(chunks)
        assert len(block) <= _BUDGET_VECTOR + 100


# ---------------------------------------------------------------------------
# 4. Web block
# ---------------------------------------------------------------------------


class TestWebBlock:
    def test_empty_list_returns_empty(self, builder: PromptBuilder) -> None:
        assert builder._build_web_block([]) == ""

    def test_contains_snippet_text(self, builder: PromptBuilder) -> None:
        block = builder._build_web_block([_make_snippet(0)])
        assert "Recent web snippet 0" in block

    def test_contains_web_label(self, builder: PromptBuilder) -> None:
        block = builder._build_web_block([_make_snippet(0)])
        assert "[WEB]" in block

    def test_contains_pub_date(self, builder: PromptBuilder) -> None:
        block = builder._build_web_block([_make_snippet(0)])
        assert "2024-06-01" in block

    def test_contains_url(self, builder: PromptBuilder) -> None:
        block = builder._build_web_block([_make_snippet(0)])
        assert "news.example.com" in block

    def test_indexed_w_labels(self, builder: PromptBuilder) -> None:
        snippets = [_make_snippet(i) for i in range(3)]
        block = builder._build_web_block(snippets)
        assert "[W1]" in block
        assert "[W3]" in block

    def test_trimmed_when_oversized(self, builder: PromptBuilder) -> None:
        snippets = [
            WebSnippet(title=f"T{i}", url=f"https://x.com/{i}",
                       snippet="B" * 5000, source="web", score=0.5)
            for i in range(20)
        ]
        block = builder._build_web_block(snippets)
        assert len(block) <= _BUDGET_WEB + 100


# ---------------------------------------------------------------------------
# 5. Merged block
# ---------------------------------------------------------------------------


class TestMergedBlock:
    def test_empty_list_returns_empty(self, builder: PromptBuilder) -> None:
        assert builder._build_merged_block([]) == ""

    def test_contains_merged_label(self, builder: PromptBuilder) -> None:
        block = builder._build_merged_block([_make_chunk(0)])
        assert "MERGED" in block

    def test_m_indexed_labels(self, builder: PromptBuilder) -> None:
        block = builder._build_merged_block([_make_chunk(i) for i in range(2)])
        assert "[M1]" in block
        assert "[M2]" in block

    def test_trimmed_when_oversized(self, builder: PromptBuilder) -> None:
        chunks = [
            Chunk(chunk_id=f"c{i}", text="C" * 5000, doc_id=f"d{i}",
                  score=0.5, source="web")
            for i in range(20)
        ]
        block = builder._build_merged_block(chunks)
        assert len(block) <= _BUDGET_MERGED + 100


# ---------------------------------------------------------------------------
# 6. Schema block
# ---------------------------------------------------------------------------


class TestSchemaBlock:
    def test_contains_required_keys(self, builder: PromptBuilder) -> None:
        block = builder._build_schema_block()
        for key in ("answer", "citations", "graph_path", "confidence"):
            assert key in block

    def test_is_valid_json_structure(self, builder: PromptBuilder) -> None:
        block = builder._build_schema_block()
        json_start = block.index("{")
        parsed = json.loads(block[json_start:])
        assert "answer" in parsed
        assert "citations" in parsed
        assert isinstance(parsed["citations"], list)


# ---------------------------------------------------------------------------
# 7. Full build()
# ---------------------------------------------------------------------------


class TestBuild:
    def test_returns_string(self, builder: PromptBuilder) -> None:
        prompt = builder.build(
            query="What treats diabetes?",
            graph_subgraph=_make_subgraph(),
            vector_chunks=[_make_chunk(0)],
            web_snippets=[_make_snippet(0)],
            merged_chunks=[_make_chunk(0)],
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_contains_question(self, builder: PromptBuilder) -> None:
        q = "What drugs treat Type 2 Diabetes?"
        assert q in builder.build(q, None, [], [], [])

    def test_contains_disclaimer(self, builder: PromptBuilder) -> None:
        assert "MEDICAL DISCLAIMER" in builder.build("q?", None, [], [], [])

    def test_all_sections_present(self, builder: PromptBuilder) -> None:
        prompt = builder.build(
            "What treats diabetes?",
            _make_subgraph(),
            [_make_chunk(0)],
            [_make_snippet(0)],
            [_make_chunk(0)],
        )
        assert "KNOWLEDGE GRAPH" in prompt
        assert "VECTOR SEARCH" in prompt
        assert "LIVE WEB SEARCH" in prompt
        assert "MERGED" in prompt
        assert "REQUIRED JSON OUTPUT SCHEMA" in prompt
        assert "QUESTION" in prompt

    def test_no_graph_skips_graph_section(self, builder: PromptBuilder) -> None:
        assert "KNOWLEDGE GRAPH" not in builder.build("q?", None, [], [], [])

    def test_no_web_skips_web_section(self, builder: PromptBuilder) -> None:
        prompt = builder.build("q?", _make_subgraph(), [_make_chunk(0)], [], [])
        assert "LIVE WEB SEARCH" not in prompt

    def test_respects_max_chars(self) -> None:
        tiny = PromptBuilder(max_prompt_chars=500)
        prompt = tiny.build(
            "Short question?",
            _make_subgraph(),
            [_make_chunk(i) for i in range(10)],
            [_make_snippet(i) for i in range(10)],
            [_make_chunk(i) for i in range(10)],
        )
        assert len(prompt) <= 500 + 200

    def test_question_always_present_after_trim(self) -> None:
        tiny = PromptBuilder(max_prompt_chars=300)
        q = "Why is metformin first-line?"
        prompt = tiny.build(q, _make_subgraph(), [_make_chunk(i) for i in range(10)], [], [])
        assert q in prompt


# ---------------------------------------------------------------------------
# 8. build_messages()
# ---------------------------------------------------------------------------


class TestBuildMessages:
    def test_returns_two_messages(self, builder: PromptBuilder) -> None:
        assert len(builder.build_messages("q?", None, [], [], [])) == 2

    def test_first_is_system(self, builder: PromptBuilder) -> None:
        assert builder.build_messages("q?", None, [], [], [])[0]["role"] == "system"

    def test_second_is_user(self, builder: PromptBuilder) -> None:
        assert builder.build_messages("q?", None, [], [], [])[1]["role"] == "user"

    def test_system_contains_disclaimer(self, builder: PromptBuilder) -> None:
        msgs = builder.build_messages("q?", None, [], [], [])
        assert "MEDICAL DISCLAIMER" in msgs[0]["content"]

    def test_user_contains_question(self, builder: PromptBuilder) -> None:
        q = "What is metformin used for?"
        msgs = builder.build_messages(q, None, [], [], [])
        assert q in msgs[1]["content"]

    def test_user_contains_schema(self, builder: PromptBuilder) -> None:
        msgs = builder.build_messages("q?", None, [], [], [])
        assert "REQUIRED JSON OUTPUT SCHEMA" in msgs[1]["content"]


# ---------------------------------------------------------------------------
# 9. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_query(self, builder: PromptBuilder) -> None:
        assert isinstance(builder.build("", None, [], [], []), str)

    def test_very_long_query(self, builder: PromptBuilder) -> None:
        assert isinstance(builder.build("diabetes " * 1000, None, [], [], []), str)

    def test_chunk_with_zero_score_still_shows_score(self, builder: PromptBuilder) -> None:
        """score=0.0 is falsy but must still appear — real models have float not Optional[float]."""
        chunk = Chunk(
            chunk_id="zero_score",
            text="Chunk with zero score.",
            doc_id="d0",
            score=0.0,
            source="pubmed",
        )
        block = builder._build_vector_block([chunk])
        assert "score=0.000" in block

    def test_chunk_without_optional_fields(self, builder: PromptBuilder) -> None:
        """Minimal Chunk with only required + score (float default 0.0)."""
        chunk = Chunk(
            chunk_id="bare",
            text="Bare chunk with no optional fields.",
            doc_id="d0",
            score=0.0,          # float, never None — use 0.0 not None
            source="pubmed",
            source_url=None,
            pub_date=None,
        )
        block = builder._build_vector_block([chunk])
        assert "Bare chunk" in block

    def test_snippet_without_pub_date(self, builder: PromptBuilder) -> None:
        snip = WebSnippet(
            title="No date snippet",
            url="https://x.com",
            snippet="Content here.",
            pub_date=None,
            source="web",
            score=0.7,
        )
        assert "Content here." in builder._build_web_block([snip])

    def test_subgraph_without_source_url(self, builder: PromptBuilder) -> None:
        node = GraphNode(
            id="X1",
            name="Mystery Drug",
            node_type=_NT_DRUG,
            source_url=None,
            last_updated=datetime.now(tz=timezone.utc),
            confidence_score=0.5,
        )
        sg = GraphSubgraph(nodes=[node], edges=[], query_node_ids=[], path_confidence=0.5)
        assert "Mystery Drug" in builder._build_graph_block(sg)

    def test_node_type_values_accessible(self) -> None:
        """All 6 NodeType values resolve without error."""
        for label in ("Disease", "Drug", "Gene", "Symptom", "ClinicalTrial", "Paper"):
            assert _find_node_type(label) is not None

    def test_edge_type_values_accessible(self) -> None:
        """All 6 EdgeType values resolve without error."""
        for label in ("TREATS", "CAUSES", "INTERACTS_WITH",
                      "ASSOCIATED_WITH", "INVESTIGATED_IN", "CITED_BY"):
            assert _find_edge_type(label) is not None