"""
Tests for Module 5 — RetrievalOrchestrator.

All external services (Neo4j, Qdrant, Tavily, Cohere) are mocked.
No API keys or running infrastructure required for the unit test suite.

Integration tests require:
  - docker compose up -d
  - RUN_INTEGRATION_TESTS=1
  - Valid TAVILY_API_KEY and COHERE_API_KEY in .env
"""

from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from src.utils.models import (
    Chunk,
    EdgeType,
    GraphEdge,
    GraphNode,
    GraphSubgraph,
    NodeType,
    RetrievalMode,
    WebSnippet,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_graph_builder():
    gb = MagicMock()
    gb.is_connected.return_value = True
    gb.search_nodes_by_name.return_value = [
        GraphNode(id="C0025598", name="Metformin", node_type=NodeType.DRUG),
        GraphNode(id="C0011860", name="Type 2 Diabetes", node_type=NodeType.DISEASE),
    ]
    gb.get_subgraph.return_value = GraphSubgraph(
        nodes=[
            GraphNode(
                id="C0025598",
                name="Metformin",
                node_type=NodeType.DRUG,
                confidence_score=0.95,
                last_updated=datetime.now(timezone.utc),
            ),
            GraphNode(
                id="C0011860",
                name="Type 2 Diabetes",
                node_type=NodeType.DISEASE,
                confidence_score=0.98,
                last_updated=datetime.now(timezone.utc),
            ),
        ],
        edges=[
            GraphEdge(
                source_id="C0025598",
                target_id="C0011860",
                edge_type=EdgeType.TREATS,
                confidence=0.85,
                source_doc_id="doc_001",
                year=2023,
            )
        ],
        query_node_ids=["C0025598"],
        path_confidence=0.85,
    )
    return gb


@pytest.fixture()
def mock_vector_indexer():
    vi = MagicMock()
    vi.similarity_search.return_value = [
        Chunk(
            chunk_id="chunk_001",
            text="Metformin is the first-line treatment for Type 2 Diabetes.",
            doc_id="doc_001",
            score=0.91,
            source="pubmed",
            source_url="https://pubmed.ncbi.nlm.nih.gov/12345/",
        ),
        Chunk(
            chunk_id="chunk_002",
            text="Metformin reduces hepatic glucose production.",
            doc_id="doc_002",
            score=0.85,
            source="pubmed",
        ),
        Chunk(
            chunk_id="chunk_003",
            text="Type 2 Diabetes management guidelines recommend metformin.",
            doc_id="doc_003",
            score=0.80,
            source="pubmed",
        ),
    ]
    return vi


@pytest.fixture()
def orchestrator(mock_graph_builder, mock_vector_indexer):
    """Orchestrator with mocked dependencies and no real API clients."""
    from src.retrieval.orchestrator import RetrievalOrchestrator

    with patch("src.retrieval.orchestrator.TavilyClient", MagicMock()):
        with patch("src.retrieval.orchestrator.cohere_lib", MagicMock()):
            orch = RetrievalOrchestrator(
                graph_builder=mock_graph_builder,
                vector_indexer=mock_vector_indexer,
                tavily_api_key="fake_tavily_key",
                cohere_api_key="fake_cohere_key",
            )
    return orch


@pytest.fixture()
def orchestrator_no_apis(mock_graph_builder, mock_vector_indexer):
    """Orchestrator with no external API clients at all."""
    from src.retrieval.orchestrator import RetrievalOrchestrator

    orch = RetrievalOrchestrator(
        graph_builder=mock_graph_builder,
        vector_indexer=mock_vector_indexer,
        tavily_api_key="",
        cohere_api_key="",
    )
    return orch


# ---------------------------------------------------------------------------
# Import and instantiation
# ---------------------------------------------------------------------------

class TestOrchestratorInit:

    def test_import(self) -> None:
        """RetrievalOrchestrator can be imported."""
        from src.retrieval.orchestrator import RetrievalOrchestrator
        assert RetrievalOrchestrator is not None

    def test_instantiation_without_services(self) -> None:
        """Orchestrator instantiates gracefully with no services configured."""
        from src.retrieval.orchestrator import RetrievalOrchestrator

        orch = RetrievalOrchestrator(
            graph_builder=None,
            vector_indexer=None,
            tavily_api_key="",
            cohere_api_key="",
        )
        assert orch is not None

    def test_retrieval_result_model(self) -> None:
        """RetrievalResult is a valid Pydantic model."""
        from src.retrieval.orchestrator import RetrievalResult

        result = RetrievalResult(
            query="test",
            retrieval_mode=RetrievalMode.HYBRID,
        )
        assert result.query == "test"
        assert result.merged_chunks == []
        assert result.web_search_triggered is False


# ---------------------------------------------------------------------------
# Query classifier
# ---------------------------------------------------------------------------

class TestQueryClassifier:

    def test_temporal_keyword_sets_web_search(self, orchestrator) -> None:
        """Queries with temporal keywords set should_web_search=True."""
        state = {
            "query": "What are the latest treatments for diabetes?",
            "retrieval_mode": "",
            "should_web_search": False,
            "query_cuis": [],
            "errors": [],
        }
        update = orchestrator._node_classify_query(state)
        assert update["should_web_search"] is True

    def test_temporal_keyword_2025(self, orchestrator) -> None:
        """Year 2025 triggers web search."""
        state = {
            "query": "diabetes treatment guidelines 2025",
            "retrieval_mode": "",
            "should_web_search": False,
            "query_cuis": [],
            "errors": [],
        }
        update = orchestrator._node_classify_query(state)
        assert update["should_web_search"] is True

    def test_non_temporal_defaults_to_hybrid(self, orchestrator) -> None:
        """Non-temporal medical queries default to hybrid mode."""
        state = {
            "query": "What drugs treat Type 2 Diabetes?",
            "retrieval_mode": "",
            "should_web_search": False,
            "query_cuis": [],
            "errors": [],
        }
        update = orchestrator._node_classify_query(state)
        assert update["retrieval_mode"] == RetrievalMode.HYBRID.value

    def test_graph_keywords_route_to_graph(self, orchestrator) -> None:
        """Queries with graph keywords route to graph mode."""
        state = {
            "query": "What is the relationship between metformin and diabetes?",
            "retrieval_mode": "",
            "should_web_search": False,
            "query_cuis": [],
            "errors": [],
        }
        update = orchestrator._node_classify_query(state)
        assert update["retrieval_mode"] == RetrievalMode.GRAPH.value

    def test_forced_mode_respected(self, orchestrator) -> None:
        """Forced retrieval mode overrides classifier logic."""
        state = {
            "query": "latest diabetes news",   # temporal keyword
            "retrieval_mode": RetrievalMode.VECTOR.value,  # forced
            "should_web_search": False,
            "query_cuis": [],
            "errors": [],
        }
        update = orchestrator._node_classify_query(state)
        assert update["retrieval_mode"] == RetrievalMode.VECTOR.value

    def test_cuis_extracted_from_graph(
        self, orchestrator, mock_graph_builder
    ) -> None:
        """Classifier extracts CUIs by searching the graph."""
        state = {
            "query": "Metformin for diabetes",
            "retrieval_mode": "",
            "should_web_search": False,
            "query_cuis": [],
            "errors": [],
        }
        update = orchestrator._node_classify_query(state)
        assert len(update["query_cuis"]) > 0

    def test_has_temporal_keywords(self, orchestrator) -> None:
        """_has_temporal_keywords correctly identifies temporal words."""
        assert orchestrator._has_temporal_keywords("latest research") is True
        assert orchestrator._has_temporal_keywords("recent studies") is True
        assert orchestrator._has_temporal_keywords("current guidelines") is True
        assert orchestrator._has_temporal_keywords("diabetes treatment") is False


# ---------------------------------------------------------------------------
# Graph retriever
# ---------------------------------------------------------------------------

class TestGraphRetriever:

    def test_returns_subgraph(self, orchestrator, mock_graph_builder) -> None:
        """GraphRetriever returns a GraphSubgraph with nodes and edges."""
        state = {
            "query": "metformin treatment",
            "retrieval_mode": RetrievalMode.HYBRID.value,
            "query_cuis": ["C0025598"],
            "should_web_search": False,
            "errors": [],
        }
        update = orchestrator._node_graph_retriever(state)

        assert "graph_subgraph" in update
        subgraph = update["graph_subgraph"]
        assert subgraph is not None
        assert len(subgraph.nodes) == 2
        assert len(subgraph.edges) == 1

    def test_skips_when_no_cuis(
        self, orchestrator, mock_graph_builder
    ) -> None:
        """GraphRetriever skips cleanly when no CUIs are found."""
        mock_graph_builder.search_nodes_by_name.return_value = []
        state = {
            "query": "xyzzy unknown term",
            "retrieval_mode": RetrievalMode.HYBRID.value,
            "query_cuis": [],
            "should_web_search": False,
            "errors": [],
        }
        update = orchestrator._node_graph_retriever(state)
        assert update.get("graph_subgraph") is None

    def test_stale_node_triggers_web_search(
        self, orchestrator, mock_graph_builder
    ) -> None:
        """A node older than staleness threshold sets should_web_search=True."""
        stale_date = datetime.now(timezone.utc) - timedelta(days=200)
        mock_graph_builder.get_subgraph.return_value = GraphSubgraph(
            nodes=[
                GraphNode(
                    id="C0025598",
                    name="Metformin",
                    node_type=NodeType.DRUG,
                    last_updated=stale_date,
                )
            ],
            edges=[],
            query_node_ids=["C0025598"],
        )
        state = {
            "query": "metformin",
            "retrieval_mode": RetrievalMode.HYBRID.value,
            "query_cuis": ["C0025598"],
            "should_web_search": False,
            "errors": [],
        }
        update = orchestrator._node_graph_retriever(state)
        assert update["should_web_search"] is True

    def test_fresh_node_does_not_trigger_web(
        self, orchestrator, mock_graph_builder
    ) -> None:
        """A recently updated node does not trigger web search."""
        state = {
            "query": "metformin",
            "retrieval_mode": RetrievalMode.HYBRID.value,
            "query_cuis": ["C0025598"],
            "should_web_search": False,
            "errors": [],
        }
        update = orchestrator._node_graph_retriever(state)
        assert update.get("should_web_search", False) is False

    def test_handles_missing_graph_builder(self) -> None:
        """GraphRetriever returns error gracefully when GraphBuilder is None."""
        from src.retrieval.orchestrator import RetrievalOrchestrator

        orch = RetrievalOrchestrator(graph_builder=None, vector_indexer=None)
        state = {
            "query": "diabetes",
            "retrieval_mode": "hybrid",
            "query_cuis": ["C0011860"],
            "should_web_search": False,
            "errors": [],
        }
        update = orch._node_graph_retriever(state)
        assert "errors" in update
        assert len(update["errors"]) > 0


# ---------------------------------------------------------------------------
# Vector retriever
# ---------------------------------------------------------------------------

class TestVectorRetriever:

    def test_returns_chunks(
        self, orchestrator, mock_vector_indexer
    ) -> None:
        """VectorRetriever returns chunks from similarity search."""
        state = {
            "query": "diabetes treatment",
            "retrieval_mode": RetrievalMode.HYBRID.value,
            "vector_chunks": [],
            "should_web_search": False,
            "errors": [],
        }
        update = orchestrator._node_vector_retriever(state)

        assert "vector_chunks" in update
        assert len(update["vector_chunks"]) == 3

    def test_low_results_triggers_web_search(
        self, orchestrator, mock_vector_indexer
    ) -> None:
        """Fewer than MIN_RESULTS_FOR_WEB chunks triggers web search."""
        mock_vector_indexer.similarity_search.return_value = [
            Chunk(chunk_id="c1", text="short result", doc_id="d1", score=0.5, source="pubmed")
        ]
        state = {
            "query": "rare condition xyz",
            "retrieval_mode": RetrievalMode.HYBRID.value,
            "vector_chunks": [],
            "should_web_search": False,
            "errors": [],
        }
        update = orchestrator._node_vector_retriever(state)
        assert update["should_web_search"] is True

    def test_sufficient_results_no_web_trigger(
        self, orchestrator, mock_vector_indexer
    ) -> None:
        """Three or more results do not trigger web search."""
        state = {
            "query": "diabetes treatment",
            "retrieval_mode": RetrievalMode.HYBRID.value,
            "vector_chunks": [],
            "should_web_search": False,
            "errors": [],
        }
        update = orchestrator._node_vector_retriever(state)
        assert update.get("should_web_search", False) is False

    def test_handles_missing_vector_indexer(self) -> None:
        """VectorRetriever returns error gracefully when VectorIndexer is None."""
        from src.retrieval.orchestrator import RetrievalOrchestrator

        orch = RetrievalOrchestrator(graph_builder=None, vector_indexer=None)
        state = {
            "query": "diabetes",
            "retrieval_mode": "hybrid",
            "vector_chunks": [],
            "should_web_search": False,
            "errors": [],
        }
        update = orch._node_vector_retriever(state)
        assert "errors" in update


# ---------------------------------------------------------------------------
# Web search node
# ---------------------------------------------------------------------------

class TestWebSearchNode:

    def test_skips_when_flag_false(self, orchestrator) -> None:
        """WebSearchNode returns empty dict when should_web_search is False."""
        orchestrator._force_web = False
        state = {
            "query": "diabetes",
            "should_web_search": False,
        }
        update = orchestrator._node_web_search(state)
        assert update == {}

    def test_returns_snippets_when_triggered(self, orchestrator) -> None:
        """WebSearchNode returns WebSnippet objects when Tavily is mocked."""
        mock_tavily = MagicMock()
        mock_tavily.search.return_value = {
            "results": [
                {
                    "title": "Metformin Review 2024",
                    "url": "https://example.com/metformin",
                    "content": "Metformin remains the gold standard for T2DM.",
                    "score": 0.88,
                    "published_date": "2024-03-15",
                }
            ]
        }
        orchestrator._tavily = mock_tavily

        state = {
            "query": "latest metformin research",
            "should_web_search": True,
        }
        update = orchestrator._node_web_search(state)

        assert "web_snippets" in update
        assert len(update["web_snippets"]) == 1
        snippet = update["web_snippets"][0]
        assert snippet.source == "web"
        assert snippet.score == 0.88
        assert "Metformin" in snippet.title

    def test_returns_error_when_tavily_unavailable(
        self, orchestrator_no_apis
    ) -> None:
        """WebSearchNode adds error when Tavily client is explicitly set to None.

        Uses direct attribute override instead of relying on an empty env var,
        so the test is robust even when TAVILY_API_KEY is present in .env.
        """
        orchestrator_no_apis._tavily = None   # force unavailable
        state = {
            "query": "latest diabetes news",
            "should_web_search": True,
        }
        update = orchestrator_no_apis._node_web_search(state)
        assert "errors" in update

    def test_tavily_exception_handled(self, orchestrator) -> None:
        """WebSearchNode handles Tavily API errors gracefully."""
        mock_tavily = MagicMock()
        mock_tavily.search.side_effect = Exception("rate limit")
        orchestrator._tavily = mock_tavily

        state = {"query": "diabetes", "should_web_search": True}
        update = orchestrator._node_web_search(state)
        assert "errors" in update


# ---------------------------------------------------------------------------
# Hybrid merger
# ---------------------------------------------------------------------------

class TestHybridMerger:

    def test_merges_all_sources(self, orchestrator, mock_graph_builder) -> None:
        """HybridMerger combines graph, vector, and web chunks."""
        # Disable Cohere so merger uses score-sort (avoids mock API shape issues)
        orchestrator._cohere = None

        state = {
            "query": "metformin diabetes",
            "graph_subgraph": mock_graph_builder.get_subgraph.return_value,
            "vector_chunks": [
                Chunk(chunk_id="v1", text="vector chunk", doc_id="d1",
                      score=0.7, source="pubmed")
            ],
            "web_snippets": [
                WebSnippet(
                    title="Web result",
                    url="https://example.com",
                    snippet="web snippet text",
                    source="web",
                    score=0.6,
                )
            ],
        }
        update = orchestrator._node_hybrid_merger(state)

        assert "merged_chunks" in update
        assert len(update["merged_chunks"]) > 0

    def test_returns_empty_for_no_input(self, orchestrator) -> None:
        """HybridMerger returns empty list when all sources are empty."""
        state = {
            "query": "nothing",
            "graph_subgraph": None,
            "vector_chunks": [],
            "web_snippets": [],
        }
        update = orchestrator._node_hybrid_merger(state)
        assert update["merged_chunks"] == []

    def test_deduplication_removes_duplicates(self, orchestrator) -> None:
        """_deduplicate removes near-identical chunks."""
        text = "Metformin treats Type 2 Diabetes mellitus effectively."
        chunks = [
            Chunk(chunk_id="c1", text=text, doc_id="d1", score=0.9, source="graph"),
            Chunk(chunk_id="c2", text=text, doc_id="d2", score=0.7, source="vector"),
        ]
        result = orchestrator._deduplicate(chunks)
        assert len(result) == 1
        assert result[0].score == 0.9  # keeps highest score

    def test_deduplication_keeps_unique(self, orchestrator) -> None:
        """_deduplicate keeps chunks with genuinely different content."""
        chunks = [
            Chunk(chunk_id="c1", text="Metformin treats diabetes.", doc_id="d1",
                  score=0.9, source="graph"),
            Chunk(chunk_id="c2", text="Insulin regulates blood glucose.", doc_id="d2",
                  score=0.8, source="vector"),
        ]
        result = orchestrator._deduplicate(chunks)
        assert len(result) == 2

    def test_cohere_rerank_fallback_on_error(self, orchestrator) -> None:
        """_cohere_rerank falls back to score sort when API fails."""
        mock_cohere = MagicMock()
        mock_cohere.rerank.side_effect = Exception("API error")
        orchestrator._cohere = mock_cohere

        chunks = [
            Chunk(chunk_id="c1", text="chunk a", doc_id="d1", score=0.5, source="graph"),
            Chunk(chunk_id="c2", text="chunk b", doc_id="d2", score=0.9, source="vector"),
        ]
        result = orchestrator._cohere_rerank("query", chunks, top_k=5)
        assert result[0].score == 0.9  # sorted by score descending

    def test_subgraph_to_chunks(self, orchestrator, mock_graph_builder) -> None:
        """_subgraph_to_chunks creates node and edge chunks."""
        subgraph = mock_graph_builder.get_subgraph.return_value
        chunks = orchestrator._subgraph_to_chunks(subgraph)

        texts = [c.text for c in chunks]
        # Should have node chunks
        assert any("Metformin" in t for t in texts)
        assert any("Type 2 Diabetes" in t for t in texts)
        # Should have edge chunk
        assert any("treats" in t.lower() for t in texts)


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

class TestRouting:

    def test_route_graph_mode(self, orchestrator) -> None:
        """Graph mode routes to graph_retriever."""
        state = {"retrieval_mode": RetrievalMode.GRAPH.value}
        assert orchestrator._route_after_classifier(state) == "graph"

    def test_route_hybrid_mode(self, orchestrator) -> None:
        """Hybrid mode routes to graph_retriever first."""
        state = {"retrieval_mode": RetrievalMode.HYBRID.value}
        assert orchestrator._route_after_classifier(state) == "hybrid"

    def test_route_vector_mode(self, orchestrator) -> None:
        """Vector mode routes to vector_retriever."""
        state = {"retrieval_mode": RetrievalMode.VECTOR.value}
        assert orchestrator._route_after_classifier(state) == "vector"

    def test_route_web_mode(self, orchestrator) -> None:
        """Web mode routes directly to web_search."""
        state = {"retrieval_mode": RetrievalMode.WEB.value}
        assert orchestrator._route_after_classifier(state) == "web"

    def test_route_after_graph_hybrid_goes_to_vector(
        self, orchestrator
    ) -> None:
        """Hybrid mode continues to vector retriever after graph."""
        state = {
            "retrieval_mode": RetrievalMode.HYBRID.value,
            "should_web_search": False,
        }
        assert orchestrator._route_after_graph(state) == "vector"

    def test_route_after_graph_web_trigger(self, orchestrator) -> None:
        """Graph-only mode routes to web when should_web_search is True."""
        state = {
            "retrieval_mode": RetrievalMode.GRAPH.value,
            "should_web_search": True,
        }
        assert orchestrator._route_after_graph(state) == "web"

    def test_route_after_graph_merge(self, orchestrator) -> None:
        """Graph-only mode routes to merge when web not needed."""
        state = {
            "retrieval_mode": RetrievalMode.GRAPH.value,
            "should_web_search": False,
        }
        assert orchestrator._route_after_graph(state) == "merge"

    def test_route_after_vector_web(self, orchestrator) -> None:
        """Vector retriever routes to web when flag is set."""
        state = {"should_web_search": True}
        assert orchestrator._route_after_vector(state) == "web"

    def test_route_after_vector_merge(self, orchestrator) -> None:
        """Vector retriever routes to merge when flag is not set."""
        state = {"should_web_search": False}
        assert orchestrator._route_after_vector(state) == "merge"


# ---------------------------------------------------------------------------
# End-to-end run() tests
# ---------------------------------------------------------------------------

class TestRunEndToEnd:

    def test_run_returns_retrieval_result(
        self, orchestrator_no_apis
    ) -> None:
        """run() returns a RetrievalResult object."""
        from src.retrieval.orchestrator import RetrievalResult

        result = orchestrator_no_apis.run("What treats Type 2 Diabetes?")
        assert isinstance(result, RetrievalResult)

    def test_run_hybrid_mode(
        self, orchestrator_no_apis
    ) -> None:
        """run() with hybrid mode populates graph and vector results."""
        result = orchestrator_no_apis.run(
            "What drugs treat Type 2 Diabetes?",
            mode=RetrievalMode.HYBRID,
        )
        assert result.retrieval_mode == RetrievalMode.HYBRID
        assert len(result.merged_chunks) > 0

    def test_run_vector_only_mode(
        self, orchestrator_no_apis
    ) -> None:
        """run() with vector mode returns chunks without graph traversal."""
        result = orchestrator_no_apis.run(
            "diabetes treatment",
            mode=RetrievalMode.VECTOR,
        )
        assert result.retrieval_mode == RetrievalMode.VECTOR
        assert len(result.vector_chunks) > 0

    def test_run_with_temporal_query(
        self, orchestrator_no_apis
    ) -> None:
        """Temporal query sets web_search_triggered=True even without Tavily."""
        result = orchestrator_no_apis.run(
            "latest diabetes treatment guidelines 2025"
        )
        assert result.web_search_triggered is True

    def test_run_graceful_with_no_dependencies(self) -> None:
        """run() with no dependencies returns empty result without crashing."""
        from src.retrieval.orchestrator import RetrievalOrchestrator
        from unittest.mock import patch

        with patch.dict("os.environ", {"TAVILY_API_KEY": "", "COHERE_API_KEY": ""}):
            orch = RetrievalOrchestrator(
                graph_builder=None,
                vector_indexer=None,
                tavily_api_key="",
                cohere_api_key="",
            )
        result = orch.run("diabetes treatment")
        assert result is not None
        assert result.merged_chunks == []

    def test_run_populates_errors_on_failure(
        self, orchestrator_no_apis, mock_vector_indexer
    ) -> None:
        """run() records errors in the result without raising."""
        mock_vector_indexer.similarity_search.side_effect = Exception("connection lost")
        result = orchestrator_no_apis.run("diabetes", mode=RetrievalMode.VECTOR)
        assert isinstance(result.errors, list)


# ---------------------------------------------------------------------------
# Staleness check
# ---------------------------------------------------------------------------

class TestStalenessCheck:

    def test_stale_node_detected(self, orchestrator) -> None:
        """Node last_updated > 180 days ago is detected as stale."""
        stale_date = datetime.now(timezone.utc) - timedelta(days=200)
        subgraph = GraphSubgraph(
            nodes=[
                GraphNode(
                    id="X",
                    name="Old Drug",
                    node_type=NodeType.DRUG,
                    last_updated=stale_date,
                )
            ],
            edges=[],
        )
        assert orchestrator._has_stale_node(subgraph) is True

    def test_fresh_node_not_stale(self, orchestrator) -> None:
        """Node last_updated within 180 days is not stale."""
        fresh_date = datetime.now(timezone.utc) - timedelta(days=10)
        subgraph = GraphSubgraph(
            nodes=[
                GraphNode(
                    id="X",
                    name="New Drug",
                    node_type=NodeType.DRUG,
                    last_updated=fresh_date,
                )
            ],
            edges=[],
        )
        assert orchestrator._has_stale_node(subgraph) is False

    def test_empty_subgraph_not_stale(self, orchestrator) -> None:
        """Empty subgraph is not considered stale."""
        assert orchestrator._has_stale_node(GraphSubgraph()) is False


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

INTEGRATION = pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS") != "1",
    reason="Set RUN_INTEGRATION_TESTS=1 with docker compose up -d and API keys in .env",
)


@INTEGRATION
def test_full_pipeline_live() -> None:
    """End-to-end retrieval against live Neo4j, Qdrant, Tavily, Cohere."""
    import socket
    from src.graph.builder import GraphBuilder
    from src.vector.indexer import VectorIndexer
    from src.retrieval.orchestrator import RetrievalOrchestrator

    # Check infrastructure reachable
    for host, port in [("localhost", 7687), ("localhost", 6333)]:
        try:
            s = socket.create_connection((host, port), timeout=2)
            s.close()
        except OSError:
            pytest.skip(f"{host}:{port} not reachable")

    gb = GraphBuilder()
    vi = VectorIndexer()
    vi.ensure_collection()

    orch = RetrievalOrchestrator(graph_builder=gb, vector_indexer=vi)
    result = orch.run("What drugs are used to treat Type 2 Diabetes?")

    assert result is not None
    assert result.retrieval_mode in list(RetrievalMode)
    assert isinstance(result.merged_chunks, list)
    assert isinstance(result.errors, list)

    gb.close()