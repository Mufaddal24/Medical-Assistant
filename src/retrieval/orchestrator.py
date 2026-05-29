"""
RetrievalOrchestrator — Module 5 of the Medical Knowledge Assistant pipeline.

Implements a LangGraph state machine with five nodes:

  QueryClassifier  → determines retrieval strategy and detects temporal intent
  GraphRetriever   → multi-hop Cypher subgraph traversal (Neo4j)
  VectorRetriever  → BioBERT ANN similarity search (Qdrant)
  WebSearchNode    → Tavily live internet search (conditional)
  HybridMerger     → Cohere re-rank + deduplication

Internet search fires when ANY of:
  (a) A matched graph node has last_updated older than WEB_SEARCH_STALENESS_DAYS
  (b) Query contains temporal keywords: latest, recent, new, current, 2024, 2025, 2026
  (c) Graph + vector retrieval together return fewer than MIN_RESULTS_FOR_WEB chunks

Graph:
  query_classifier
       │
       ├──(graph|hybrid)──► graph_retriever ──(hybrid)──► vector_retriever
       │                          │(graph)                      │
       │                          ▼                             ▼
       ├──(vector)──────────────────────────────────────► check_web?
       │                                                       │
       └──(web)──────────────────────────────────►  web_search_node
                                                          │
                                                          ▼
                                                    hybrid_merger ──► END

Environment variables (loaded from .env)
-----------------------------------------
  TAVILY_API_KEY              Tavily search API key
  COHERE_API_KEY              Cohere rerank API key
  WEB_SEARCH_STALENESS_DAYS   Days before node is stale (default 180)
  COHERE_RERANK_MODEL         Cohere model (default rerank-english-v3.0)
"""

from __future__ import annotations

import logging
import operator
import os
import re
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel

from src.graph.builder import GraphBuilder
from src.utils.models import (
    Chunk,
    GraphSubgraph,
    RetrievalMode,
    WebSnippet,
)
from src.vector.indexer import VectorIndexer

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    from langgraph.graph import END, StateGraph
    from typing import TypedDict
    LANGGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover
    LANGGRAPH_AVAILABLE = False
    logger.warning("langgraph not installed — RetrievalOrchestrator will use fallback mode")

try:
    import cohere as cohere_lib
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False
    logger.warning("cohere not installed — re-ranking disabled")

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    logger.warning("tavily-python not installed — web search disabled")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TEMPORAL_KEYWORDS: set[str] = {
    "latest", "recent", "new", "current", "now", "today",
    "2024", "2025", "2026",
}

_GRAPH_KEYWORDS: set[str] = {
    "relationship", "connected", "path", "link", "between",
    "related to", "connection", "associated",
}

_STALENESS_DAYS: int = int(os.getenv("WEB_SEARCH_STALENESS_DAYS", "180"))
_MIN_RESULTS_FOR_WEB: int = 3
_COHERE_MODEL: str = os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0")
_WEB_SEARCH_MAX: int = 5
_VECTOR_TOP_K: int = 5
_GRAPH_HOPS: int = 2
_MERGE_TOP_K: int = 10


# ---------------------------------------------------------------------------
# LangGraph state definition
# ---------------------------------------------------------------------------

if LANGGRAPH_AVAILABLE:
    class RetrievalState(TypedDict):
        """
        State that flows through every node of the LangGraph pipeline.

        Fields prefixed with Annotated[List, operator.add] are *append-only*
        — each node adds to the list rather than replacing it.
        """
        query: str
        retrieval_mode: str                           # RetrievalMode.value
        graph_subgraph: Optional[GraphSubgraph]
        vector_chunks: Annotated[List[Chunk], operator.add]
        web_snippets: Annotated[List[WebSnippet], operator.add]
        merged_chunks: List[Chunk]                    # final output
        should_web_search: bool
        query_cuis: List[str]                         # CUIs identified by classifier
        errors: Annotated[List[str], operator.add]
else:
    RetrievalState = dict  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------

class RetrievalResult(BaseModel):
    """
    The structured output of RetrievalOrchestrator.run().

    Consumed by PromptBuilder (Module 6) to assemble the LLM prompt.
    """
    query: str
    retrieval_mode: RetrievalMode
    graph_subgraph: Optional[GraphSubgraph] = None
    vector_chunks: List[Chunk] = []
    web_snippets: List[WebSnippet] = []
    merged_chunks: List[Chunk] = []
    web_search_triggered: bool = False
    errors: List[str] = []

    class Config:
        arbitrary_types_allowed = True


# ---------------------------------------------------------------------------
# RetrievalOrchestrator
# ---------------------------------------------------------------------------

class RetrievalOrchestrator:
    """
    Hybrid retrieval pipeline implemented as a LangGraph state machine.

    Parameters
    ----------
    graph_builder:
        Initialised GraphBuilder instance connected to Neo4j.
    vector_indexer:
        Initialised VectorIndexer instance connected to Qdrant.
    tavily_api_key:
        Tavily API key. Falls back to TAVILY_API_KEY env var.
    cohere_api_key:
        Cohere API key. Falls back to COHERE_API_KEY env var.
    force_web_search:
        Always trigger web search regardless of conditions (for testing).

    Example
    -------
    >>> gb = GraphBuilder()
    >>> vi = VectorIndexer()
    >>> orch = RetrievalOrchestrator(gb, vi)
    >>> result = orch.run("What drugs treat Type 2 Diabetes?")
    >>> len(result.merged_chunks)
    8
    """

    def __init__(
        self,
        graph_builder: Optional[GraphBuilder] = None,
        vector_indexer: Optional[VectorIndexer] = None,
        tavily_api_key: Optional[str] = None,
        cohere_api_key: Optional[str] = None,
        force_web_search: bool = False,
    ) -> None:
        self._gb = graph_builder
        self._vi = vector_indexer
        self._force_web = force_web_search

        self._tavily = self._init_tavily(tavily_api_key)
        self._cohere = self._init_cohere(cohere_api_key)

        if LANGGRAPH_AVAILABLE:
            self._app = self._build_graph()
        else:
            self._app = None
            logger.warning("LangGraph unavailable — using sequential fallback")

        logger.info(
            "RetrievalOrchestrator ready "
            "(tavily=%s, cohere=%s, langgraph=%s)",
            self._tavily is not None,
            self._cohere is not None,
            LANGGRAPH_AVAILABLE,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        mode: Optional[RetrievalMode] = None,
    ) -> RetrievalResult:
        """
        Execute the full retrieval pipeline for *query*.

        Parameters
        ----------
        query:
            The user's natural-language medical question.
        mode:
            Force a specific retrieval mode. When ``None`` the
            QueryClassifier determines the mode automatically.

        Returns
        -------
        RetrievalResult
            Structured result containing graph subgraph, vector chunks,
            web snippets, merged/re-ranked chunks, and metadata.
        """
        logger.info("RetrievalOrchestrator.run query=%r mode=%s", query[:80], mode)

        initial_state: RetrievalState = {
            "query": query,
            "retrieval_mode": mode.value if mode else "",
            "graph_subgraph": None,
            "vector_chunks": [],
            "web_snippets": [],
            "merged_chunks": [],
            "should_web_search": self._force_web,
            "query_cuis": [],
            "errors": [],
        }

        if self._app is not None:
            try:
                final_state = self._app.invoke(initial_state)
            except Exception as exc:  # noqa: BLE001
                logger.error("LangGraph pipeline failed: %s — using fallback", exc)
                final_state = self._sequential_fallback(initial_state)
        else:
            final_state = self._sequential_fallback(initial_state)

        return self._build_result(final_state)

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> Any:
        """Compile the LangGraph state machine."""
        workflow: StateGraph = StateGraph(RetrievalState)

        # Register nodes
        workflow.add_node("query_classifier", self._node_classify_query)
        workflow.add_node("graph_retriever",  self._node_graph_retriever)
        workflow.add_node("vector_retriever", self._node_vector_retriever)
        workflow.add_node("web_search",       self._node_web_search)
        workflow.add_node("hybrid_merger",    self._node_hybrid_merger)

        # Entry point
        workflow.set_entry_point("query_classifier")

        # After classifier: route based on detected mode
        workflow.add_conditional_edges(
            "query_classifier",
            self._route_after_classifier,
            {
                "graph":  "graph_retriever",
                "hybrid": "graph_retriever",
                "vector": "vector_retriever",
                "web":    "web_search",
            },
        )

        # After graph retriever: hybrid continues to vector; graph-only checks web
        workflow.add_conditional_edges(
            "graph_retriever",
            self._route_after_graph,
            {
                "vector": "vector_retriever",
                "web":    "web_search",
                "merge":  "hybrid_merger",
            },
        )

        # After vector retriever: check web conditions
        workflow.add_conditional_edges(
            "vector_retriever",
            self._route_after_vector,
            {
                "web":   "web_search",
                "merge": "hybrid_merger",
            },
        )

        # Web search always leads to merger
        workflow.add_edge("web_search", "hybrid_merger")
        workflow.add_edge("hybrid_merger", END)

        return workflow.compile()

    # ------------------------------------------------------------------
    # Node: QueryClassifier
    # ------------------------------------------------------------------

    def _node_classify_query(self, state: RetrievalState) -> Dict[str, Any]:
        """
        Analyse the query and determine:
        - retrieval_mode (graph / vector / hybrid / web)
        - query_cuis (node IDs found in the graph matching query terms)
        - should_web_search (True when temporal keywords are detected)
        """
        query: str = state["query"]
        forced_mode: str = state["retrieval_mode"]

        # Detect temporal keywords → triggers web search
        has_temporal = self._has_temporal_keywords(query)

        # Determine mode (respect forced override)
        if forced_mode:
            mode = forced_mode
        elif has_temporal:
            mode = RetrievalMode.HYBRID.value   # hybrid + web
        elif self._has_graph_keywords(query):
            mode = RetrievalMode.GRAPH.value
        else:
            mode = RetrievalMode.HYBRID.value   # default

        # Look up entity CUIs in the graph for query terms
        query_cuis = self._extract_cuis_from_query(query)

        should_web = state["should_web_search"] or has_temporal

        logger.info(
            "QueryClassifier: mode=%s cuis=%s temporal=%s web=%s",
            mode, query_cuis, has_temporal, should_web,
        )

        return {
            "retrieval_mode": mode,
            "query_cuis": query_cuis,
            "should_web_search": should_web,
        }

    # ------------------------------------------------------------------
    # Node: GraphRetriever
    # ------------------------------------------------------------------

    def _node_graph_retriever(self, state: RetrievalState) -> Dict[str, Any]:
        """
        Retrieve a multi-hop subgraph from Neo4j centred on the query CUIs.
        Also flags should_web_search=True if any matched node is stale.
        """
        if self._gb is None:
            logger.warning("GraphRetriever: GraphBuilder not available")
            return {"graph_subgraph": None, "errors": ["GraphBuilder not configured"]}

        query_cuis: List[str] = state["query_cuis"]

        # If no CUIs from classifier, search by name
        if not query_cuis:
            query_cuis = self._extract_cuis_from_query(state["query"])

        if not query_cuis:
            logger.info("GraphRetriever: no CUIs found — skipping graph retrieval")
            return {"graph_subgraph": None}

        try:
            subgraph: GraphSubgraph = self._gb.get_subgraph(
                query_cuis=query_cuis,
                hops=_GRAPH_HOPS,
            )
            logger.info(
                "GraphRetriever: %d nodes, %d edges, confidence=%.3f",
                len(subgraph.nodes),
                len(subgraph.edges),
                subgraph.path_confidence,
            )

            # Condition (a): stale node triggers web search
            should_web = state["should_web_search"]
            if not should_web and self._has_stale_node(subgraph):
                logger.info("GraphRetriever: stale node detected → triggering web search")
                should_web = True

            return {
                "graph_subgraph": subgraph,
                "should_web_search": should_web,
            }
        except Exception as exc:  # noqa: BLE001
            logger.error("GraphRetriever failed: %s", exc)
            return {"graph_subgraph": None, "errors": [f"GraphRetriever: {exc}"]}

    # ------------------------------------------------------------------
    # Node: VectorRetriever
    # ------------------------------------------------------------------

    def _node_vector_retriever(self, state: RetrievalState) -> Dict[str, Any]:
        """
        Retrieve top-K semantically similar chunks from Qdrant using BioBERT.
        Sets should_web_search=True if combined results fall below threshold.
        """
        if self._vi is None:
            logger.warning("VectorRetriever: VectorIndexer not available")
            return {"errors": ["VectorIndexer not configured"]}

        query: str = state["query"]

        try:
            chunks: List[Chunk] = self._vi.similarity_search(
                query=query,
                top_k=_VECTOR_TOP_K,
            )
            logger.info("VectorRetriever: %d chunks returned", len(chunks))

            # Condition (c): fewer than MIN_RESULTS_FOR_WEB triggers web search
            total = len(chunks) + len(state.get("vector_chunks", []))
            should_web = state["should_web_search"]
            if not should_web and total < _MIN_RESULTS_FOR_WEB:
                logger.info(
                    "VectorRetriever: only %d total results — triggering web search",
                    total,
                )
                should_web = True

            return {
                "vector_chunks": chunks,
                "should_web_search": should_web,
            }
        except Exception as exc:  # noqa: BLE001
            logger.error("VectorRetriever failed: %s", exc)
            return {"errors": [f"VectorRetriever: {exc}"]}

    # ------------------------------------------------------------------
    # Node: WebSearchNode
    # ------------------------------------------------------------------

    def _node_web_search(self, state: RetrievalState) -> Dict[str, Any]:
        """
        Execute a Tavily live web search when should_web_search is True.

        Each result is tagged with its publication date and source='web'
        before being stored in the state.
        """
        if not state["should_web_search"] and not self._force_web:
            logger.info("WebSearchNode: skipping (should_web_search=False)")
            return {}

        if self._tavily is None:
            logger.warning("WebSearchNode: Tavily client not available")
            return {"errors": ["TAVILY_API_KEY not configured"]}

        query: str = state["query"]
        snippets: List[WebSnippet] = []

        try:
            response = self._tavily.search(
                query=query,
                max_results=_WEB_SEARCH_MAX,
            )
            for item in response.get("results", []):
                snippets.append(
                    WebSnippet(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        snippet=item.get("content", "")[:800],
                        pub_date=item.get("published_date"),
                        source="web",
                        score=float(item.get("score", 0.0)),
                    )
                )
            logger.info("WebSearchNode: %d snippets retrieved", len(snippets))
        except Exception as exc:  # noqa: BLE001
            logger.error("WebSearchNode failed: %s", exc)
            return {"errors": [f"WebSearchNode: {exc}"]}

        return {"web_snippets": snippets}

    # ------------------------------------------------------------------
    # Node: HybridMerger
    # ------------------------------------------------------------------

    def _node_hybrid_merger(self, state: RetrievalState) -> Dict[str, Any]:
        """
        Merge graph-derived chunks, vector chunks, and web snippets into a
        single ranked list.

        Strategy:
        1. Convert graph subgraph nodes into Chunk objects
        2. Combine with vector_chunks and web-derived chunks
        3. Deduplicate by text similarity
        4. Re-rank with Cohere if available, otherwise sort by score
        5. Return top _MERGE_TOP_K chunks
        """
        query: str = state["query"]
        all_chunks: List[Chunk] = []

        # 1. Graph subgraph → chunks
        if state.get("graph_subgraph"):
            graph_chunks = self._subgraph_to_chunks(state["graph_subgraph"])
            all_chunks.extend(graph_chunks)

        # 2. Vector chunks
        all_chunks.extend(state.get("vector_chunks", []))

        # 3. Web snippets → chunks
        for snippet in state.get("web_snippets", []):
            all_chunks.append(
                Chunk(
                    chunk_id=f"web_{hash(snippet.url) & 0xFFFFFFFF:08x}",
                    text=f"{snippet.title}. {snippet.snippet}",
                    doc_id=snippet.url,
                    score=snippet.score,
                    source="web",
                    source_url=snippet.url,
                    pub_date=None,
                )
            )

        if not all_chunks:
            logger.warning("HybridMerger: no chunks to merge")
            return {"merged_chunks": []}

        # 4. Deduplicate
        unique_chunks = self._deduplicate(all_chunks)

        # 5. Re-rank or sort
        if self._cohere is not None and len(unique_chunks) > 1:
            ranked = self._cohere_rerank(query, unique_chunks, top_k=_MERGE_TOP_K)
        else:
            ranked = sorted(unique_chunks, key=lambda c: c.score, reverse=True)[:_MERGE_TOP_K]

        logger.info(
            "HybridMerger: %d input → %d unique → %d ranked",
            len(all_chunks),
            len(unique_chunks),
            len(ranked),
        )

        return {"merged_chunks": ranked}

    # ------------------------------------------------------------------
    # Routing functions
    # ------------------------------------------------------------------

    def _route_after_classifier(self, state: RetrievalState) -> str:
        """Route to graph_retriever, vector_retriever, or web_search."""
        mode = state["retrieval_mode"]
        if mode in (RetrievalMode.GRAPH.value, RetrievalMode.HYBRID.value):
            return "graph" if mode == RetrievalMode.GRAPH.value else "hybrid"
        elif mode == RetrievalMode.VECTOR.value:
            return "vector"
        elif mode == RetrievalMode.WEB.value:
            return "web"
        return "hybrid"  # default

    def _route_after_graph(self, state: RetrievalState) -> str:
        """After graph retriever: hybrid continues to vector; graph-only checks web."""
        mode = state["retrieval_mode"]
        if mode == RetrievalMode.HYBRID.value:
            return "vector"
        # graph-only mode
        if state["should_web_search"]:
            return "web"
        return "merge"

    def _route_after_vector(self, state: RetrievalState) -> str:
        """After vector retriever: trigger web search or go straight to merge."""
        if state["should_web_search"]:
            return "web"
        return "merge"

    # ------------------------------------------------------------------
    # Sequential fallback (when LangGraph is unavailable)
    # ------------------------------------------------------------------

    def _sequential_fallback(self, state: RetrievalState) -> RetrievalState:
        """Run all pipeline steps sequentially without LangGraph."""
        state.update(self._node_classify_query(state))
        state.update(self._node_graph_retriever(state))
        state.update(self._node_vector_retriever(state))
        if state.get("should_web_search"):
            state.update(self._node_web_search(state))
        state.update(self._node_hybrid_merger(state))
        return state

    # ------------------------------------------------------------------
    # Helpers — initialisation
    # ------------------------------------------------------------------

    def _init_tavily(self, api_key: Optional[str]) -> Optional[Any]:
        """Initialise Tavily client; returns None if key is missing."""
        key = api_key or os.getenv("TAVILY_API_KEY", "")
        if not key:
            logger.warning("TAVILY_API_KEY not set — web search disabled")
            return None
        if not TAVILY_AVAILABLE:
            logger.warning("tavily-python not installed — web search disabled")
            return None
        try:
            client = TavilyClient(api_key=key)
            return client
        except Exception as exc:  # noqa: BLE001
            logger.error("Tavily init failed: %s", exc)
            return None

    def _init_cohere(self, api_key: Optional[str]) -> Optional[Any]:
        """Initialise Cohere client; returns None if key is missing."""
        key = api_key or os.getenv("COHERE_API_KEY", "")
        if not key:
            logger.warning("COHERE_API_KEY not set — re-ranking disabled")
            return None
        if not COHERE_AVAILABLE:
            logger.warning("cohere not installed — re-ranking disabled")
            return None
        try:
            # cohere v7 uses ClientV2
            if hasattr(cohere_lib, "ClientV2"):
                client = cohere_lib.ClientV2(api_key=key)
            else:
                client = cohere_lib.Client(api_key=key)
            return client
        except Exception as exc:  # noqa: BLE001
            logger.error("Cohere init failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Helpers — query analysis
    # ------------------------------------------------------------------

    def _has_temporal_keywords(self, query: str) -> bool:
        """Return True if the query contains any temporal keyword."""
        words = set(re.findall(r"\b\w+\b", query.lower()))
        return bool(words & _TEMPORAL_KEYWORDS)

    def _has_graph_keywords(self, query: str) -> bool:
        """Return True if the query is asking about graph relationships."""
        lower = query.lower()
        return any(kw in lower for kw in _GRAPH_KEYWORDS)

    def _extract_cuis_from_query(self, query: str) -> List[str]:
        """
        Search the graph for nodes whose names appear in the query.
        Returns a list of node IDs (CUIs or synthetic IDs).
        """
        if self._gb is None or not self._gb.is_connected():
            return []

        cuis: List[str] = []
        # Try multi-word phrases (up to 4 words) then single words
        words = query.split()
        candidates: List[str] = []
        for length in range(min(4, len(words)), 0, -1):
            for start in range(len(words) - length + 1):
                candidates.append(" ".join(words[start: start + length]))

        seen_text: set[str] = set()
        for candidate in candidates:
            if candidate.lower() in seen_text or len(candidate) < 3:
                continue
            try:
                nodes = self._gb.search_nodes_by_name(candidate, limit=2)
                for node in nodes:
                    if node.id not in cuis:
                        cuis.append(node.id)
                        seen_text.add(candidate.lower())
                        if len(cuis) >= 5:   # cap to avoid huge subgraphs
                            return cuis
            except Exception:  # noqa: BLE001
                pass

        return cuis

    # ------------------------------------------------------------------
    # Helpers — staleness check
    # ------------------------------------------------------------------

    def _has_stale_node(self, subgraph: GraphSubgraph) -> bool:
        """
        Return True if any node in the subgraph has last_updated older
        than WEB_SEARCH_STALENESS_DAYS.
        """
        cutoff = datetime.now(timezone.utc).timestamp() - (
            _STALENESS_DAYS * 86400
        )
        for node in subgraph.nodes:
            ts = node.last_updated
            if ts is None:
                continue
            # Ensure tz-aware comparison
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts.timestamp() < cutoff:
                return True
        return False

    # ------------------------------------------------------------------
    # Helpers — graph → chunks conversion
    # ------------------------------------------------------------------

    def _subgraph_to_chunks(self, subgraph: GraphSubgraph) -> List[Chunk]:
        """
        Convert a GraphSubgraph into text Chunk objects for ranking.

        Each node becomes a short descriptive chunk; each edge is
        formatted as a triple sentence.
        """
        chunks: List[Chunk] = []
        node_map = {n.id: n for n in subgraph.nodes}

        # Node chunks
        for node in subgraph.nodes:
            text = f"{node.node_type.value}: {node.name}"
            chunks.append(
                Chunk(
                    chunk_id=f"graph_node_{node.id}",
                    text=text,
                    doc_id=node.id,
                    node_cui=node.id,
                    score=node.confidence_score,
                    source="graph",
                    source_url=node.source_url,
                )
            )

        # Edge chunks (triples as sentences)
        for edge in subgraph.edges:
            src = node_map.get(edge.source_id)
            tgt = node_map.get(edge.target_id)
            if src and tgt:
                text = (
                    f"{src.name} {edge.edge_type.value.lower().replace('_', ' ')} "
                    f"{tgt.name}."
                )
                if edge.year:
                    text += f" (source: {edge.source_doc_id or 'graph'}, {edge.year})"
                chunks.append(
                    Chunk(
                        chunk_id=f"graph_edge_{edge.source_id}_{edge.target_id}",
                        text=text,
                        doc_id=edge.source_doc_id or edge.source_id,
                        node_cui=edge.source_id,
                        score=edge.confidence,
                        source="graph",
                    )
                )

        return chunks

    # ------------------------------------------------------------------
    # Helpers — deduplication
    # ------------------------------------------------------------------

    def _deduplicate(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Remove near-duplicate chunks using a simple normalised-text fingerprint.
        Keeps the highest-scored copy when duplicates are found.
        """
        seen: Dict[str, Chunk] = {}
        for chunk in chunks:
            # Fingerprint: first 120 normalised characters
            fp = re.sub(r"\s+", " ", chunk.text.lower().strip())[:120]
            if fp not in seen or chunk.score > seen[fp].score:
                seen[fp] = chunk
        return list(seen.values())

    # ------------------------------------------------------------------
    # Helpers — Cohere rerank
    # ------------------------------------------------------------------

    def _cohere_rerank(
        self,
        query: str,
        chunks: List[Chunk],
        top_k: int = _MERGE_TOP_K,
    ) -> List[Chunk]:
        """
        Re-rank chunks using the Cohere Rerank API.

        Falls back to score-based sorting if the API call fails.
        """
        if not chunks:
            return []

        documents = [c.text[:512] for c in chunks]   # API has token limits

        try:
            response = self._cohere.rerank(
                model=_COHERE_MODEL,
                query=query,
                documents=documents,
                top_n=min(top_k, len(chunks)),
            )
            reranked: List[Chunk] = []
            for result in response.results:
                chunk = chunks[result.index]
                # Update score with Cohere relevance score
                reranked.append(
                    chunk.model_copy(update={"score": float(result.relevance_score)})
                )
            logger.info(
                "Cohere rerank: %d → %d chunks (top score=%.3f)",
                len(chunks),
                len(reranked),
                reranked[0].score if reranked else 0.0,
            )
            return reranked

        except Exception as exc:  # noqa: BLE001
            logger.warning("Cohere rerank failed (%s) — using score sort", exc)
            return sorted(chunks, key=lambda c: c.score, reverse=True)[:top_k]

    # ------------------------------------------------------------------
    # Result builder
    # ------------------------------------------------------------------

    def _build_result(self, state: RetrievalState) -> RetrievalResult:
        """Convert the final LangGraph state to a RetrievalResult."""
        try:
            mode = RetrievalMode(state.get("retrieval_mode", "hybrid"))
        except ValueError:
            mode = RetrievalMode.HYBRID

        return RetrievalResult(
            query=state.get("query", ""),
            retrieval_mode=mode,
            graph_subgraph=state.get("graph_subgraph"),
            vector_chunks=state.get("vector_chunks", []),
            web_snippets=state.get("web_snippets", []),
            merged_chunks=state.get("merged_chunks", []),
            web_search_triggered=bool(state.get("should_web_search", False)),
            errors=state.get("errors", []),
        )
