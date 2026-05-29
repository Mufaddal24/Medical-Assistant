"""
Module 6 — PromptBuilder
Assembles the final prompt string passed to the LLM.

Responsibilities:
- Inject a medical safety disclaimer into the system message
- Format graph triples as structured context
- Include vector chunks as supporting evidence with source URLs
- Tag web snippets with pub_date and a 'web' source label
- Request a structured JSON response matching MedicalAnswer
- Keep total prompt size under ~12 000 tokens (~48 000 chars)
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

from src.utils.models import (
    Chunk,
    GraphSubgraph,
    WebSnippet,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MEDICAL_DISCLAIMER: str = (
    "⚠️  MEDICAL DISCLAIMER: This system is for educational and research "
    "purposes only. It does NOT constitute medical advice, diagnosis, or "
    "treatment. Always consult a qualified healthcare professional before "
    "making any medical decisions. Information may be incomplete or outdated."
)

# Approximate character budget to stay well under 12 000 tokens
# (GPT-4o tokenises at ~4 chars/token; 12 000 tokens ≈ 48 000 chars)
_MAX_PROMPT_CHARS: int = 44_000

# Per-section character budgets (approximate)
_BUDGET_GRAPH: int = 8_000
_BUDGET_VECTOR: int = 10_000
_BUDGET_WEB: int = 6_000
_BUDGET_MERGED: int = 10_000

# JSON response schema injected into the prompt
_RESPONSE_SCHEMA: str = json.dumps(
    {
        "answer": "<string — synthesised medical answer>",
        "citations": [
            {
                "citation_id": "<unique short id, e.g. 'c1'>",
                "title": "<source title>",
                "url": "<source URL or empty string>",
                "source": "<pubmed | openfda | clinicaltrials | umls | web>",
                "year": "<int or null>",
                "authors": ["<author names>"],
                "relevance_score": "<float 0-1>",
            }
        ],
        "graph_path": ["<ordered node names / CUIs traversed>"],
        "confidence": "<float 0-1 — product of edge confidences along the answer path>",
    },
    indent=2,
)


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------


class PromptBuilder:
    """Assemble a structured prompt for the LLM from retrieval results.

    Usage::

        builder = PromptBuilder()
        prompt = builder.build(
            query="What drugs treat Type 2 Diabetes?",
            graph_subgraph=subgraph,
            vector_chunks=chunks,
            web_snippets=snippets,
            merged_chunks=merged,
        )
    """

    def __init__(self, max_prompt_chars: int = _MAX_PROMPT_CHARS) -> None:
        """Initialise the builder.

        Args:
            max_prompt_chars: Hard character ceiling for the assembled prompt.
                Sections are trimmed proportionally when this limit is approached.
        """
        self.max_prompt_chars = max_prompt_chars

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        query: str,
        graph_subgraph: Optional[GraphSubgraph],
        vector_chunks: List[Chunk],
        web_snippets: List[WebSnippet],
        merged_chunks: List[Chunk],
    ) -> str:
        """Assemble the final prompt string to pass to the LLM.

        The returned string is a single block that concatenates:
        1. System message with role definition and medical disclaimer
        2. Graph context (nodes + edge triples from Neo4j)
        3. Vector evidence (BioBERT-retrieved text chunks)
        4. Web search results (Tavily snippets, if present)
        5. Merged / re-ranked context summary
        6. JSON output schema
        7. The user question

        Args:
            query: The original natural-language medical question.
            graph_subgraph: Multi-hop subgraph returned by GraphRetriever
                (may be ``None`` if graph retrieval was skipped or empty).
            vector_chunks: Top-K BioBERT similarity chunks from Qdrant.
            web_snippets: Tavily live-search results (empty list if not triggered).
            merged_chunks: Final Cohere-re-ranked, deduplicated chunk list.

        Returns:
            A formatted prompt string ready for ``LLMInterface.call_llm()``.
        """
        sections: List[str] = []

        # 1 — System / role block
        sections.append(self._build_system_block())

        # 2 — Graph context
        graph_block = self._build_graph_block(graph_subgraph)
        if graph_block:
            sections.append(graph_block)

        # 3 — Vector evidence
        vector_block = self._build_vector_block(vector_chunks)
        if vector_block:
            sections.append(vector_block)

        # 4 — Web search results
        web_block = self._build_web_block(web_snippets)
        if web_block:
            sections.append(web_block)

        # 5 — Merged / re-ranked context
        merged_block = self._build_merged_block(merged_chunks)
        if merged_block:
            sections.append(merged_block)

        # 6 — Output schema
        sections.append(self._build_schema_block())

        # 7 — User question
        sections.append(self._build_question_block(query))

        prompt = "\n\n".join(sections)

        # Safety trim — keep within token budget
        if len(prompt) > self.max_prompt_chars:
            logger.warning(
                "Prompt length %d exceeds budget %d — trimming.",
                len(prompt),
                self.max_prompt_chars,
            )
            question_tail = self._build_question_block(query)
            prompt = (
                prompt[: self.max_prompt_chars - len(question_tail) - 4]
                + "\n\n"
                + question_tail
            )

        logger.info("PromptBuilder assembled prompt (%d chars).", len(prompt))
        return prompt

    def build_messages(
        self,
        query: str,
        graph_subgraph: Optional[GraphSubgraph],
        vector_chunks: List[Chunk],
        web_snippets: List[WebSnippet],
        merged_chunks: List[Chunk],
    ) -> List[dict]:
        """Return OpenAI-style ``messages`` list instead of a flat string.

        Splits the prompt into a ``system`` message (disclaimer + role) and a
        ``user`` message (all context + question), which is the recommended
        format for GPT-4o chat completions.

        Returns:
            List of ``{"role": ..., "content": ...}`` dicts.
        """
        system_content = self._build_system_block()

        user_sections: List[str] = []

        graph_block = self._build_graph_block(graph_subgraph)
        if graph_block:
            user_sections.append(graph_block)

        vector_block = self._build_vector_block(vector_chunks)
        if vector_block:
            user_sections.append(vector_block)

        web_block = self._build_web_block(web_snippets)
        if web_block:
            user_sections.append(web_block)

        merged_block = self._build_merged_block(merged_chunks)
        if merged_block:
            user_sections.append(merged_block)

        user_sections.append(self._build_schema_block())
        user_sections.append(self._build_question_block(query))

        user_content = "\n\n".join(user_sections)

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    # ------------------------------------------------------------------
    # Private section builders
    # ------------------------------------------------------------------

    def _build_system_block(self) -> str:
        """Return the system-role preamble including the medical disclaimer."""
        return (
            "=== SYSTEM ===\n"
            "You are a biomedical knowledge assistant that answers clinical and "
            "research questions using a knowledge graph, semantic search, and "
            "real-time web sources. You synthesise evidence from multiple sources "
            "into a clear, accurate, and well-cited answer.\n\n"
            f"{MEDICAL_DISCLAIMER}\n\n"
            "Rules:\n"
            "1. Base every claim on the provided context. Do NOT hallucinate facts.\n"
            "2. Cite every source you use (include citation_id in your answer text).\n"
            "3. If the evidence is insufficient, say so explicitly.\n"
            "4. Return ONLY valid JSON matching the schema below — no markdown fences, "
            "no prose outside the JSON object."
        )

    def _build_graph_block(self, subgraph: Optional[GraphSubgraph]) -> str:
        """Format Neo4j subgraph nodes and edges as readable triples."""
        if subgraph is None:
            return ""
        if not subgraph.nodes and not subgraph.edges:
            return ""

        lines: List[str] = [
            f"=== KNOWLEDGE GRAPH CONTEXT (path_confidence={subgraph.path_confidence:.3f}) ==="
        ]

        # Nodes
        if subgraph.nodes:
            lines.append("-- Entities --")
            for node in subgraph.nodes:
                highlight = " [QUERY MATCH]" if node.id in subgraph.query_node_ids else ""
                url_note = f" | url={node.source_url}" if node.source_url else ""
                lines.append(
                    f"  [{node.node_type.value}] {node.name} (id={node.id}"
                    f", conf={node.confidence_score:.2f}{url_note}){highlight}"
                )

        # Edges as triples
        if subgraph.edges:
            id_to_name = {n.id: n.name for n in subgraph.nodes}
            lines.append("-- Relations --")
            for edge in subgraph.edges:
                src = id_to_name.get(edge.source_id, edge.source_id)
                tgt = id_to_name.get(edge.target_id, edge.target_id)
                year_note = f" ({edge.year})" if edge.year else ""
                lines.append(
                    f"  {src} -[{edge.edge_type.value}, conf={edge.confidence:.2f}]"
                    f"-> {tgt}{year_note}"
                )

        block = "\n".join(lines)
        if len(block) > _BUDGET_GRAPH:
            block = block[: _BUDGET_GRAPH] + "\n  ... [graph context trimmed]"
        return block

    def _build_vector_block(self, chunks: List[Chunk]) -> str:
        """Format Qdrant BioBERT chunks as numbered evidence passages."""
        if not chunks:
            return ""

        lines: List[str] = ["=== VECTOR SEARCH EVIDENCE (BioBERT semantic similarity) ==="]
        budget_per_chunk = max(200, _BUDGET_VECTOR // max(len(chunks), 1))

        for i, chunk in enumerate(chunks, start=1):
            text = chunk.text[:budget_per_chunk].strip()
            url_note = f" | {chunk.source_url}" if chunk.source_url else ""
            date_note = f" | {chunk.pub_date}" if chunk.pub_date else ""
            # score is a float field (non-optional) — always display it
            score_note = f" | score={chunk.score:.3f}"
            lines.append(
                f"[V{i}] source={chunk.source}{url_note}{date_note}{score_note}\n"
                f"      {text}"
            )

        block = "\n".join(lines)
        if len(block) > _BUDGET_VECTOR:
            block = block[: _BUDGET_VECTOR] + "\n  ... [vector evidence trimmed]"
        return block

    def _build_web_block(self, snippets: List[WebSnippet]) -> str:
        """Format Tavily web snippets with publication dates and web labels."""
        if not snippets:
            return ""

        lines: List[str] = ["=== LIVE WEB SEARCH RESULTS (Tavily) ==="]
        budget_per_snippet = max(200, _BUDGET_WEB // max(len(snippets), 1))

        for i, snip in enumerate(snippets, start=1):
            text = snip.snippet[:budget_per_snippet].strip()
            date_note = f" | published={snip.pub_date}" if snip.pub_date else ""
            # score is a float field (non-optional) — always display it
            score_note = f" | score={snip.score:.3f}"
            lines.append(
                f"[W{i}] [WEB] {snip.title}{date_note}{score_note}\n"
                f"      url={snip.url}\n"
                f"      {text}"
            )

        block = "\n".join(lines)
        if len(block) > _BUDGET_WEB:
            block = block[: _BUDGET_WEB] + "\n  ... [web results trimmed]"
        return block

    def _build_merged_block(self, chunks: List[Chunk]) -> str:
        """Format the final re-ranked merged chunk list."""
        if not chunks:
            return ""

        lines: List[str] = ["=== MERGED & RE-RANKED CONTEXT (top results across all sources) ==="]
        budget_per_chunk = max(200, _BUDGET_MERGED // max(len(chunks), 1))

        for i, chunk in enumerate(chunks, start=1):
            text = chunk.text[:budget_per_chunk].strip()
            url_note = f" | {chunk.source_url}" if chunk.source_url else ""
            date_note = f" | {chunk.pub_date}" if chunk.pub_date else ""
            score_note = f" | score={chunk.score:.3f}"
            lines.append(
                f"[M{i}] source={chunk.source}{url_note}{date_note}{score_note}\n"
                f"      {text}"
            )

        block = "\n".join(lines)
        if len(block) > _BUDGET_MERGED:
            block = block[: _BUDGET_MERGED] + "\n  ... [merged context trimmed]"
        return block

    def _build_schema_block(self) -> str:
        """Return the JSON output schema the LLM must follow."""
        return (
            "=== REQUIRED JSON OUTPUT SCHEMA ===\n"
            "Return ONLY a single JSON object with this exact structure "
            "(no markdown, no extra keys):\n"
            f"{_RESPONSE_SCHEMA}"
        )

    def _build_question_block(self, query: str) -> str:
        """Return the user question block."""
        return f"=== QUESTION ===\n{query}"