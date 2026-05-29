"""
Shared Pydantic data models for the Medical Knowledge Assistant.

All inter-module data contracts are defined here to ensure type safety
and consistent serialisation across the pipeline.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, HttpUrl, validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class NodeType(str, Enum):
    """Allowed Neo4j node label types."""

    DISEASE = "Disease"
    DRUG = "Drug"
    GENE = "Gene"
    SYMPTOM = "Symptom"
    CLINICAL_TRIAL = "ClinicalTrial"
    PAPER = "Paper"


class EdgeType(str, Enum):
    """Allowed Neo4j relationship types."""

    TREATS = "TREATS"
    CAUSES = "CAUSES"
    INTERACTS_WITH = "INTERACTS_WITH"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    INVESTIGATED_IN = "INVESTIGATED_IN"
    CITED_BY = "CITED_BY"


class RetrievalMode(str, Enum):
    """Query routing modes for the RetrievalOrchestrator."""

    GRAPH = "graph"
    VECTOR = "vector"
    HYBRID = "hybrid"
    WEB = "web"


# ---------------------------------------------------------------------------
# Ingestion models
# ---------------------------------------------------------------------------


class Document(BaseModel):
    """
    A raw document returned by any DataFetcher method.

    This is the canonical unit that flows from ingestion into the NLP layer.
    """

    doc_id: str = Field(..., description="Unique identifier for this document")
    title: str = Field(..., description="Document title")
    abstract: str = Field(default="", description="Abstract or body text")
    source: str = Field(..., description="Data source name, e.g. 'pubmed', 'openfda'")
    source_url: Optional[str] = Field(None, description="Canonical URL of the document")
    publication_date: Optional[datetime] = Field(None, description="Date published")
    authors: List[str] = Field(default_factory=list, description="Author list")
    mesh_terms: List[str] = Field(default_factory=list, description="MeSH annotations")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Source-specific extra fields"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ---------------------------------------------------------------------------
# NLP models
# ---------------------------------------------------------------------------


class Entity(BaseModel):
    """
    A named medical entity extracted from text.

    Produced by NLPProcessor.extract_entities() and optionally enriched
    with a UMLS CUI by NLPProcessor.link_to_umls().
    """

    text: str = Field(..., description="Surface form of the entity as found in text")
    entity_type: NodeType = Field(..., description="Normalised node label type")
    cui: Optional[str] = Field(None, description="UMLS Concept Unique Identifier")
    start_char: int = Field(..., description="Character offset start in source text")
    end_char: int = Field(..., description="Character offset end in source text")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="NER model confidence"
    )
    canonical_name: Optional[str] = Field(
        None, description="Preferred name from UMLS / ICD-11"
    )
    source_doc_id: Optional[str] = Field(
        None, description="doc_id of the Document this entity was extracted from"
    )


class Triple(BaseModel):
    """
    A subject–predicate–object relation triple.

    Produced by NLPProcessor.extract_relations() and consumed by
    GraphBuilder.upsert_batch().
    """

    subject: Entity
    predicate: EdgeType
    obj: Entity  # 'object' is a reserved Python keyword
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source_doc_id: Optional[str] = Field(None)
    year: Optional[int] = Field(None)
    evidence_text: Optional[str] = Field(
        None, description="Sentence(s) providing evidence for this triple"
    )


# ---------------------------------------------------------------------------
# Graph models
# ---------------------------------------------------------------------------


class GraphNode(BaseModel):
    """Represents a node persisted in (or retrieved from) Neo4j."""

    id: str = Field(..., description="UMLS CUI or synthetic unique id")
    name: str
    node_type: NodeType
    source_url: Optional[str] = None
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    properties: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class GraphEdge(BaseModel):
    """Represents a relationship persisted in (or retrieved from) Neo4j."""

    source_id: str
    target_id: str
    edge_type: EdgeType
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source_doc_id: Optional[str] = None
    year: Optional[int] = None
    relation_type: Optional[str] = None


class GraphSubgraph(BaseModel):
    """A subgraph returned by the GraphRetriever."""

    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)
    query_node_ids: List[str] = Field(
        default_factory=list,
        description="Node IDs that directly matched the query (vs. hop neighbours)",
    )
    path_confidence: float = Field(
        default=1.0,
        description="Product of edge confidences along the highest-confidence path",
    )


# ---------------------------------------------------------------------------
# Vector / retrieval models
# ---------------------------------------------------------------------------


class Chunk(BaseModel):
    """A text chunk stored in Qdrant with its embedding."""

    chunk_id: str
    text: str
    doc_id: str
    node_cui: Optional[str] = None
    score: float = Field(default=0.0, description="Similarity score from Qdrant")
    source: str = Field(default="vector")
    source_url: Optional[str] = None
    pub_date: Optional[datetime] = None

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class WebSnippet(BaseModel):
    """A web search result snippet from Tavily."""

    title: str
    url: str
    snippet: str
    pub_date: Optional[str] = None
    source: str = Field(default="web")
    score: float = Field(default=0.0)


# ---------------------------------------------------------------------------
# Answer models
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    """A source citation included in a MedicalAnswer."""

    citation_id: str
    title: str
    url: Optional[str] = None
    source: str = Field(description="'pubmed' | 'openfda' | 'trials' | 'web' | 'graph'")
    year: Optional[int] = None
    authors: List[str] = Field(default_factory=list)
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)


class MedicalAnswer(BaseModel):
    """
    Structured answer returned by LLMInterface.call_llm().

    This is the final output of the entire pipeline and the payload
    returned by the FastAPI POST /query endpoint.
    """

    answer: str = Field(..., description="The synthesised medical answer")
    citations: List[Citation] = Field(default_factory=list)
    graph_path: List[str] = Field(
        default_factory=list,
        description="Ordered list of node names/CUIs traversed in the graph",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Product of edge confidences along the answer path",
    )
    retrieval_mode: RetrievalMode = Field(default=RetrievalMode.HYBRID)
    disclaimer: str = Field(
        default=(
            "⚠️ This information is for educational purposes only and does not "
            "constitute medical advice. Always consult a qualified healthcare "
            "professional before making any medical decisions."
        )
    )
    raw_graph_subgraph: Optional[GraphSubgraph] = Field(
        None, description="Subgraph used in answering — attached for visualisation"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ---------------------------------------------------------------------------
# Query models (FastAPI request)
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Incoming query payload for POST /query."""

    query: str = Field(..., min_length=3, max_length=2000)
    max_results: int = Field(default=10, ge=1, le=50)
    include_graph_viz: bool = Field(
        default=True, description="Whether to embed pyvis HTML in the response"
    )
    retrieval_mode: Optional[RetrievalMode] = Field(
        None,
        description="Force a specific retrieval mode (None = auto-route)",
    )
