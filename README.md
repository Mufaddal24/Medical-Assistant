# Medical Knowledge Assistant
### Graph-RAG System with Live Internet Search Augmentation

A production-quality medical question-answering system that constructs a
knowledge graph from real biomedical data sources, retrieves answers through
hybrid graph + vector search, and augments with live internet search when
needed. Built on Neo4j, Qdrant, BioBERT, LangGraph, and GPT-4o.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [High-Level Design (HLD)](#2-high-level-design-hld)
3. [Low-Level Design (LLD)](#3-low-level-design-lld)
4. [Project Structure](#4-project-structure)
5. [Modules Implemented](#5-modules-implemented)
   - [Module 0 — Shared Models](#module-0--shared-models-srcutilsmodelspy)
   - [Module 1 — DataFetcher](#module-1--datafetcher-srcdatafetcherpy)
   - [Module 2 — NLPProcessor](#module-2--nlpprocessor-srcnlpprocessorpy)
   - [Module 3 — GraphBuilder](#module-3--graphbuilder-srcgraphbuilderpy)
   - [Module 4 — VectorIndexer](#module-4--vectorindexer-srcvectorindexerpy)
   - [Module 5 — RetrievalOrchestrator](#module-5--retrievalorchestrator-srcretrievalorchestratorp)
   - [Module 6 — PromptBuilder](#module-6--promptbuilder-srcgenerationprompt_builderpy)
   - [Module 7 — LLMInterface](#module-7--llminterface-srcgenerationllm_interfacepy)
6. [Modules Pending](#6-modules-pending)
7. [Data Flow Walkthrough](#7-data-flow-walkthrough)
8. [Graph Schema](#8-graph-schema)
9. [Setup & Installation](#9-setup--installation)
10. [Running the Tests](#10-running-the-tests)
11. [Environment Variables](#11-environment-variables)
12. [Docker Infrastructure](#12-docker-infrastructure)

---

## 1. Project Overview

The Medical Knowledge Assistant answers clinical and biomedical questions by:

1. **Ingesting** real documents from PubMed, OpenFDA, ClinicalTrials.gov, and UMLS.
2. **Extracting** medical entities (diseases, drugs, genes, symptoms) and the
   relations between them using biomedical NLP.
3. **Persisting** those entities and relations as a property graph in Neo4j.
4. **Retrieving** relevant subgraphs via multi-hop Cypher traversal and relevant
   text chunks via BioBERT vector similarity in Qdrant.
5. **Augmenting** with live Tavily internet search when the local knowledge base
   is stale or insufficient.
6. **Generating** a structured answer with citations, a traversed graph path,
   and a confidence score using GPT-4o.

**Seed condition for all demos:** `"Type 2 Diabetes"`

---

## 2. High-Level Design (HLD)

The system is organised into six horizontal layers. Data flows top-to-bottom
during ingestion and bottom-to-top during query answering.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LAYER 1 — INGESTION                          │
│                                                                     │
│   PubMed ──┐                                                        │
│  OpenFDA ──┼──► DataFetcher ──► List[Document]                      │
│  Trials  ──┤                                                        │
│   UMLS   ──┘                                                        │
└─────────────────────────────┬───────────────────────────────────────┘
                              │ List[Document]
┌─────────────────────────────▼───────────────────────────────────────┐
│                         LAYER 2 — NLP                               │
│                                                                     │
│   NLPProcessor                                                      │
│     ├── extract_entities(text)  ──► List[Entity]                    │
│     ├── extract_relations(text) ──► List[Triple]                    │
│     └── link_to_umls(entity)    ──► CUI string                      │
└──────────────┬──────────────────────────────┬───────────────────────┘
               │ List[Triple]                 │ List[Entity] + vectors
┌──────────────▼──────────────┐  ┌────────────▼───────────────────────┐
│    LAYER 3 — GRAPH STORE    │  │     LAYER 3b — VECTOR STORE        │
│                             │  │                                    │
│   GraphBuilder              │  │   VectorIndexer                    │
│     ├── add_node()          │  │     ├── embed_chunk(text)          │
│     ├── add_edge()          │  │     ├── upsert_to_qdrant()         │
│     └── upsert_batch()      │  │     └── similarity_search()        │
│                             │  │                                    │
│   Neo4j Property Graph      │  │   Qdrant Vector Collections        │
│   (nodes + relationships)   │  │   (BioBERT embeddings)             │
└──────────────┬──────────────┘  └────────────┬───────────────────────┘
               │                              │
┌──────────────▼──────────────────────────────▼───────────────────────┐
│                      LAYER 4 — RETRIEVAL                            │
│                                                                     │
│   RetrievalOrchestrator  (LangGraph state machine)                  │
│     ├── QueryClassifier   ──► routes to: graph|vector|hybrid|web    │
│     ├── GraphRetriever    ──► multi-hop Cypher subgraph             │
│     ├── VectorRetriever   ──► top-K BioBERT similarity chunks       │
│     ├── WebSearchNode     ──► Tavily live search (conditional)      │
│     └── HybridMerger      ──► Cohere re-rank + deduplication        │
└─────────────────────────────┬───────────────────────────────────────┘
                              │ merged context
┌─────────────────────────────▼───────────────────────────────────────┐
│                      LAYER 5 — AUGMENTATION                         │
│                                                                     │
│   PromptBuilder                                                     │
│     └── build(graph_triples, vector_chunks, web_snippets) ► str     │
│         • injects medical safety disclaimer                         │
│         • structures context for LLM consumption                    │
└─────────────────────────────┬───────────────────────────────────────┘
                              │ structured prompt
┌─────────────────────────────▼───────────────────────────────────────┐
│                      LAYER 6 — GENERATION                           │
│                                                                     │
│   LLMInterface                                                      │
│     └── call_llm(prompt) ──► MedicalAnswer                          │
│         • answer (str)                                              │
│         • citations (List[Citation])                                │
│         • graph_path (List[str])                                    │
│         • confidence (float)                                        │
└─────────────────────────────────────────────────────────────────────┘
```

### HLD Component Descriptions

#### Layer 1 — Ingestion
Responsible for pulling raw documents from four authoritative medical data
sources and normalising them into a single `Document` schema. This layer is
the only part of the system that communicates with external APIs. It is
designed to be fault-tolerant: if any one source fails, the others continue.
All HTTP calls use exponential back-off retry logic.

#### Layer 2 — NLP
Transforms unstructured text (titles, abstracts, drug labels) into structured
knowledge. It uses scispaCy's biomedical NER model (`en_core_sci_lg`) to
identify medical entities and a rule-based relation extraction system to
identify how those entities relate to each other. The UMLS entity linker
maps surface forms to canonical Concept Unique Identifiers (CUIs), enabling
cross-document entity merging.

#### Layer 3 — Graph Store
Persists the structured knowledge extracted in Layer 2 as a property graph
in Neo4j. Every write is idempotent — using Cypher's `MERGE` instead of
`CREATE` — so re-running ingestion never creates duplicate nodes or edges.
Confidence scores on edges propagate through multi-hop paths as a product,
giving the system a quantitative measure of answer reliability.

#### Layer 3b — Vector Store
Runs in parallel with the graph store. Each text chunk is embedded using
BioBERT (`dmis-lab/biobert-base-cased-v1.2`) and stored in Qdrant, a
high-performance vector database. This enables semantic similarity search
that captures concepts even when exact entity names differ.

#### Layer 4 — Retrieval
The most complex layer, implemented as a LangGraph state machine. A
`QueryClassifier` first determines the best retrieval strategy for the
incoming query. A `GraphRetriever` runs multi-hop Cypher queries. A
`VectorRetriever` performs BioBERT similarity search. A `WebSearchNode`
conditionally calls the Tavily API when the local knowledge base is
insufficient or stale. A `HybridMerger` deduplicates and re-ranks all
results using the Cohere Rerank API.

**Internet search fires when ANY condition is true:**
- The matched graph node `last_updated` is older than 180 days
- The query contains temporal keywords: `latest`, `recent`, `new`, `current`, `2024`, `2025`, `2026`
- Graph + vector retrieval together return fewer than 3 result chunks

#### Layer 5 — Augmentation
The `PromptBuilder` assembles a structured prompt from the retrieved graph
triples, vector chunks, and web snippets. It always prepends a medical safety
disclaimer and requests a structured JSON response with citations.

#### Layer 6 — Generation
`LLMInterface` sends the assembled prompt to GPT-4o (with Llama-3-8B via
Ollama as a local fallback). The LLM returns a `MedicalAnswer` object
containing the answer text, a list of citations with source URLs, the
graph path traversed, and a confidence score.

---

## 3. Low-Level Design (LLD)

### Module Map

```
src/
├── utils/
│   └── models.py              ← Shared Pydantic data contracts (all modules)
├── data/
│   └── fetcher.py             ← Module 1: DataFetcher
├── nlp/
│   └── processor.py           ← Module 2: NLPProcessor
├── graph/
│   └── builder.py             ← Module 3: GraphBuilder
├── vector/
│   └── indexer.py             ← Module 4: VectorIndexer
├── retrieval/
│   └── orchestrator.py        ← Module 5: RetrievalOrchestrator
├── generation/
│   ├── prompt_builder.py      ← Module 6: PromptBuilder
│   └── llm_interface.py       ← Module 7: LLMInterface
├── api/
│   └── app.py                 ← Module 9: FastAPI app           [pending]
├── ui/
│   └── streamlit_app.py       ← Module 10: Streamlit UI         [pending]
└── tests/
    ├── test_data_nlp.py        ← Tests for Modules 1 & 2
    └── test_graph_builder.py   ← Tests for Module 3
```

### LLD Component Descriptions

#### `src/utils/models.py` — Shared Data Contracts
The single source of truth for all data schemas. Every module imports from
here. Prevents circular dependencies and ensures consistent serialisation
across the entire pipeline. Uses Pydantic v2 for runtime type validation.

**Key types:**
- `Document` — raw document from any data source
- `Entity` — a named medical entity with type, CUI, and character offsets
- `Triple` — a subject–predicate–object relation (the unit of graph knowledge)
- `GraphNode` / `GraphEdge` — Neo4j persistence models
- `GraphSubgraph` — a retrieved subgraph with path confidence
- `Chunk` — a vector-indexed text chunk from Qdrant
- `WebSnippet` — a Tavily search result
- `MedicalAnswer` — the final pipeline output
- `Citation` — a source reference included in the answer

#### `src/data/fetcher.py` — DataFetcher
Communicates with four external APIs. Each fetch method is independent and
returns `List[Document]`. Uses `_get_with_retry()` for fault-tolerant HTTP
with exponential back-off. All methods return an empty list on failure —
never raise — so the pipeline continues even if one source is unavailable.

#### `src/nlp/processor.py` — NLPProcessor
Stateless text processing. Load once, call repeatedly. Two execution paths:
production (scispaCy `en_core_sci_lg`) and fallback (curated regex patterns
for the diabetes domain). Relation extraction works at sentence level to
minimise false positives from long-range co-occurrence.

#### `src/graph/builder.py` — GraphBuilder
Stateful Neo4j connection wrapper. Manages the driver lifecycle, exposes
clean CRUD methods, and handles all Cypher. The `upsert_batch()` method is
the primary ingestion entry point — it processes a list of triples in a
single pass, writing nodes before edges to respect referential integrity.

#### `src/vector/indexer.py` — VectorIndexer
Embeds text chunks with BioBERT and stores/retrieves them from Qdrant.
Implements lazy model loading (BioBERT is only loaded on the first
`embed_chunk()` call), batched embedding for efficiency, and deterministic
`chunk_id → UUID` mapping so the same document can be re-ingested without
creating duplicate points. Gracefully degrades to zero vectors when
`transformers`/`torch` are not installed, and to dry-run mode when Qdrant
is unreachable.

#### `src/retrieval/orchestrator.py` — RetrievalOrchestrator
LangGraph state machine with five nodes: `QueryClassifier`, `GraphRetriever`,
`VectorRetriever`, `WebSearchNode`, and `HybridMerger`. The graph is compiled
at init time and invoked per query. Each node returns a partial state update.
Conditional edges route between graph-only, vector-only, hybrid, and web-first
strategies. Falls back to sequential execution when LangGraph is unavailable.
Cohere Rerank re-scores the merged result set; deduplication uses a normalised
120-character text fingerprint. Returns a `RetrievalResult` containing the
subgraph, vector chunks, web snippets, and merged final ranking.

#### `src/generation/prompt_builder.py` — PromptBuilder
Assembles the final LLM prompt from all retrieved context. Splits output
into a `system` message (role definition + medical disclaimer) and a `user`
message (graph triples, vector chunks, web snippets, JSON output schema, and
the question). Exposes both `build()` (flat string) and `build_messages()`
(OpenAI-style list) to support different LLM call patterns. Each section has
an independent character budget; total prompt is hard-capped at ~44 000 chars
(≈ 11 000 tokens) with the question always preserved at the tail even after
trimming. Always displays relevance scores; handles `score=0.0` correctly
(does not suppress falsy float scores).

#### `src/generation/llm_interface.py` — LLMInterface
Single-responsibility LLM call wrapper. Accepts a prompt from
`PromptBuilder.build()` (flat string) or `PromptBuilder.build_messages()`
(OpenAI-style list) interchangeably via `Union[str, List[dict]]`. Tries
GPT-4o first via the OpenAI Chat Completions API with JSON mode enabled;
falls back to Llama-3-8B via Ollama on any failure; returns a graceful
fallback `MedicalAnswer` if both providers fail — never raises. Clients
are initialised lazily on first use. Derives `MedicalAnswer.confidence`
from `graph_subgraph.path_confidence` (product of Neo4j edge confidences)
when available, otherwise uses the LLM-reported value. Handles all real-world
LLM output patterns: clean JSON, markdown-fenced JSON, and JSON embedded in
prose.

#### `src/api/app.py` — FastAPI App *(pending)*
Two endpoints: `POST /query` accepts a `QueryRequest` and returns a
`MedicalAnswer` JSON; `GET /graph` returns a pyvis HTML visualisation
of the subgraph used in the last answer.

#### `src/ui/streamlit_app.py` — Streamlit UI *(pending)*
Single-page app with a query input box, answer panel, embedded graph
visualisation (the pyvis HTML rendered in an iframe), and a citation
list with clickable source links.

---

## 4. Project Structure

```
medical-kg-assistant/
├── src/
│   ├── utils/
│   │   └── models.py
│   ├── data/
│   │   └── fetcher.py
│   ├── nlp/
│   │   └── processor.py
│   ├── graph/
│   │   └── builder.py
│   ├── vector/
│   │   └── indexer.py
│   ├── retrieval/
│   │   └── orchestrator.py
│   ├── generation/
│   │   ├── prompt_builder.py
│   │   └── llm_interface.py
│   ├── api/                       (pending)
│   ├── ui/                        (pending)
│   └── tests/
│       ├── test_data_nlp.py
│       ├── test_graph_builder.py
│       ├── test_vector_indexer.py
│       ├── test_orchestrator.py
│       ├── test_prompt_builder.py
│       └── test_llm_interface.py
├── docker-compose.yml
├── requirements.txt
├── pytest.ini
├── conftest.py
├── .env.example
├── .gitignore
└── README.md
```

---

## 5. Modules Implemented

---

### Module 0 — Shared Models (`src/utils/models.py`)

The foundation of the entire project. All other modules import their data
types from here. No module-to-module imports exist except through this file,
which prevents circular dependencies.

#### Enums

| Enum | Values | Purpose |
|------|--------|---------|
| `NodeType` | `Disease`, `Drug`, `Gene`, `Symptom`, `ClinicalTrial`, `Paper` | Valid Neo4j node labels |
| `EdgeType` | `TREATS`, `CAUSES`, `INTERACTS_WITH`, `ASSOCIATED_WITH`, `INVESTIGATED_IN`, `CITED_BY` | Valid Neo4j relationship types |
| `RetrievalMode` | `graph`, `vector`, `hybrid`, `web` | Query routing decisions |

#### Pydantic Models

---

##### `Document`
The universal output of every DataFetcher method. Normalises data from
PubMed, OpenFDA, ClinicalTrials, and UMLS into a single schema.

| Field | Type | Description |
|-------|------|-------------|
| `doc_id` | `str` | MD5 hash of `source::raw_id` — unique per document |
| `title` | `str` | Document title |
| `abstract` | `str` | Abstract, body text, or flattened label sections |
| `source` | `str` | Origin: `pubmed`, `openfda`, `clinicaltrials`, `umls` |
| `source_url` | `Optional[str]` | Canonical URL for the document |
| `publication_date` | `Optional[datetime]` | Date published |
| `authors` | `List[str]` | Author names |
| `mesh_terms` | `List[str]` | MeSH annotations (PubMed only) |
| `metadata` | `Dict[str, Any]` | Source-specific extra fields (e.g. NCT ID, brand names) |

---

##### `Entity`
A named medical entity extracted from text by NLPProcessor.

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | Surface form as it appears in the source text |
| `entity_type` | `NodeType` | Normalised node label type |
| `cui` | `Optional[str]` | UMLS Concept Unique Identifier (e.g. `C0025598`) |
| `start_char` | `int` | Character offset start in source text |
| `end_char` | `int` | Character offset end in source text |
| `confidence` | `float` | NER model confidence score (0–1) |
| `canonical_name` | `Optional[str]` | Preferred name from UMLS |
| `source_doc_id` | `Optional[str]` | The `doc_id` this entity was extracted from |

---

##### `Triple`
A subject–predicate–object relation. The atomic unit of knowledge that
flows from NLPProcessor into GraphBuilder.

| Field | Type | Description |
|-------|------|-------------|
| `subject` | `Entity` | The subject entity |
| `predicate` | `EdgeType` | The relationship type |
| `obj` | `Entity` | The object entity |
| `confidence` | `float` | Relation confidence score (0–1) |
| `source_doc_id` | `Optional[str]` | Provenance document |
| `year` | `Optional[int]` | Publication year |
| `evidence_text` | `Optional[str]` | The sentence this triple was extracted from |

---

##### `GraphNode`
Mirrors a node stored in Neo4j.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | UMLS CUI or synthetic `SYN_` hash id |
| `name` | `str` | Human-readable entity name |
| `node_type` | `NodeType` | Neo4j label |
| `source_url` | `Optional[str]` | Origin URL |
| `last_updated` | `datetime` | Timestamp of last write (used for staleness check) |
| `confidence_score` | `float` | NER confidence at time of extraction |

---

##### `GraphEdge`
Mirrors a relationship stored in Neo4j.

| Field | Type | Description |
|-------|------|-------------|
| `source_id` | `str` | `id` of the source node |
| `target_id` | `str` | `id` of the target node |
| `edge_type` | `EdgeType` | Relationship type |
| `confidence` | `float` | Relation confidence (used in path confidence calculation) |
| `source_doc_id` | `Optional[str]` | Provenance document |
| `year` | `Optional[int]` | Publication year |

---

##### `GraphSubgraph`
The result of a multi-hop graph retrieval.

| Field | Type | Description |
|-------|------|-------------|
| `nodes` | `List[GraphNode]` | All nodes in the subgraph |
| `edges` | `List[GraphEdge]` | All edges in the subgraph |
| `query_node_ids` | `List[str]` | Node IDs that directly matched the query |
| `path_confidence` | `float` | Product of all edge confidences along the path |

---

##### `MedicalAnswer`
The final output of the entire pipeline. Returned by `LLMInterface.call_llm()`
and by the FastAPI `POST /query` endpoint.

| Field | Type | Description |
|-------|------|-------------|
| `answer` | `str` | The synthesised medical answer |
| `citations` | `List[Citation]` | Source references used |
| `graph_path` | `List[str]` | Ordered node names/CUIs traversed |
| `confidence` | `float` | Product of edge confidences along the answer path |
| `retrieval_mode` | `RetrievalMode` | Which retrieval strategy was used |
| `disclaimer` | `str` | Medical safety disclaimer (always included) |
| `raw_graph_subgraph` | `Optional[GraphSubgraph]` | Attached for visualisation |

---

### Module 1 — DataFetcher (`src/data/fetcher.py`)

#### Position in the pipeline
```
[External APIs] ──► DataFetcher ──► List[Document] ──► NLPProcessor
```
This is the first module to execute during ingestion. It has no dependencies
on other project modules — only on external APIs and the `Document` model.

#### Class: `DataFetcher`

**Constructor**
```python
DataFetcher(
    pubmed_email: Optional[str] = None,    # loaded from PUBMED_EMAIL env var
    umls_api_key: Optional[str] = None,    # loaded from UMLS_API_KEY env var
    openfda_api_key: Optional[str] = None  # loaded from OPENFDA_API_KEY env var
)
```

---

##### `fetch_pubmed(query, max_results) → List[Document]`

Fetches PubMed articles matching a search query.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | required | Free-text or MeSH query, e.g. `"Type 2 Diabetes[MeSH Terms]"` |
| `max_results` | `int` | `10` | Maximum articles to retrieve |

**Returns:** `List[Document]` — one Document per article with `source="pubmed"`

**Behaviour:**
- Uses the `pymed` library when installed; falls back to NCBI E-utilities REST API
- Parses title, abstract, authors, MeSH terms, and publication date
- Sets `source_url` to `https://pubmed.ncbi.nlm.nih.gov/{pmid}/`
- Returns `[]` on any network error — never raises

**Example:**
```python
fetcher = DataFetcher(pubmed_email="you@example.com")
docs = fetcher.fetch_pubmed("Type 2 Diabetes[MeSH Terms]", max_results=5)
# docs[0].source == "pubmed"
# docs[0].mesh_terms == ["Diabetes Mellitus, Type 2", ...]
```

---

##### `fetch_openfda(drug_name, max_results) → List[Document]`

Fetches FDA drug label records from the openFDA API.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `drug_name` | `str` | required | Generic or brand name, e.g. `"metformin"` |
| `max_results` | `int` | `5` | Maximum label records to return |

**Returns:** `List[Document]` — one Document per FDA label with `source="openfda"`

**Behaviour:**
- Queries `GET https://api.fda.gov/drug/label.json`
- Flattens `description`, `indications_and_usage`, `warnings`, `adverse_reactions`,
  `drug_interactions`, and `mechanism_of_action` into the `abstract` field
- Sets `source_url` to the FDA application page
- Returns `[]` on 404 (drug not found) or any other error

**Example:**
```python
docs = fetcher.fetch_openfda("metformin", max_results=2)
# docs[0].title == "FDA Label: Glucophage"
# docs[0].metadata["brand_names"] == ["Glucophage"]
# docs[0].metadata["generic_names"] == ["METFORMIN HYDROCHLORIDE"]
```

---

##### `fetch_trials(condition, max_results) → List[Document]`

Fetches clinical trials from ClinicalTrials.gov v2 REST API.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `condition` | `str` | required | Disease or condition, e.g. `"Type 2 Diabetes"` |
| `max_results` | `int` | `10` | Maximum trials to return |

**Returns:** `List[Document]` — one Document per trial with `source="clinicaltrials"`

**Behaviour:**
- Queries `GET https://clinicaltrials.gov/api/v2/studies`
- Parses the nested `protocolSection` structure
- Sets `source_url` to `https://clinicaltrials.gov/study/{nct_id}`
- `metadata` contains `nct_id`, `phase`, `status`, `interventions`, `conditions`

**Example:**
```python
docs = fetcher.fetch_trials("Type 2 Diabetes", max_results=3)
# docs[0].metadata["nct_id"] == "NCT01234567"
# docs[0].metadata["status"] == "COMPLETED"
# docs[0].metadata["interventions"] == ["Metformin", "Placebo"]
```

---

##### `fetch_umls_concepts(term, max_results) → List[Document]`

Searches the UMLS Metathesaurus for concepts matching a term.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `term` | `str` | required | Medical term to look up |
| `max_results` | `int` | `5` | Maximum concepts to return |

**Returns:** `List[Document]` — one Document per UMLS concept with `source="umls"`

**Behaviour:**
- Requires `UMLS_API_KEY` in environment — returns `[]` without it
- Uses two-step UMLS authentication: TGT → service ticket → search
- `metadata` contains `cui`, `name`, and `root_source`

---

##### `fetch_all(condition, drug_name, max_results_per_source) → List[Document]`

Convenience method that calls all four sources and returns deduplicated results.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `condition` | `str` | required | Medical condition to query |
| `drug_name` | `Optional[str]` | `None` | Drug name for OpenFDA (defaults to condition) |
| `max_results_per_source` | `int` | `5` | Maximum results per source |

**Returns:** `List[Document]` — combined, deduplicated by `doc_id`

---

#### Internal helpers

| Method | Description |
|--------|-------------|
| `_make_doc_id(source, raw_id)` | Returns MD5 hash of `"source::raw_id"` — deterministic and URL-safe |
| `_get_with_retry(url, params)` | GET request with exponential back-off (delays: 1s, 2s, 4s) |

---

### Module 2 — NLPProcessor (`src/nlp/processor.py`)

#### Position in the pipeline
```
List[Document] ──► NLPProcessor ──► List[Entity] + List[Triple] ──► GraphBuilder
```
Receives Documents from DataFetcher. Produces Entities and Triples consumed
by GraphBuilder (for Neo4j) and VectorIndexer (for Qdrant).

#### Class: `NLPProcessor`

**Constructor**
```python
NLPProcessor(
    model_name: str = "en_core_sci_lg",     # scispaCy model name
    enable_entity_linker: bool = True,       # load UMLS KB linker
    linker_name: str = "umls"               # linker identifier
)
```

**Execution paths:**
- **Production:** scispaCy `en_core_sci_lg` NER + UMLS entity linker
- **Fallback (CI / offline):** curated regex patterns for the diabetes domain

---

##### `extract_entities(text, source_doc_id) → List[Entity]`

Extracts biomedical named entities from text.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | required | Raw text to process (title + abstract recommended) |
| `source_doc_id` | `Optional[str]` | `None` | Attached to each entity for provenance |

**Returns:** `List[Entity]` — de-duplicated by canonical lowercase form

**Behaviour:**
- With scispaCy: runs full NER pipeline, maps labels via `_LABEL_MAP` to `NodeType`
- Without scispaCy: regex patterns cover drugs (metformin, insulin, etc.),
  diseases (Type 2 Diabetes, T2DM), symptoms (hyperglycemia, neuropathy),
  and genes (HbA1c, GLUT4, TCF7L2)
- De-duplicates entities by `(entity_type, lowercase_text)` to avoid
  counting the same mention twice

**Label mapping (scispaCy → NodeType):**

| scispaCy Label | NodeType |
|----------------|----------|
| `DISEASE`, `DISORDER` | `Disease` |
| `SIGN_OR_SYMPTOM`, `SYMPTOM` | `Symptom` |
| `CHEMICAL`, `DRUG`, `SIMPLE_CHEMICAL` | `Drug` |
| `GENE_OR_GENE_PRODUCT`, `GENE`, `PROTEIN` | `Gene` |

---

##### `extract_relations(text, source_doc_id, year) → List[Triple]`

Extracts subject–predicate–object relation triples from text.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | required | Source text to process |
| `source_doc_id` | `Optional[str]` | `None` | Provenance document ID |
| `year` | `Optional[int]` | `None` | Publication year, attached to triples |

**Returns:** `List[Triple]` — one triple per entity pair per sentence where a
relation keyword is detected

**Behaviour:**
1. Splits text into sentences
2. Runs `extract_entities()` on each sentence
3. For sentences with ≥ 2 entities, calls `_classify_relation()` to find
   the best-matching relation keyword
4. Pairs subject (first entity) with all subsequent entities in the sentence
5. Falls back to `ASSOCIATED_WITH` with confidence 0.3 if no keyword found

**Relation patterns:**

| Keyword(s) | EdgeType | Confidence |
|------------|----------|------------|
| `treats`, `treatment of`, `therapy for` | `TREATS` | 0.6 |
| `causes`, `induces`, `leads to` | `CAUSES` | 0.6 |
| `interacts with`, `drug interaction` | `INTERACTS_WITH` | 0.6 |
| `associated with`, `linked to` | `ASSOCIATED_WITH` | 0.4 |
| `investigated in`, `clinical trial` | `INVESTIGATED_IN` | 0.6 |
| `cites`, `cited by`, `referenced in` | `CITED_BY` | 0.6 |
| *(no match)* | `ASSOCIATED_WITH` | 0.3 |

**Key design detail:** `_classify_relation()` selects the pattern whose match
appears **earliest** in the sentence. This prevents incidental words later in
the sentence from overriding the primary relation (e.g. `"Metformin interacts
with pioglitazone and may cause acidosis"` correctly returns `INTERACTS_WITH`
not `CAUSES`).

---

##### `link_to_umls(entity) → str`

Maps an Entity surface form to a UMLS CUI.

| Parameter | Type | Description |
|-----------|------|-------------|
| `entity` | `Entity` | The entity to link |

**Returns:** `str` — the CUI (e.g. `"C0025598"`), or `""` if linking fails

**Behaviour:**
- If `entity.cui` is already populated, returns it immediately (idempotent)
- If scispaCy linker is loaded, runs the entity text through the pipeline
  and returns the top-scored CUI from `ent._.kb_ents`
- Returns `""` gracefully if the linker is not available

---

##### `process_document(doc) → Tuple[List[Entity], List[Triple]]`

Convenience entry point: processes a complete Document in one call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `doc` | `Document` | A Document from DataFetcher |

**Returns:** `Tuple[List[Entity], List[Triple]]`

**Behaviour:**
1. Concatenates `doc.title + ". " + doc.abstract`
2. Calls `extract_entities()` and enriches each entity with a UMLS CUI
3. Calls `extract_relations()` with the document year
4. Returns `(entities, triples)` ready for `GraphBuilder.upsert_batch()`

---

### Module 3 — GraphBuilder (`src/graph/builder.py`)

#### Position in the pipeline
```
List[Triple] ──► GraphBuilder ──► Neo4j
                     │
              get_subgraph() ◄── RetrievalOrchestrator (query time)
```
Consumes Triples produced by NLPProcessor. Also serves the RetrievalOrchestrator
at query time by returning subgraphs.

#### Class: `GraphBuilder`

**Constructor**
```python
GraphBuilder(
    uri: Optional[str] = None,       # loaded from NEO4J_URI env var
    user: Optional[str] = None,      # loaded from NEO4J_USER env var
    password: Optional[str] = None,  # loaded from NEO4J_PASSWORD env var
    database: str = "neo4j"
)
```

Supports use as a context manager:
```python
with GraphBuilder() as gb:
    gb.upsert_batch(triples)
# driver automatically closed on exit
```

---

##### `create_indexes() → None`

Creates Neo4j indexes on the `id` property for all node labels.

**Behaviour:**
- Uses `CREATE INDEX ... IF NOT EXISTS` — safe to call repeatedly
- Creates one index per `NodeType`: Disease, Drug, Gene, Symptom, ClinicalTrial, Paper
- Should be called once on application startup before any writes

---

##### `add_node(entity, extra_props) → Optional[GraphNode]`

MERGEs a single entity node into Neo4j.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `entity` | `Entity` | required | Entity to persist |
| `extra_props` | `Optional[Dict]` | `None` | Additional properties to set |

**Returns:** `Optional[GraphNode]` — the persisted node, or `None` on failure

**Cypher behaviour (`MERGE` semantics):**
- `ON CREATE`: sets all properties on first write
- `ON MATCH`: updates `name` (if non-empty), `source_url` (if provided),
  and `last_updated` timestamp — preserves existing data otherwise

**Node ID strategy:**
- Uses `entity.cui` when available (e.g. `"C0025598"`)
- Generates `"SYN_" + MD5(type::lowercase_text)[:12]` when no CUI exists
- This ensures the same concept from different documents merges to one node

---

##### `add_edge(source_entity, relation, target_entity, confidence, source_doc_id, year) → bool`

MERGEs a directed relationship between two entities.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source_entity` | `Entity` | required | Subject entity |
| `relation` | `EdgeType` | required | Relationship type |
| `target_entity` | `Entity` | required | Object entity |
| `confidence` | `float` | `0.5` | Relation confidence (0–1) |
| `source_doc_id` | `Optional[str]` | `None` | Provenance document |
| `year` | `Optional[int]` | `None` | Publication year |

**Returns:** `bool` — `True` if written successfully, `False` on error

**Cypher behaviour (`MERGE` semantics):**
- `ON CREATE`: sets all edge properties
- `ON MATCH`: keeps the **higher** confidence score (never downgrades)

---

##### `upsert_batch(triples) → Tuple[int, int]`

Bulk idempotent write of a list of Triple objects.

| Parameter | Type | Description |
|-----------|------|-------------|
| `triples` | `List[Triple]` | Triples from NLPProcessor |

**Returns:** `Tuple[int, int]` — `(nodes_written, edges_written)`

**Behaviour:**
1. For each triple: writes subject node → object node → edge
2. Skips the edge write if either node write fails
3. Logs a summary at INFO level on completion

**Example:**
```python
gb = GraphBuilder()
gb.create_indexes()
nodes, edges = gb.upsert_batch(triples)
# nodes=42, edges=18
```

---

##### `get_subgraph(query_cuis, hops) → GraphSubgraph`

Retrieves a multi-hop subgraph centred on given CUIs.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query_cuis` | `List[str]` | required | Starting node IDs |
| `hops` | `int` | `2` | Number of relationship hops to traverse |

**Returns:** `GraphSubgraph` with nodes, edges, and `path_confidence`

**Confidence propagation:**
```
path_confidence = edge_1.confidence × edge_2.confidence × ... × edge_n.confidence
```
For a 2-hop path with edges of confidence 0.8 and 0.5, `path_confidence = 0.4`.
This value is surfaced in `MedicalAnswer.confidence`.

---

##### `search_nodes_by_name(name, node_type, limit) → List[GraphNode]`

Case-insensitive partial name search across the graph.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | required | Partial or full node name |
| `node_type` | `Optional[NodeType]` | `None` | Filter by label |
| `limit` | `int` | `10` | Max results |

**Returns:** `List[GraphNode]`

---

##### `get_graph_stats() → Dict[str, Any]`

Returns node counts per label and total edge count.

**Returns:** `Dict[str, Any]` — e.g.
```python
{
    "Disease": 42,
    "Drug": 18,
    "Gene": 7,
    "Symptom": 11,
    "ClinicalTrial": 5,
    "Paper": 23,
    "total_edges": 89
}
```

---

---

### Module 4 — VectorIndexer (`src/vector/indexer.py`)

#### Position in the pipeline
```
List[Document] ──► VectorIndexer ──► Qdrant
                        │
              similarity_search() ◄── RetrievalOrchestrator (query time)
```
Runs in parallel with GraphBuilder during ingestion. Each Document is chunked
and embedded, then stored in Qdrant. At query time, the RetrievalOrchestrator
calls `similarity_search()` to retrieve the most semantically relevant chunks
via BioBERT cosine similarity.

#### Class: `VectorIndexer`

**Constructor**
```python
VectorIndexer(
    model_name: Optional[str] = None,         # loaded from BIOBERT_MODEL env var
    qdrant_host: Optional[str] = None,         # loaded from QDRANT_HOST env var
    qdrant_port: Optional[int] = None,         # loaded from QDRANT_PORT env var
    qdrant_api_key: Optional[str] = None,      # loaded from QDRANT_API_KEY env var
    collection_name: Optional[str] = None,     # loaded from QDRANT_COLLECTION env var
    device: Optional[str] = None              # "cuda" or "cpu" — auto-detected
)
```

**Execution paths:**
- **Production:** BioBERT `dmis-lab/biobert-base-cased-v1.2` loaded via HuggingFace
- **Fallback (no model):** zero vectors returned — pipeline continues without embeddings
- **Fallback (no Qdrant):** dry-run mode — all writes silently succeed, searches return `[]`

---

##### `ensure_collection() → bool`

Creates the Qdrant collection if it does not already exist.

**Returns:** `bool` — `True` if the collection exists or was created

**Behaviour:**
- Uses cosine distance — matches BioBERT's representation space
- Vector dimension is fixed at 768 (BioBERT output size)
- Safe to call repeatedly — checks for existence before creating
- Must be called once before any upsert operations

**Example:**
```python
indexer = VectorIndexer()
indexer.ensure_collection()   # idempotent
```

---

##### `embed_chunk(text) → List[float]`

Generates a single BioBERT embedding for a text string.

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | Text to embed — truncated to 512 tokens (BioBERT limit) |

**Returns:** `List[float]` — 768-dimensional embedding vector

**Behaviour:**
- Lazy-loads BioBERT on first call (model stays in memory for subsequent calls)
- Uses **mean-pooling** over the final hidden states of all non-padding tokens
  to produce a stable sentence-level representation
- Returns a zero vector `[0.0] * 768` if BioBERT is unavailable
- Handles empty text gracefully without raising

**Mean-pooling formula:**
```
embedding = sum(token_hidden_states * attention_mask) / sum(attention_mask)
```
This is preferred over CLS-token pooling for sentence similarity tasks.

---

##### `embed_batch(texts) → List[List[float]]`

Generates BioBERT embeddings for multiple texts efficiently.

| Parameter | Type | Description |
|-----------|------|-------------|
| `texts` | `List[str]` | Texts to embed |

**Returns:** `List[List[float]]` — one 768-dimensional vector per input text

**Behaviour:**
- Processes texts in batches of 16 (`_BATCH_EMBED_SIZE`) to avoid OOM errors
- Single tokeniser + forward pass per batch — much faster than calling
  `embed_chunk()` in a loop
- Pads remaining slots with zero vectors if a batch fails mid-way

---

##### `chunk_document(doc) → List[Chunk]`

Splits a Document into overlapping text chunks ready for indexing.

| Parameter | Type | Description |
|-----------|------|-------------|
| `doc` | `Document` | A Document from DataFetcher |

**Returns:** `List[Chunk]` — one Chunk per text segment

**Chunking strategy:**
- Concatenates `doc.title + ". " + doc.abstract`
- Splits into 400-character chunks with 80-character overlap
- Overlap preserves context continuity at chunk boundaries
- Each `chunk_id` is deterministic: `"{doc_id}_chunk_{index:04d}"`
- All metadata (source, source_url, pub_date) is propagated from the parent Document

**Example:**
```python
chunks = indexer.chunk_document(doc)
# chunks[0].chunk_id == "abc123_chunk_0000"
# chunks[1].text starts 80 chars before where chunks[0] ends
```

---

##### `upsert_to_qdrant(chunk, vector) → bool`

Upserts a single Chunk into Qdrant.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chunk` | `Chunk` | required | The Chunk to store |
| `vector` | `Optional[List[float]]` | `None` | Pre-computed vector; generated if not provided |

**Returns:** `bool` — `True` if the upsert succeeded

**Behaviour:**
- Converts `chunk_id` to a deterministic UUID via MD5 hash
- Stores the full chunk payload alongside the vector for retrieval
- Qdrant upsert semantics: same point ID overwrites the previous version
  (idempotent re-ingestion)

---

##### `upsert_batch(chunks) → Tuple[int, int]`

Batch upserts chunks with pre-computed embeddings.

| Parameter | Type | Description |
|-----------|------|-------------|
| `chunks` | `List[Chunk]` | Chunks to index |

**Returns:** `Tuple[int, int]` — `(attempted, succeeded)` counts

**Behaviour:**
- Calls `embed_batch()` once for all chunks (single forward pass)
- Sends all points to Qdrant in a single batch call
- Much more efficient than calling `upsert_to_qdrant()` in a loop

---

##### `upsert_document(doc) → int`

Full ingestion pipeline for a single Document.

| Parameter | Type | Description |
|-----------|------|-------------|
| `doc` | `Document` | A Document from DataFetcher |

**Returns:** `int` — number of chunks successfully written to Qdrant

**Behaviour:**
1. Calls `chunk_document(doc)` to split into text segments
2. Calls `upsert_batch(chunks)` to embed and store all segments
3. Returns the count of successfully written chunks

**Example:**
```python
indexer = VectorIndexer()
indexer.ensure_collection()
count = indexer.upsert_document(doc)
# count == 2  (for a short abstract split into 2 chunks)
```

---

##### `similarity_search(query, top_k, score_threshold, filter_source) → List[Chunk]`

Searches for the most semantically similar chunks to a query.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | required | Natural-language query |
| `top_k` | `int` | `5` | Maximum results to return |
| `score_threshold` | `float` | `0.0` | Minimum cosine similarity (0–1) |
| `filter_source` | `Optional[str]` | `None` | Filter by source (e.g. `"pubmed"`) |

**Returns:** `List[Chunk]` — top-k results ordered by descending similarity score

**Behaviour:**
- Embeds `query` with BioBERT using `embed_chunk()`
- Calls Qdrant `query_points()` (the modern API, replaces the deprecated `.search()`)
- Applies optional payload filter when `filter_source` is set
- Returns `[]` on any error — never raises

**Example:**
```python
results = indexer.similarity_search("what drugs treat diabetes", top_k=5)
# results[0].score == 0.91
# results[0].text == "Metformin is a first-line treatment..."
# results[0].source_url == "https://pubmed.ncbi.nlm.nih.gov/12345/"
```

---

##### `delete_by_doc_id(doc_id) → int`

Deletes all Qdrant points belonging to a document.

| Parameter | Type | Description |
|-----------|------|-------------|
| `doc_id` | `str` | The document ID whose chunks should be removed |

**Returns:** `int` — number of points deleted

**Behaviour:**
- Scrolls through all matching points in batches of 100
- Used before re-ingesting an updated document to avoid stale chunks

---

##### `get_collection_stats() → Dict[str, Any]`

Returns statistics about the Qdrant collection.

**Returns:** `Dict[str, Any]` — e.g.
```python
{
    "name": "med_kg_chunks",
    "vectors_count": 142,
    "indexed_vectors_count": 142,
    "status": "green",
    "dimension": 768,
    "distance": "cosine"
}
```

---

#### Internal helpers

| Helper | Description |
|--------|-------------|
| `_text_to_chunks(text, chunk_size, overlap)` | Splits text into overlapping character-level chunks |
| `_chunk_id_to_uuid(chunk_id)` | Converts a string chunk_id to a Qdrant-compatible UUID via MD5 |
| `_load_model()` | Lazy-loads BioBERT tokeniser and model on first embed call |
| `_connect_qdrant()` | Establishes Qdrant connection with `check_compatibility=False` to avoid version mismatch errors |


---

### Module 5 — RetrievalOrchestrator (`src/retrieval/orchestrator.py`)

#### Position in the pipeline
```
GraphBuilder ──┐
               ├──► RetrievalOrchestrator ──► RetrievalResult ──► PromptBuilder
VectorIndexer ─┘          │
                     Tavily (web)
                     Cohere (rerank)
```
The central query-time component. Receives a user query, decides the best
retrieval strategy, fetches relevant context from all sources, and returns
a ranked `RetrievalResult` ready for the PromptBuilder.

#### LangGraph State Machine

```
                    ┌──(graph|hybrid)──► graph_retriever ──(hybrid)──► vector_retriever
query_classifier ───┤                         │(graph only)                  │
                    ├──(vector)───────────────────────────────────────► check web?
                    │                                                        │
                    └──(web)────────────────────────────────►  web_search ◄──┘
                                                                  │
                                                            hybrid_merger ──► END
```

**Internet search fires when ANY condition is true:**
- **(a)** A matched graph node has `last_updated` older than `WEB_SEARCH_STALENESS_DAYS` (default 180)
- **(b)** Query contains temporal keywords: `latest`, `recent`, `new`, `current`, `2024`, `2025`, `2026`
- **(c)** Graph + vector retrieval together return fewer than 3 result chunks

#### Class: `RetrievalOrchestrator`

**Constructor**
```python
RetrievalOrchestrator(
    graph_builder: Optional[GraphBuilder] = None,
    vector_indexer: Optional[VectorIndexer] = None,
    tavily_api_key: Optional[str] = None,    # loaded from TAVILY_API_KEY env var
    cohere_api_key: Optional[str] = None,    # loaded from COHERE_API_KEY env var
    force_web_search: bool = False           # always trigger web (for testing)
)
```

**Graceful degradation:** Every dependency is optional. The orchestrator starts
and runs even with no Neo4j, no Qdrant, no Tavily, and no Cohere. Missing
components produce empty results and logged warnings rather than exceptions.

---

##### `run(query, mode) → RetrievalResult`

Execute the full retrieval pipeline for a query.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | required | Natural-language medical question |
| `mode` | `Optional[RetrievalMode]` | `None` | Force a retrieval mode; `None` = auto-route |

**Returns:** `RetrievalResult` — structured result with all retrieved context

**Example:**
```python
orch = RetrievalOrchestrator(graph_builder=gb, vector_indexer=vi)
result = orch.run("What drugs treat Type 2 Diabetes?")
# result.retrieval_mode == RetrievalMode.HYBRID
# result.merged_chunks[0].score == 0.91
# result.web_search_triggered == False
```

---

#### Output Model: `RetrievalResult`

| Field | Type | Description |
|-------|------|-------------|
| `query` | `str` | The original user query |
| `retrieval_mode` | `RetrievalMode` | Mode used: graph, vector, hybrid, or web |
| `graph_subgraph` | `Optional[GraphSubgraph]` | Multi-hop subgraph from Neo4j |
| `vector_chunks` | `List[Chunk]` | Top-K BioBERT similarity results from Qdrant |
| `web_snippets` | `List[WebSnippet]` | Tavily search results (empty if not triggered) |
| `merged_chunks` | `List[Chunk]` | Final re-ranked and deduplicated result list |
| `web_search_triggered` | `bool` | Whether internet search fired |
| `errors` | `List[str]` | Non-fatal errors logged during retrieval |

---

#### Node: `_node_classify_query`

Analyses the query to determine retrieval strategy.

**Routing table:**

| Condition | Mode set |
|-----------|----------|
| Temporal keywords present | `hybrid` (+ web flagged) |
| Graph keywords (`relationship`, `path`, `connected`) | `graph` |
| Default | `hybrid` |
| Mode explicitly forced | Uses forced value |

---

#### Node: `_node_graph_retriever`

Retrieves a multi-hop subgraph from Neo4j. Checks every returned node's
`last_updated` — sets `should_web_search=True` if any node is older than
`WEB_SEARCH_STALENESS_DAYS` (condition a).

---

#### Node: `_node_vector_retriever`

Retrieves semantically similar chunks from Qdrant via BioBERT. Sets
`should_web_search=True` if total results < 3 (condition c).

---

#### Node: `_node_web_search`

Executes a Tavily live web search when triggered. Skips entirely when
`should_web_search=False`. Truncates snippet content to 800 characters
and tags every result with `source="web"`.

---

#### Node: `_node_hybrid_merger`

Merges all sources into a single ranked list.

1. Converts `graph_subgraph` nodes → `Chunk` objects (one per node + one per edge triple)
2. Combines with `vector_chunks` and web-converted chunks
3. Deduplicates using a 120-character normalised text fingerprint
4. Re-ranks with Cohere Rerank API when available; falls back to score-descending sort
5. Returns top `_MERGE_TOP_K` (10) chunks

---

#### Internal helpers

| Helper | Description |
|--------|-------------|
| `_has_temporal_keywords(query)` | Checks for temporal words triggering condition (b) |
| `_has_graph_keywords(query)` | Detects relationship-focused queries |
| `_extract_cuis_from_query(query)` | Searches graph for entity names in the query (up to 5 CUIs) |
| `_has_stale_node(subgraph)` | Returns True if any node `last_updated` > staleness threshold |
| `_subgraph_to_chunks(subgraph)` | Converts Neo4j nodes and edges into Chunk objects for ranking |
| `_deduplicate(chunks)` | Removes near-duplicate chunks by text fingerprint |
| `_cohere_rerank(query, chunks, top_k)` | Calls Cohere Rerank API; falls back to score sort on error |
| `_sequential_fallback(state)` | Runs all nodes in sequence when LangGraph is unavailable |

---

### Module 6 — PromptBuilder (`src/generation/prompt_builder.py`)

#### Position in the pipeline
```
RetrievalResult ──► PromptBuilder ──► str / List[dict] ──► LLMInterface
```
Receives the `RetrievalResult` from the orchestrator (graph subgraph, vector
chunks, web snippets, merged ranking) and assembles a structured prompt ready
for GPT-4o. Has no external dependencies — pure Python string manipulation.

#### Class: `PromptBuilder`

**Constructor**
```python
PromptBuilder(
    max_prompt_chars: int = 44_000  # hard ceiling ≈ 11 000 GPT-4o tokens
)
```

---

##### `build(query, graph_subgraph, vector_chunks, web_snippets, merged_chunks) → str`

Assemble a flat prompt string for direct injection into an LLM call.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | required | Original natural-language question |
| `graph_subgraph` | `Optional[GraphSubgraph]` | required | Neo4j subgraph (or `None`) |
| `vector_chunks` | `List[Chunk]` | required | BioBERT similarity results |
| `web_snippets` | `List[WebSnippet]` | required | Tavily live results (empty if not triggered) |
| `merged_chunks` | `List[Chunk]` | required | Final re-ranked merged list |

**Returns:** `str` — a single formatted prompt string

**Prompt structure (7 labelled sections):**

| Section | Header | Content |
|---------|--------|---------|
| 1 | `=== SYSTEM ===` | Role definition + medical safety disclaimer + output rules |
| 2 | `=== KNOWLEDGE GRAPH CONTEXT ===` | Neo4j nodes with `[QUERY MATCH]` markers and edge triples |
| 3 | `=== VECTOR SEARCH EVIDENCE ===` | BioBERT chunks indexed `[V1]`, `[V2]`, … with scores and source URLs |
| 4 | `=== LIVE WEB SEARCH RESULTS ===` | Tavily snippets tagged `[WEB]` with `pub_date` indexed `[W1]`, `[W2]`, … |
| 5 | `=== MERGED & RE-RANKED CONTEXT ===` | Cohere-ranked final list indexed `[M1]`, `[M2]`, … |
| 6 | `=== REQUIRED JSON OUTPUT SCHEMA ===` | Exact JSON structure the LLM must return |
| 7 | `=== QUESTION ===` | User query — always last, always preserved after trimming |

Sections 2–5 are omitted when their source list is empty (no orphan headers).
Each section has an independent character budget; total prompt is hard-capped
at `max_prompt_chars` (default 44 000). If the limit is hit, sections are
trimmed from the middle outward and the question is always re-attached at the
end.

**Example:**
```python
builder = PromptBuilder()
prompt = builder.build(
    query="What drugs treat Type 2 Diabetes?",
    graph_subgraph=result.graph_subgraph,
    vector_chunks=result.vector_chunks,
    web_snippets=result.web_snippets,
    merged_chunks=result.merged_chunks,
)
# prompt starts with "=== SYSTEM ===" and ends with "=== QUESTION ==="
# len(prompt) <= 44_000
```

---

##### `build_messages(query, graph_subgraph, vector_chunks, web_snippets, merged_chunks) → List[dict]`

Return OpenAI-style `messages` list for GPT-4o chat completions.

**Returns:** `[{"role": "system", "content": ...}, {"role": "user", "content": ...}]`

The `system` message contains the role definition and medical disclaimer.
The `user` message contains all context sections, the output schema, and the
question. This is the recommended format when calling the OpenAI Chat API.

**Example:**
```python
messages = builder.build_messages(query, subgraph, chunks, snippets, merged)
response = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
)
```

---

#### Section character budgets

| Section | Budget |
|---------|--------|
| Graph context | 8 000 chars |
| Vector evidence | 10 000 chars |
| Web results | 6 000 chars |
| Merged context | 10 000 chars |
| Total hard cap | 44 000 chars |

Each section trims its own content independently when over budget, appending
`... [section trimmed]` so the LLM always knows context was cut.

---

#### Medical safety disclaimer

The following disclaimer is always injected into the system message and cannot
be suppressed:

```
⚠️  MEDICAL DISCLAIMER: This system is for educational and research purposes
only. It does NOT constitute medical advice, diagnosis, or treatment. Always
consult a qualified healthcare professional before making any medical
decisions. Information may be incomplete or outdated.
```

---

#### Internal helpers

| Helper | Description |
|--------|-------------|
| `_build_system_block()` | Role definition + disclaimer + output rules |
| `_build_graph_block(subgraph)` | Formats nodes with `[QUERY MATCH]` markers and edge triples |
| `_build_vector_block(chunks)` | Formats BioBERT chunks as `[V1]…[Vn]` indexed passages |
| `_build_web_block(snippets)` | Formats Tavily snippets as `[W1]…[Wn]` with `pub_date` and `[WEB]` tag |
| `_build_merged_block(chunks)` | Formats re-ranked chunks as `[M1]…[Mn]` |
| `_build_schema_block()` | Injects the JSON output schema the LLM must follow |
| `_build_question_block(query)` | Wraps the user question in its section header |

---

### Module 7 — LLMInterface (`src/generation/llm_interface.py`)

#### Position in the pipeline
```
PromptBuilder ──► LLMInterface ──► MedicalAnswer
                       │
               GPT-4o (primary)
               Llama-3-8B via Ollama (fallback)
```
The final generation step. Receives the assembled prompt from PromptBuilder,
calls the LLM, and returns a fully-parsed `MedicalAnswer`. Has no external
dependencies beyond the `openai` package.

#### Class: `LLMInterface`

**Constructor**
```python
LLMInterface(
    openai_api_key: Optional[str] = None,   # loaded from OPENAI_API_KEY env var
    openai_model: Optional[str] = None,     # loaded from OPENAI_MODEL (default: gpt-4o)
    ollama_base_url: Optional[str] = None,  # loaded from OLLAMA_BASE_URL
    ollama_model: Optional[str] = None,     # loaded from OLLAMA_MODEL (default: llama3:8b)
    max_tokens: int = 2000,
    temperature: float = 0.1,               # low for reproducible medical answers
)
```

**Key design decisions:**
- `openai_api_key=None` reads from env; `openai_api_key=""` explicitly disables OpenAI (used in tests — distinguishes "not provided" from "intentionally empty")
- Both clients (`_openai_client`, `_ollama_client`) are `None` at construction and initialised lazily on first use
- Ollama is accessed via `openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")` — no extra library needed

---

##### `call_llm(prompt, retrieval_result) → MedicalAnswer`

Send a prompt to the LLM and return a structured answer. Never raises.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `Union[str, List[dict]]` | required | Flat string from `build()` or messages list from `build_messages()` |
| `retrieval_result` | `Optional[RetrievalResult]` | `None` | Used for confidence and retrieval mode propagation |

**Returns:** `MedicalAnswer` — always a valid object, even on full provider failure

**Provider cascade:**
1. **OpenAI GPT-4o** (if `openai_api_key` is set) — called with `response_format={"type": "json_object"}` for guaranteed JSON output
2. **Ollama** (fallback) — same OpenAI client pointed at `localhost:11434/v1`; JSON mode attempted first, retried without it if unsupported
3. **Graceful fallback answer** — if both fail, returns a `MedicalAnswer` with `answer=_FALLBACK_ANSWER_TEXT` and `confidence=0.0`

**Example:**
```python
llm = LLMInterface()
messages = builder.build_messages(query, subgraph, chunks, snippets, merged)
answer = llm.call_llm(messages, retrieval_result=result)
# answer.answer     == "Metformin is the first-line treatment..."
# answer.confidence == 0.72   (from graph_subgraph.path_confidence)
# answer.citations  == [Citation(citation_id="c1", ...)]
# answer.disclaimer == "⚠️ This information is for educational purposes only..."
```

---

#### Output: `MedicalAnswer`

| Field | Source | Description |
|-------|--------|-------------|
| `answer` | LLM JSON `answer` field | Synthesised medical answer text |
| `citations` | LLM JSON `citations` array | Parsed `Citation` objects with id, title, url, source, year |
| `graph_path` | LLM JSON `graph_path` array | Ordered node names/CUIs traversed |
| `confidence` | `graph_subgraph.path_confidence` (preferred) or LLM JSON `confidence` | Clamped to [0.0, 1.0] |
| `retrieval_mode` | `retrieval_result.retrieval_mode` | graph / vector / hybrid / web |
| `disclaimer` | Model default | Medical safety disclaimer (always populated) |
| `raw_graph_subgraph` | `retrieval_result.graph_subgraph` | Attached for downstream visualisation |

---

#### Confidence rule

Per the project specification, confidence is the **product of Neo4j edge confidences** along the retrieved graph path (`graph_subgraph.path_confidence`). This overrides the LLM's self-reported confidence whenever a graph path exists. The LLM-reported value is used only as a fallback when no graph was traversed.

```
if graph_subgraph and path_confidence > 0:
    confidence = path_confidence          # authoritative (product of edge confs)
else:
    confidence = llm_json["confidence"]   # fallback (LLM self-assessment)

confidence = clamp(confidence, 0.0, 1.0)
```

---

#### JSON extraction

`_extract_json()` handles all real-world LLM output patterns without requiring JSON mode to be supported:

| Pattern | Example |
|---------|---------|
| Clean JSON | `{"answer": "Metformin..."}` |
| Fenced with language tag | ` ```json\n{...}\n``` ` |
| Fenced without tag | ` ```\n{...}\n``` ` |
| JSON in prose | `"Here is the result: {...} Hope this helps."` |

Extraction uses brace-depth tracking to find the outermost JSON object, making it robust against LLMs that add explanatory text around their JSON response.

---

#### Internal helpers

| Helper | Description |
|--------|-------------|
| `_to_messages(prompt)` | Normalises `str` or `List[dict]` to an OpenAI messages list |
| `_call_openai(messages)` | Calls GPT-4o with JSON mode; raises on any API error |
| `_call_ollama(messages)` | Calls Ollama; retries without JSON mode if unsupported |
| `_extract_json(text)` | Strips fences and extracts outermost JSON object via brace tracking |
| `_parse_response(raw, result)` | Parses JSON string into `MedicalAnswer`; fills missing fields with defaults |
| `_build_confidence(llm_conf, result)` | Applies graph path confidence rule; clamps to [0.0, 1.0] |
| `_ensure_openai_client()` | Lazy-init OpenAI client; returns `None` if key missing |
| `_ensure_ollama_client()` | Lazy-init Ollama client via OpenAI SDK; returns `None` if unavailable |
| `_fallback_answer(error_msg)` | Returns a safe placeholder `MedicalAnswer` when all providers fail |

## 6. Modules Pending

| Module | File | Key Responsibility |
|--------|------|--------------------|
| 8 — GraphVisualizer | `src/graph/visualizer.py` | pyvis HTML graph output |
| 9 — FastAPI App | `src/api/app.py` | REST API endpoints |
| 10 — Streamlit UI | `src/ui/streamlit_app.py` | Interactive web interface |

---

## 7. Data Flow Walkthrough

### Ingestion Flow (one-time / scheduled)

```
1. DataFetcher.fetch_all("Type 2 Diabetes")
        │
        ▼
2. [List[Document]] — 20 documents from 4 sources
        │
        ▼
3. NLPProcessor.process_document(doc) — for each document
        │
        ├── extract_entities()  →  [Metformin(Drug), Type2Diabetes(Disease), ...]
        ├── link_to_umls()      →  CUIs attached to each entity
        └── extract_relations() →  [Metformin -TREATS-> Type2Diabetes, ...]
        │
        ▼
4. GraphBuilder.upsert_batch(triples)
        │
        ├── add_node(Metformin)         →  Neo4j: (Drug {id: "C0025598"})
        ├── add_node(Type2Diabetes)     →  Neo4j: (Disease {id: "C0011860"})
        └── add_edge(TREATS, conf=0.6)  →  Neo4j: -[:TREATS {confidence: 0.6}]->
        │
        ▼
5. VectorIndexer.upsert_document(doc)
        │
        └── BioBERT embedding → Qdrant point {id: chunk_id, vector: [...]}
```

### Query Flow (per user request)

```
1. User: "What drugs treat Type 2 Diabetes?"
        │
        ▼
2. RetrievalOrchestrator
        │
        ├── QueryClassifier  →  mode: "hybrid"
        ├── GraphRetriever   →  Cypher: MATCH path from "Type 2 Diabetes" hops=2
        ├── VectorRetriever  →  Qdrant: top-5 BioBERT similar chunks
        ├── WebSearchNode    →  Tavily: fires if < 3 results or stale
        └── HybridMerger     →  Cohere re-rank → merged context
        │
        ▼
3. PromptBuilder.build(graph_triples, vector_chunks, web_snippets)
        │
        └── "System: You are a medical assistant. ⚠️ Disclaimer...
             Graph context: Metformin -TREATS-> Type2Diabetes (conf: 0.6)
             Supporting evidence: [chunk1, chunk2...]
             Web results: [snippet1 (2024-03-15)...]
             Question: What drugs treat Type 2 Diabetes?"
        │
        ▼
4. LLMInterface.call_llm(prompt)  →  GPT-4o
        │
        ▼
5. MedicalAnswer {
        answer: "First-line treatments include Metformin...",
        citations: [{title: "...", url: "pubmed...", year: 2023}],
        graph_path: ["Type2Diabetes", "Metformin"],
        confidence: 0.42,
        disclaimer: "⚠️ For educational purposes only..."
   }
```

---

## 8. Graph Schema

### Node Labels and Properties

| Label | id | name | source_url | last_updated | confidence_score |
|-------|----|------|------------|--------------|-----------------|
| `Disease` | UMLS CUI | e.g. "Type 2 Diabetes" | PubMed/UMLS URL | datetime | float |
| `Drug` | UMLS CUI | e.g. "Metformin" | FDA URL | datetime | float |
| `Gene` | UMLS CUI | e.g. "TCF7L2" | UMLS URL | datetime | float |
| `Symptom` | UMLS CUI | e.g. "Hyperglycemia" | — | datetime | float |
| `ClinicalTrial` | NCT ID | trial title | clinicaltrials.gov URL | datetime | float |
| `Paper` | PMID hash | paper title | PubMed URL | datetime | float |

### Relationship Types and Properties

| Type | Meaning | Example |
|------|---------|---------|
| `TREATS` | Drug treats disease | Metformin -TREATS-> Type2Diabetes |
| `CAUSES` | Entity causes condition | Obesity -CAUSES-> Type2Diabetes |
| `INTERACTS_WITH` | Drug–drug interaction | Metformin -INTERACTS_WITH-> Contrast_Dye |
| `ASSOCIATED_WITH` | Statistical association | TCF7L2 -ASSOCIATED_WITH-> Type2Diabetes |
| `INVESTIGATED_IN` | Entity studied in trial | Metformin -INVESTIGATED_IN-> NCT01234567 |
| `CITED_BY` | Paper citation | Paper_A -CITED_BY-> Paper_B |

All relationships carry: `confidence (float)`, `source_doc_id (str)`, `year (int)`, `relation_type (str)`

### Confidence Propagation Rule

```
Multi-hop answer confidence = ∏ edge.confidence along the path

Example (2-hop):
  Type2Diabetes <-[TREATS, 0.8]- Metformin -[INTERACTS_WITH, 0.5]-> ContrastDye
  path_confidence = 0.8 × 0.5 = 0.40
```

---

## 9. Setup & Installation

### Prerequisites

- Python 3.11 (required — scispaCy does not support 3.12+)
- Docker Desktop
- Git

### Step 1 — Clone and set up environment

```powershell
git clone <repo-url>
cd medical-kg-assistant

py -3.11 -m venv venv
venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt --only-binary=:all:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Step 2 — Install the scispaCy biomedical model

```powershell
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
```

### Step 3 — Configure environment variables

```powershell
copy .env.example .env
# Edit .env and fill in your API keys
```

Minimum required keys:

| Key | Where to get it |
|-----|----------------|
| `PUBMED_EMAIL` | Any valid email address |
| `OPENAI_API_KEY` | https://platform.openai.com/api-keys |
| `TAVILY_API_KEY` | https://tavily.com |

Optional:

| Key | Purpose |
|-----|---------|
| `UMLS_API_KEY` | https://uts.nlm.nih.gov/uts/signup-login |
| `COHERE_API_KEY` | https://dashboard.cohere.com |
| `OPENFDA_API_KEY` | Increases OpenFDA rate limit |

### Step 4 — Start infrastructure

```powershell
docker compose up -d
docker ps
# Should show med_kg_neo4j and med_kg_qdrant
```

Verify Neo4j at: http://localhost:7474 (user: `neo4j`, password: `medkg_password`)
Verify Qdrant at: http://localhost:6333/dashboard

---

## 10. Running the Tests

```powershell
# All unit tests (no infrastructure needed)
python -m pytest src/tests/ -v

# With coverage report
python -m pytest src/tests/ -v --cov=src --cov-report=term-missing

# Integration tests (requires docker compose up -d)
$env:RUN_INTEGRATION_TESTS = "1"
python -m pytest src/tests/ -v

# Single module tests
python -m pytest src/tests/test_data_nlp.py -v
python -m pytest src/tests/test_graph_builder.py -v
python -m pytest src/tests/test_vector_indexer.py -v
python -m pytest src/tests/test_orchestrator.py -v
python -m pytest src/tests/test_prompt_builder.py -v
python -m pytest src/tests/test_llm_interface.py -v
```

### Test inventory

| Test file | Tests | What is covered |
|-----------|-------|----------------|
| `test_data_nlp.py` | 25 unit + 3 integration | DataFetcher (all 4 sources), NLPProcessor (entities, relations, UMLS linking) |
| `test_graph_builder.py` | 24 unit + 1 integration | GraphBuilder (connection, node/edge CRUD, batch upsert, subgraph retrieval, confidence calculation) |
| `test_vector_indexer.py` | 38 unit + 1 integration | VectorIndexer (chunking, embedding, collection management, upsert, similarity search, deletion, stats) |
| `test_orchestrator.py` | 47 unit + 1 integration | RetrievalOrchestrator (classifier, graph/vector/web nodes, merger, routing, staleness, end-to-end run) |
| `test_prompt_builder.py` | 52 unit | PromptBuilder (all 7 sections, build/build_messages, trimming, zero-score display, edge cases) |
| `test_llm_interface.py` | 68 unit | LLMInterface (provider cascade, JSON extraction, confidence rule, citation parsing, fallback answer, lazy client init) |

---

## 11. Environment Variables

Full reference of all variables read from `.env`:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PUBMED_EMAIL` | Yes | `user@example.com` | NCBI requires an email for E-utilities |
| `UMLS_API_KEY` | No | — | Enables UMLS concept fetching and entity linking |
| `OPENFDA_API_KEY` | No | — | Increases OpenFDA rate limit from 240 to 1000 req/min |
| `NEO4J_URI` | Yes | `bolt://localhost:7687` | Neo4j Bolt connection URI |
| `NEO4J_USER` | Yes | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | Yes | `medkg_password` | Neo4j password |
| `QDRANT_HOST` | Yes | `localhost` | Qdrant host |
| `QDRANT_PORT` | Yes | `6333` | Qdrant HTTP port |
| `QDRANT_API_KEY` | No | — | Only needed if Qdrant auth is enabled |
| `QDRANT_COLLECTION` | Yes | `med_kg_chunks` | Qdrant collection name |
| `BIOBERT_MODEL` | No | `dmis-lab/biobert-base-cased-v1.2` | HuggingFace model used for embeddings |
| `OPENAI_API_KEY` | Yes* | — | GPT-4o API key (*required unless using Ollama) |
| `OPENAI_MODEL` | No | `gpt-4o` | OpenAI model name |
| `OLLAMA_BASE_URL` | No | `http://localhost:11434` | Ollama local LLM base URL |
| `OLLAMA_MODEL` | No | `llama3:8b` | Ollama model name |
| `TAVILY_API_KEY` | Yes | — | Tavily web search API key |
| `COHERE_API_KEY` | No | — | Cohere Rerank API key |
| `LOG_LEVEL` | No | `INFO` | Python logging level |
| `FASTAPI_HOST` | No | `0.0.0.0` | FastAPI bind host |
| `FASTAPI_PORT` | No | `8000` | FastAPI port |
| `STREAMLIT_PORT` | No | `8501` | Streamlit port |
| `WEB_SEARCH_STALENESS_DAYS` | No | `180` | Days before a node is considered stale |

---

## 12. Docker Infrastructure

### Services

#### Neo4j 5.26.0
- **Browser UI:** http://localhost:7474
- **Bolt port:** `7687`
- **Default credentials:** `neo4j` / `medkg_password`
- **Plugins:** APOC (advanced Cypher procedures)
- **Memory:** 512MB heap initial, 2GB heap max, 512MB page cache

#### Qdrant 1.9.4
- **REST API:** http://localhost:6333
- **gRPC:** port `6334`
- **Dashboard:** http://localhost:6333/dashboard

### Commands

```powershell
# Start all services
docker compose up -d

# Start individual service
docker compose up -d neo4j
docker compose up -d qdrant

# View logs
docker logs med_kg_neo4j
docker logs med_kg_qdrant

# Stop services (preserves data volumes)
docker compose down

# Full reset (destroys all data)
docker compose down -v
```