"""
VectorIndexer — Module 4 of the Medical Knowledge Assistant pipeline.

Responsibilities
----------------
* embed_chunk(text)                    → List[float]  (BioBERT vector)
* upsert_to_qdrant(chunk)              → bool
* upsert_batch(chunks)                 → Tuple[int, int]
* similarity_search(query, top_k)      → List[Chunk]
* chunk_document(doc)                  → List[Chunk]
* upsert_document(doc)                 → int          (chunks written)
* delete_by_doc_id(doc_id)             → int          (chunks deleted)
* get_collection_stats()               → Dict[str, Any]

Embedding model
---------------
BioBERT — dmis-lab/biobert-base-cased-v1.2 (HuggingFace)
Embedding dimension: 768

Each Qdrant point stores:
  id      : UUID derived from chunk_id (deterministic)
  vector  : 768-dimensional BioBERT embedding
  payload : {
      chunk_id, text, doc_id, node_cui,
      source, source_url, pub_date
  }

Environment variables (loaded from .env)
-----------------------------------------
QDRANT_HOST         localhost
QDRANT_PORT         6333
QDRANT_API_KEY      (optional — only if Qdrant auth is enabled)
QDRANT_COLLECTION   med_kg_chunks
BIOBERT_MODEL       dmis-lab/biobert-base-cased-v1.2  (override if needed)
"""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from src.utils.models import Chunk, Document

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — graceful degradation when libraries are absent
# ---------------------------------------------------------------------------

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "transformers / torch not installed — "
        "VectorIndexer will use a zero-vector fallback for embeddings. "
        "Install with: pip install transformers torch"
    )

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
    from qdrant_client.http.exceptions import UnexpectedResponse
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning(
        "qdrant-client not installed — "
        "VectorIndexer will operate in dry-run mode. "
        "Install with: pip install qdrant-client"
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BIOBERT_DIM = 768
_DEFAULT_MODEL = "dmis-lab/biobert-base-cased-v1.2"
_MAX_TOKEN_LENGTH = 512       # BioBERT hard limit
_CHUNK_SIZE = 400             # characters per chunk
_CHUNK_OVERLAP = 80           # overlap between consecutive chunks
_BATCH_EMBED_SIZE = 16        # documents per embedding batch


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _chunk_id_to_uuid(chunk_id: str) -> str:
    """Convert a deterministic string chunk_id to a UUID string for Qdrant."""
    return str(uuid.UUID(hashlib.md5(chunk_id.encode()).hexdigest()))


def _text_to_chunks(
    text: str,
    chunk_size: int = _CHUNK_SIZE,
    overlap: int = _CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping character-level chunks.

    Parameters
    ----------
    text:
        Source text to split.
    chunk_size:
        Maximum characters per chunk.
    overlap:
        Character overlap between adjacent chunks.

    Returns
    -------
    List[str]
        Non-empty text chunks.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap  # back up by overlap for context continuity

    return chunks


# ---------------------------------------------------------------------------
# VectorIndexer
# ---------------------------------------------------------------------------


class VectorIndexer:
    """
    Manages BioBERT embedding generation and Qdrant vector store operations.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier for the embedding model.
        Defaults to ``dmis-lab/biobert-base-cased-v1.2``.
    qdrant_host:
        Hostname of the Qdrant instance.
    qdrant_port:
        HTTP port of the Qdrant instance.
    qdrant_api_key:
        Optional API key for authenticated Qdrant deployments.
    collection_name:
        Qdrant collection to store embeddings in.
    device:
        PyTorch device — ``"cuda"`` or ``"cpu"``. Auto-detected if None.

    Example
    -------
    >>> indexer = VectorIndexer()
    >>> indexer.ensure_collection()
    >>> chunks_written = indexer.upsert_document(doc)
    >>> results = indexer.similarity_search("metformin diabetes treatment", top_k=5)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        qdrant_host: Optional[str] = None,
        qdrant_port: Optional[int] = None,
        qdrant_api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self._model_name = (
            model_name
            or os.getenv("BIOBERT_MODEL", _DEFAULT_MODEL)
        )
        self._qdrant_host = qdrant_host or os.getenv("QDRANT_HOST", "localhost")
        self._qdrant_port = qdrant_port or int(os.getenv("QDRANT_PORT", "6333"))
        self._qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY", "") or None
        self.collection_name = (
            collection_name or os.getenv("QDRANT_COLLECTION", "med_kg_chunks")
        )

        # Device selection
        if device:
            self._device = device
        elif TRANSFORMERS_AVAILABLE:
            import torch  # noqa: PLC0415
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = "cpu"

        # Lazy-loaded model components
        self._tokenizer: Optional[Any] = None
        self._model: Optional[Any] = None
        self._qdrant: Optional[Any] = None

        # Connect to Qdrant eagerly
        self._connect_qdrant()

        logger.info(
            "VectorIndexer initialised (model=%s, device=%s, qdrant=%s:%d, collection=%s)",
            self._model_name,
            self._device,
            self._qdrant_host,
            self._qdrant_port,
            self.collection_name,
        )

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _connect_qdrant(self) -> None:
        """Establish connection to Qdrant."""
        if not QDRANT_AVAILABLE:
            logger.warning("qdrant-client not available — running in dry-run mode")
            return
        try:
            self._qdrant = QdrantClient(
                host=self._qdrant_host,
                port=self._qdrant_port,
                api_key=self._qdrant_api_key,
                timeout=30,
                check_compatibility=False,
            )
            # Lightweight connectivity check
            self._qdrant.get_collections()
            logger.info(
                "Connected to Qdrant at %s:%d", self._qdrant_host, self._qdrant_port
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Qdrant not reachable at %s:%d — is the container running? "
                "Run: docker compose up -d qdrant  |  Error: %s",
                self._qdrant_host,
                self._qdrant_port,
                exc,
            )
            self._qdrant = None

    def is_connected(self) -> bool:
        """Return True if Qdrant is reachable."""
        if not self._qdrant:
            return False
        try:
            self._qdrant.get_collections()
            return True
        except Exception:  # noqa: BLE001
            return False

    def _load_model(self) -> None:
        """Lazy-load the BioBERT tokenizer and model on first use."""
        if self._tokenizer is not None and self._model is not None:
            return  # already loaded

        if not TRANSFORMERS_AVAILABLE:
            logger.warning(
                "Cannot load BioBERT — transformers not installed. "
                "Embeddings will be zero vectors."
            )
            return

        logger.info("Loading BioBERT model: %s (this may take a moment…)", self._model_name)
        try:
            import torch  # noqa: PLC0415
            from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModel.from_pretrained(self._model_name)
            self._model.to(self._device)
            self._model.eval()
            logger.info("BioBERT model loaded on device=%s", self._device)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load BioBERT model: %s", exc)
            self._tokenizer = None
            self._model = None

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def ensure_collection(self) -> bool:
        """
        Create the Qdrant collection if it does not already exist.

        Uses cosine distance for semantic similarity, matching BioBERT's
        representation space.

        Returns
        -------
        bool
            True if the collection exists or was created successfully.
        """
        if not self._qdrant:
            logger.warning("ensure_collection skipped — not connected to Qdrant")
            return False

        try:
            existing = [c.name for c in self._qdrant.get_collections().collections]
            if self.collection_name in existing:
                logger.info("Qdrant collection '%s' already exists", self.collection_name)
                return True

            self._qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=_BIOBERT_DIM,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            logger.info(
                "Created Qdrant collection '%s' (dim=%d, metric=cosine)",
                self.collection_name,
                _BIOBERT_DIM,
            )
            return True

        except Exception as exc:  # noqa: BLE001
            logger.error("ensure_collection failed: %s", exc)
            return False

    def delete_collection(self) -> bool:
        """
        Delete the Qdrant collection and all its vectors.

        Use with caution — this deletes all stored embeddings.

        Returns
        -------
        bool
            True if deleted successfully.
        """
        if not self._qdrant:
            return False
        try:
            self._qdrant.delete_collection(self.collection_name)
            logger.info("Deleted Qdrant collection '%s'", self.collection_name)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("delete_collection failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed_chunk(self, text: str) -> List[float]:
        """
        Generate a BioBERT embedding for a single text string.

        Uses mean-pooling over the final hidden states of all non-padding
        tokens, which produces a stable fixed-length sentence representation.

        Parameters
        ----------
        text:
            The text to embed. Truncated to 512 tokens (BioBERT limit).

        Returns
        -------
        List[float]
            768-dimensional embedding vector. Returns a zero vector if the
            model is unavailable.
        """
        self._load_model()

        if self._model is None or self._tokenizer is None:
            logger.debug("Returning zero vector — BioBERT not loaded")
            return [0.0] * _BIOBERT_DIM

        try:
            import torch  # noqa: PLC0415

            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=_MAX_TOKEN_LENGTH,
                padding=True,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

            # Mean-pool over non-padding tokens
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            mask_expanded = (
                attention_mask.unsqueeze(-1)
                .expand(token_embeddings.size())
                .float()
            )
            sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask

            return mean_pooled.squeeze().cpu().tolist()

        except Exception as exc:  # noqa: BLE001
            logger.error("embed_chunk failed: %s", exc)
            return [0.0] * _BIOBERT_DIM

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate BioBERT embeddings for a list of texts efficiently.

        Processes texts in batches of ``_BATCH_EMBED_SIZE`` to avoid
        out-of-memory errors on large corpora.

        Parameters
        ----------
        texts:
            List of texts to embed.

        Returns
        -------
        List[List[float]]
            One 768-dimensional vector per input text.
        """
        if not texts:
            return []

        self._load_model()

        if self._model is None or self._tokenizer is None:
            return [[0.0] * _BIOBERT_DIM for _ in texts]

        all_vectors: List[List[float]] = []

        try:
            import torch  # noqa: PLC0415

            for batch_start in range(0, len(texts), _BATCH_EMBED_SIZE):
                batch = texts[batch_start: batch_start + _BATCH_EMBED_SIZE]

                inputs = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    max_length=_MAX_TOKEN_LENGTH,
                    padding=True,
                )
                inputs = {k: v.to(self._device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self._model(**inputs)

                attention_mask = inputs["attention_mask"]
                token_embeddings = outputs.last_hidden_state
                mask_expanded = (
                    attention_mask.unsqueeze(-1)
                    .expand(token_embeddings.size())
                    .float()
                )
                sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                mean_pooled = sum_embeddings / sum_mask

                batch_vectors = mean_pooled.cpu().tolist()
                all_vectors.extend(batch_vectors)
                logger.debug(
                    "Embedded batch %d-%d / %d",
                    batch_start,
                    batch_start + len(batch),
                    len(texts),
                )

        except Exception as exc:  # noqa: BLE001
            logger.error("embed_batch failed at batch starting %d: %s", batch_start, exc)
            # Pad remaining with zero vectors
            while len(all_vectors) < len(texts):
                all_vectors.append([0.0] * _BIOBERT_DIM)

        return all_vectors

    # ------------------------------------------------------------------
    # Document chunking
    # ------------------------------------------------------------------

    def chunk_document(self, doc: Document) -> List[Chunk]:
        """
        Split a Document into overlapping text chunks ready for indexing.

        Concatenates title and abstract, then splits into fixed-size
        character chunks with overlap for context continuity.

        Parameters
        ----------
        doc:
            A Document from DataFetcher.

        Returns
        -------
        List[Chunk]
            One Chunk per text segment. Each chunk has a deterministic
            ``chunk_id`` derived from the doc_id and chunk index.
        """
        full_text = f"{doc.title}. {doc.abstract}".strip()
        raw_chunks = _text_to_chunks(full_text)

        pub_date: Optional[datetime] = doc.publication_date

        chunks: List[Chunk] = []
        for idx, text in enumerate(raw_chunks):
            chunk_id = f"{doc.doc_id}_chunk_{idx:04d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=text,
                    doc_id=doc.doc_id,
                    node_cui=None,        # enriched later by NLPProcessor
                    score=0.0,
                    source=doc.source,
                    source_url=doc.source_url,
                    pub_date=pub_date,
                )
            )

        logger.debug(
            "chunk_document: doc_id=%s → %d chunks", doc.doc_id, len(chunks)
        )
        return chunks

    # ------------------------------------------------------------------
    # Qdrant upsert
    # ------------------------------------------------------------------

    def upsert_to_qdrant(self, chunk: Chunk, vector: Optional[List[float]] = None) -> bool:
        """
        Upsert a single Chunk (with its embedding) into Qdrant.

        If *vector* is not provided, ``embed_chunk()`` is called internally.

        Parameters
        ----------
        chunk:
            The Chunk to store.
        vector:
            Pre-computed embedding vector. If None, one is generated.

        Returns
        -------
        bool
            True if the upsert succeeded.
        """
        if not self._qdrant:
            logger.warning("upsert_to_qdrant skipped — not connected to Qdrant")
            return False

        if vector is None:
            vector = self.embed_chunk(chunk.text)

        point_id = _chunk_id_to_uuid(chunk.chunk_id)

        payload: Dict[str, Any] = {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "doc_id": chunk.doc_id,
            "node_cui": chunk.node_cui,
            "source": chunk.source,
            "source_url": chunk.source_url,
            "pub_date": chunk.pub_date.isoformat() if chunk.pub_date else None,
        }

        try:
            self._qdrant.upsert(
                collection_name=self.collection_name,
                points=[
                    qmodels.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                ],
            )
            logger.debug("Upserted chunk %s → Qdrant point %s", chunk.chunk_id, point_id)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("upsert_to_qdrant failed for chunk %s: %s", chunk.chunk_id, exc)
            return False

    def upsert_batch(
        self, chunks: List[Chunk]
    ) -> Tuple[int, int]:
        """
        Batch upsert chunks into Qdrant with pre-computed embeddings.

        Generates all embeddings in one batched forward pass for efficiency,
        then uploads all points in a single Qdrant batch call.

        Parameters
        ----------
        chunks:
            List of Chunk objects to index.

        Returns
        -------
        Tuple[int, int]
            ``(attempted, succeeded)`` counts.
        """
        if not chunks:
            return 0, 0

        if not self._qdrant:
            logger.warning("upsert_batch skipped — not connected to Qdrant")
            return len(chunks), 0

        # Generate all embeddings in one batched call
        texts = [c.text for c in chunks]
        vectors = self.embed_batch(texts)

        points: List[Any] = []
        for chunk, vector in zip(chunks, vectors):
            point_id = _chunk_id_to_uuid(chunk.chunk_id)
            payload: Dict[str, Any] = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "doc_id": chunk.doc_id,
                "node_cui": chunk.node_cui,
                "source": chunk.source,
                "source_url": chunk.source_url,
                "pub_date": chunk.pub_date.isoformat() if chunk.pub_date else None,
            }
            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        try:
            self._qdrant.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            logger.info(
                "upsert_batch: %d chunks written to collection '%s'",
                len(points),
                self.collection_name,
            )
            return len(chunks), len(points)
        except Exception as exc:  # noqa: BLE001
            logger.error("upsert_batch failed: %s", exc)
            return len(chunks), 0

    def upsert_document(self, doc: Document) -> int:
        """
        Full pipeline: chunk a Document, embed chunks, upsert to Qdrant.

        This is the primary ingestion entry point for a single document.

        Parameters
        ----------
        doc:
            A Document from DataFetcher.

        Returns
        -------
        int
            Number of chunks successfully written to Qdrant.
        """
        chunks = self.chunk_document(doc)
        if not chunks:
            logger.warning("upsert_document: no chunks generated for doc %s", doc.doc_id)
            return 0

        _, succeeded = self.upsert_batch(chunks)
        logger.info(
            "upsert_document: doc_id=%s → %d/%d chunks indexed",
            doc.doc_id,
            succeeded,
            len(chunks),
        )
        return succeeded

    # ------------------------------------------------------------------
    # Similarity search
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filter_source: Optional[str] = None,
    ) -> List[Chunk]:
        """
        Search for the most semantically similar chunks to *query*.

        Embeds the query with BioBERT and performs approximate nearest
        neighbour search in Qdrant.

        Parameters
        ----------
        query:
            The natural-language query to search for.
        top_k:
            Maximum number of results to return.
        score_threshold:
            Minimum cosine similarity score (0–1). Results below this
            threshold are filtered out.
        filter_source:
            Optional source filter (e.g. ``"pubmed"``). Only returns
            chunks from this source when set.

        Returns
        -------
        List[Chunk]
            Top-k chunks ordered by descending similarity score.
        """
        if not self._qdrant:
            logger.warning("similarity_search skipped — not connected to Qdrant")
            return []

        query_vector = self.embed_chunk(query)

        # Build optional payload filter
        qdrant_filter: Optional[Any] = None
        if filter_source and QDRANT_AVAILABLE:
            qdrant_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="source",
                        match=qmodels.MatchValue(value=filter_source),
                    )
                ]
            )

        try:
            response = self._qdrant.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                score_threshold=score_threshold if score_threshold > 0 else None,
                query_filter=qdrant_filter,
                with_payload=True,
            )

            chunks: List[Chunk] = []
            for hit in response.points:
                payload = hit.payload or {}
                pub_date: Optional[datetime] = None
                if payload.get("pub_date"):
                    try:
                        pub_date = datetime.fromisoformat(payload["pub_date"])
                    except (ValueError, TypeError):
                        pass

                chunks.append(
                    Chunk(
                        chunk_id=payload.get("chunk_id", str(hit.id)),
                        text=payload.get("text", ""),
                        doc_id=payload.get("doc_id", ""),
                        node_cui=payload.get("node_cui"),
                        score=hit.score,
                        source=payload.get("source", "vector"),
                        source_url=payload.get("source_url"),
                        pub_date=pub_date,
                    )
                )

            logger.info(
                "similarity_search: query=%r → %d results (top score=%.3f)",
                query[:50],
                len(chunks),
                chunks[0].score if chunks else 0.0,
            )
            return chunks

        except Exception as exc:  # noqa: BLE001
            logger.error("similarity_search failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def delete_by_doc_id(self, doc_id: str) -> int:
        """
        Delete all Qdrant points associated with a document ID.

        Used when re-ingesting an updated document to avoid stale chunks.

        Parameters
        ----------
        doc_id:
            The document ID whose chunks should be removed.

        Returns
        -------
        int
            Number of points deleted (0 if none found or error).
        """
        if not self._qdrant:
            return 0

        try:
            # Scroll to find all point IDs for this doc_id
            scroll_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="doc_id",
                        match=qmodels.MatchValue(value=doc_id),
                    )
                ]
            )

            point_ids: List[Any] = []
            offset = None
            while True:
                results, offset = self._qdrant.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=scroll_filter,
                    limit=100,
                    offset=offset,
                    with_payload=False,
                    with_vectors=False,
                )
                point_ids.extend([p.id for p in results])
                if offset is None:
                    break

            if not point_ids:
                logger.debug("delete_by_doc_id: no points found for doc_id=%s", doc_id)
                return 0

            self._qdrant.delete(
                collection_name=self.collection_name,
                points_selector=qmodels.PointIdsList(points=point_ids),
            )
            logger.info(
                "delete_by_doc_id: deleted %d points for doc_id=%s",
                len(point_ids),
                doc_id,
            )
            return len(point_ids)

        except Exception as exc:  # noqa: BLE001
            logger.error("delete_by_doc_id failed for doc_id=%s: %s", doc_id, exc)
            return 0

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Return basic statistics about the Qdrant collection.

        Returns
        -------
        Dict[str, Any]
            Dictionary with keys: ``name``, ``points_count``,
            ``vectors_count`` (alias), ``indexed_vectors_count``, ``status``.

        Notes
        -----
        qdrant-client >= 1.7 renamed ``vectors_count`` to ``points_count``.
        Both keys are returned for backward compatibility.
        """
        if not self._qdrant:
            return {"error": "not connected"}

        try:
            info = self._qdrant.get_collection(self.collection_name)

            # qdrant-client >= 1.7 uses points_count; older used vectors_count
            points_count = (
                getattr(info, "points_count", None)
                or getattr(info, "vectors_count", 0)
                or 0
            )
            indexed_count = (
                getattr(info, "indexed_vectors_count", None)
                or points_count
            )

            return {
                "name": self.collection_name,
                "points_count": points_count,
                "vectors_count": points_count,   # kept for backward compat
                "indexed_vectors_count": indexed_count,
                "status": str(info.status),
                "dimension": _BIOBERT_DIM,
                "distance": "cosine",
            }
        except Exception as exc:  # noqa: BLE001
            logger.error("get_collection_stats failed: %s", exc)
            return {"error": str(exc)}