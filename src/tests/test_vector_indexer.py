"""
Tests for Module 4 — VectorIndexer.

Unit tests mock both Qdrant and the BioBERT model so the suite runs
without any infrastructure or large model downloads.

Integration tests (marked @pytest.mark.integration) require:
  - docker compose up -d qdrant
  - RUN_INTEGRATION_TESTS=1 set in environment
  - BioBERT downloaded (or BIOBERT_MODEL set to a small test model)
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, List
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.utils.models import Chunk, Document


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_document() -> Document:
    return Document(
        doc_id="test_doc_vec_001",
        title="Metformin Treatment for Type 2 Diabetes",
        abstract=(
            "Metformin is a biguanide drug used as first-line treatment for "
            "Type 2 Diabetes mellitus. It reduces hepatic glucose production "
            "and improves insulin sensitivity. Common side effects include "
            "gastrointestinal symptoms. Metformin is associated with reduced "
            "cardiovascular risk in diabetic patients."
        ),
        source="pubmed",
        source_url="https://pubmed.ncbi.nlm.nih.gov/12345/",
        publication_date=datetime(2023, 3, 15),
    )


@pytest.fixture()
def sample_chunk() -> Chunk:
    return Chunk(
        chunk_id="test_doc_vec_001_chunk_0000",
        text="Metformin is a biguanide drug used as first-line treatment for Type 2 Diabetes mellitus.",
        doc_id="test_doc_vec_001",
        node_cui="C0025598",
        score=0.0,
        source="pubmed",
        source_url="https://pubmed.ncbi.nlm.nih.gov/12345/",
        pub_date=datetime(2023, 3, 15),
    )


@pytest.fixture()
def mock_qdrant_client():
    """A MagicMock standing in for QdrantClient."""
    client = MagicMock()
    client.get_collections.return_value = MagicMock(collections=[])
    client.get_collection.return_value = MagicMock(
        vectors_count=10,
        indexed_vectors_count=10,
        status="green",
    )
    return client


@pytest.fixture()
def indexer(mock_qdrant_client):
    """VectorIndexer with mocked Qdrant and BioBERT disabled."""
    from src.vector.indexer import VectorIndexer

    with patch("src.vector.indexer.QdrantClient", return_value=mock_qdrant_client):
        vi = VectorIndexer(
            qdrant_host="localhost",
            qdrant_port=6333,
            collection_name="test_collection",
        )
    # Force BioBERT off so embed_chunk returns zero vectors predictably
    vi._tokenizer = None
    vi._model = None
    return vi


# ---------------------------------------------------------------------------
# Text chunking tests (pure Python — no mocks needed)
# ---------------------------------------------------------------------------


class TestTextChunking:

    def test_short_text_returns_single_chunk(self) -> None:
        """Text shorter than chunk_size returns as one chunk."""
        from src.vector.indexer import _text_to_chunks

        result = _text_to_chunks("Short text.", chunk_size=400)
        assert len(result) == 1
        assert result[0] == "Short text."

    def test_empty_text_returns_empty_list(self) -> None:
        """Empty or whitespace-only text returns []."""
        from src.vector.indexer import _text_to_chunks

        assert _text_to_chunks("") == []
        assert _text_to_chunks("   ") == []

    def test_long_text_splits_into_multiple_chunks(self) -> None:
        """Text longer than chunk_size is split into multiple chunks."""
        from src.vector.indexer import _text_to_chunks

        long_text = "word " * 200          # 1000 characters
        chunks = _text_to_chunks(long_text, chunk_size=400, overlap=80)
        assert len(chunks) > 1

    def test_chunks_have_overlap(self) -> None:
        """Consecutive chunks share overlapping content."""
        from src.vector.indexer import _text_to_chunks

        text = "A" * 500
        chunks = _text_to_chunks(text, chunk_size=300, overlap=100)
        assert len(chunks) >= 2
        # Last 100 chars of chunk 0 should appear at start of chunk 1
        assert chunks[0][-50:] in chunks[1]

    def test_all_text_covered(self) -> None:
        """All characters from the source text appear in some chunk."""
        from src.vector.indexer import _text_to_chunks

        text = "The quick brown fox " * 30   # 600 chars
        chunks = _text_to_chunks(text, chunk_size=200, overlap=40)
        combined = " ".join(chunks)
        # Every word from original should appear somewhere
        for word in ["quick", "brown", "fox"]:
            assert word in combined

    def test_chunk_id_to_uuid_deterministic(self) -> None:
        """Same chunk_id always produces the same UUID."""
        from src.vector.indexer import _chunk_id_to_uuid

        uid1 = _chunk_id_to_uuid("doc_001_chunk_0000")
        uid2 = _chunk_id_to_uuid("doc_001_chunk_0000")
        assert uid1 == uid2

    def test_chunk_id_to_uuid_different_ids(self) -> None:
        """Different chunk_ids produce different UUIDs."""
        from src.vector.indexer import _chunk_id_to_uuid

        assert _chunk_id_to_uuid("doc_001_chunk_0000") != _chunk_id_to_uuid("doc_001_chunk_0001")


# ---------------------------------------------------------------------------
# VectorIndexer initialisation tests
# ---------------------------------------------------------------------------


class TestVectorIndexerInit:

    def test_import_and_instantiation(self, indexer) -> None:
        """VectorIndexer can be imported and instantiated."""
        assert indexer is not None
        assert indexer.collection_name == "test_collection"

    def test_is_connected_true(self, indexer) -> None:
        """is_connected returns True when Qdrant mock is healthy."""
        assert indexer.is_connected() is True

    def test_is_connected_false_without_client(self) -> None:
        """is_connected returns False when _qdrant is None."""
        from src.vector.indexer import VectorIndexer

        with patch("src.vector.indexer.QdrantClient", side_effect=Exception("conn fail")):
            vi = VectorIndexer()
        assert vi.is_connected() is False


# ---------------------------------------------------------------------------
# Collection management tests
# ---------------------------------------------------------------------------


class TestCollectionManagement:

    def test_ensure_collection_creates_when_absent(
        self, indexer, mock_qdrant_client
    ) -> None:
        """ensure_collection calls create_collection when collection is missing."""
        mock_qdrant_client.get_collections.return_value = MagicMock(collections=[])

        result = indexer.ensure_collection()

        assert result is True
        mock_qdrant_client.create_collection.assert_called_once()
        call_kwargs = mock_qdrant_client.create_collection.call_args[1]
        assert call_kwargs["collection_name"] == "test_collection"

    def test_ensure_collection_skips_when_exists(
        self, indexer, mock_qdrant_client
    ) -> None:
        """ensure_collection does not recreate an existing collection."""
        existing = MagicMock()
        existing.name = "test_collection"
        mock_qdrant_client.get_collections.return_value = MagicMock(
            collections=[existing]
        )

        result = indexer.ensure_collection()

        assert result is True
        mock_qdrant_client.create_collection.assert_not_called()

    def test_ensure_collection_returns_false_when_disconnected(self) -> None:
        """ensure_collection returns False when Qdrant is not connected."""
        from src.vector.indexer import VectorIndexer

        with patch("src.vector.indexer.QdrantClient", side_effect=Exception):
            vi = VectorIndexer()

        result = vi.ensure_collection()
        assert result is False

    def test_delete_collection(self, indexer, mock_qdrant_client) -> None:
        """delete_collection calls Qdrant delete_collection."""
        result = indexer.delete_collection()
        assert result is True
        mock_qdrant_client.delete_collection.assert_called_once_with("test_collection")


# ---------------------------------------------------------------------------
# Embedding tests
# ---------------------------------------------------------------------------


class TestEmbedding:

    def test_embed_chunk_returns_zero_vector_without_model(
        self, indexer
    ) -> None:
        """embed_chunk returns a 768-dimensional zero vector when BioBERT is off.

        Patches _load_model as a no-op so that transformers (if installed)
        does not reload the model during this test.
        """
        with patch.object(indexer, "_load_model"):   # prevent reload
            indexer._tokenizer = None
            indexer._model = None
            vector = indexer.embed_chunk("Metformin treats diabetes.")
        assert len(vector) == 768
        assert all(v == 0.0 for v in vector)

    def test_embed_chunk_empty_text(self, indexer) -> None:
        """embed_chunk handles empty text gracefully."""
        vector = indexer.embed_chunk("")
        assert len(vector) == 768

    def test_embed_batch_returns_correct_count(self, indexer) -> None:
        """embed_batch returns one vector per input text."""
        texts = ["text one", "text two", "text three"]
        vectors = indexer.embed_batch(texts)
        assert len(vectors) == 3
        for v in vectors:
            assert len(v) == 768

    def test_embed_batch_empty_input(self, indexer) -> None:
        """embed_batch returns [] for empty input."""
        assert indexer.embed_batch([]) == []

    def test_embed_chunk_with_mock_model(self, indexer) -> None:
        """embed_chunk uses the model when loaded."""
        import torch

        # Simulate a loaded model returning a specific tensor
        mock_output = MagicMock()
        mock_tensor = torch.ones(1, 5, 768)  # batch=1, seq=5, dim=768
        mock_output.last_hidden_state = mock_tensor

        mock_model = MagicMock()
        mock_model.return_value = mock_output

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.ones(1, 5, dtype=torch.long),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
        }

        indexer._model = mock_model
        indexer._tokenizer = mock_tokenizer

        vector = indexer.embed_chunk("test text")
        assert len(vector) == 768
        # With all-ones mask and all-ones hidden states, result should be ~1.0
        assert all(abs(v - 1.0) < 1e-3 for v in vector)


# ---------------------------------------------------------------------------
# Document chunking tests
# ---------------------------------------------------------------------------


class TestChunkDocument:

    def test_chunk_document_returns_chunks(
        self, indexer, sample_document: Document
    ) -> None:
        """chunk_document returns at least one Chunk for a non-empty document."""
        chunks = indexer.chunk_document(sample_document)
        assert len(chunks) >= 1

    def test_chunk_ids_are_unique(
        self, indexer, sample_document: Document
    ) -> None:
        """All chunk_ids within a document are unique."""
        chunks = indexer.chunk_document(sample_document)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_metadata_propagated(
        self, indexer, sample_document: Document
    ) -> None:
        """Chunks carry doc_id, source, source_url, and pub_date from parent Document."""
        chunks = indexer.chunk_document(sample_document)
        for chunk in chunks:
            assert chunk.doc_id == sample_document.doc_id
            assert chunk.source == sample_document.source
            assert chunk.source_url == sample_document.source_url
            assert chunk.pub_date == sample_document.publication_date

    def test_chunk_document_empty_abstract(self, indexer) -> None:
        """chunk_document handles a document with empty abstract."""
        doc = Document(
            doc_id="empty_abstract_doc",
            title="Short Title",
            abstract="",
            source="pubmed",
        )
        chunks = indexer.chunk_document(doc)
        # Title alone should produce one chunk
        assert len(chunks) == 1
        assert "Short Title" in chunks[0].text

    def test_chunk_ids_include_doc_id(
        self, indexer, sample_document: Document
    ) -> None:
        """chunk_id is prefixed with the parent doc_id."""
        chunks = indexer.chunk_document(sample_document)
        for chunk in chunks:
            assert chunk.chunk_id.startswith(sample_document.doc_id)


# ---------------------------------------------------------------------------
# Upsert tests
# ---------------------------------------------------------------------------


class TestUpsert:

    def test_upsert_to_qdrant_success(
        self, indexer, mock_qdrant_client, sample_chunk: Chunk
    ) -> None:
        """upsert_to_qdrant calls qdrant.upsert with correct payload."""
        mock_qdrant_client.upsert.return_value = MagicMock()

        result = indexer.upsert_to_qdrant(sample_chunk)

        assert result is True
        mock_qdrant_client.upsert.assert_called_once()
        call_kwargs = mock_qdrant_client.upsert.call_args[1]
        assert call_kwargs["collection_name"] == "test_collection"
        point = call_kwargs["points"][0]
        assert point.payload["chunk_id"] == sample_chunk.chunk_id
        assert point.payload["doc_id"] == sample_chunk.doc_id

    def test_upsert_to_qdrant_uses_provided_vector(
        self, indexer, mock_qdrant_client, sample_chunk: Chunk
    ) -> None:
        """upsert_to_qdrant uses the pre-computed vector without calling embed_chunk."""
        mock_qdrant_client.upsert.return_value = MagicMock()
        custom_vector = [0.5] * 768

        with patch.object(indexer, "embed_chunk") as mock_embed:
            indexer.upsert_to_qdrant(sample_chunk, vector=custom_vector)

        mock_embed.assert_not_called()
        point = mock_qdrant_client.upsert.call_args[1]["points"][0]
        assert point.vector == custom_vector

    def test_upsert_to_qdrant_returns_false_on_error(
        self, indexer, mock_qdrant_client, sample_chunk: Chunk
    ) -> None:
        """upsert_to_qdrant returns False when Qdrant raises."""
        mock_qdrant_client.upsert.side_effect = Exception("write failed")
        result = indexer.upsert_to_qdrant(sample_chunk)
        assert result is False

    def test_upsert_batch_returns_correct_counts(
        self, indexer, mock_qdrant_client, sample_document: Document
    ) -> None:
        """upsert_batch returns (attempted, succeeded) counts."""
        mock_qdrant_client.upsert.return_value = MagicMock()
        chunks = indexer.chunk_document(sample_document)

        attempted, succeeded = indexer.upsert_batch(chunks)

        assert attempted == len(chunks)
        assert succeeded == len(chunks)
        mock_qdrant_client.upsert.assert_called_once()

    def test_upsert_batch_empty_list(self, indexer) -> None:
        """upsert_batch returns (0, 0) for empty input."""
        assert indexer.upsert_batch([]) == (0, 0)

    def test_upsert_document_end_to_end(
        self, indexer, mock_qdrant_client, sample_document: Document
    ) -> None:
        """upsert_document chunks the doc and returns count of written chunks."""
        mock_qdrant_client.upsert.return_value = MagicMock()

        count = indexer.upsert_document(sample_document)

        assert count > 0
        mock_qdrant_client.upsert.assert_called_once()


# ---------------------------------------------------------------------------
# Similarity search tests
# ---------------------------------------------------------------------------


class TestSimilaritySearch:

    def test_similarity_search_returns_chunks(
        self, indexer, mock_qdrant_client
    ) -> None:
        """similarity_search converts Qdrant hits to Chunk objects."""
        mock_hit = MagicMock()
        mock_hit.id = "some-uuid"
        mock_hit.score = 0.91
        mock_hit.payload = {
            "chunk_id": "doc_001_chunk_0000",
            "text": "Metformin is used to treat diabetes.",
            "doc_id": "doc_001",
            "node_cui": "C0025598",
            "source": "pubmed",
            "source_url": "https://pubmed.ncbi.nlm.nih.gov/12345/",
            "pub_date": "2023-03-15T00:00:00",
        }
        mock_response = MagicMock()
        mock_response.points = [mock_hit]
        mock_qdrant_client.query_points.return_value = mock_response

        results = indexer.similarity_search("diabetes treatment", top_k=5)

        assert len(results) == 1
        chunk = results[0]
        assert isinstance(chunk, Chunk)
        assert chunk.score == 0.91
        assert chunk.text == "Metformin is used to treat diabetes."
        assert chunk.node_cui == "C0025598"

    def test_similarity_search_returns_empty_on_error(
        self, indexer, mock_qdrant_client
    ) -> None:
        """similarity_search returns [] when Qdrant raises."""
        mock_qdrant_client.query_points.side_effect = Exception("search failed")

        results = indexer.similarity_search("diabetes")
        assert results == []

    def test_similarity_search_returns_empty_when_disconnected(self) -> None:
        """similarity_search returns [] when not connected."""
        from src.vector.indexer import VectorIndexer

        with patch("src.vector.indexer.QdrantClient", side_effect=Exception):
            vi = VectorIndexer()

        results = vi.similarity_search("diabetes")
        assert results == []

    def test_similarity_search_with_source_filter(
        self, indexer, mock_qdrant_client
    ) -> None:
        """similarity_search passes a payload filter when filter_source is set."""
        mock_response = MagicMock()
        mock_response.points = []
        mock_qdrant_client.query_points.return_value = mock_response

        indexer.similarity_search("diabetes", filter_source="pubmed")

        call_kwargs = mock_qdrant_client.query_points.call_args[1]
        assert call_kwargs.get("query_filter") is not None

    def test_similarity_search_ordered_by_score(
        self, indexer, mock_qdrant_client
    ) -> None:
        """Results maintain Qdrant's ordering (highest score first)."""
        hits = []
        for i, score in enumerate([0.95, 0.80, 0.60]):
            hit = MagicMock()
            hit.id = f"uuid-{i}"
            hit.score = score
            hit.payload = {
                "chunk_id": f"chunk_{i}",
                "text": f"text {i}",
                "doc_id": "doc_x",
                "node_cui": None,
                "source": "pubmed",
                "source_url": None,
                "pub_date": None,
            }
            hits.append(hit)

        mock_response = MagicMock()
        mock_response.points = hits
        mock_qdrant_client.query_points.return_value = mock_response
        results = indexer.similarity_search("query", top_k=3)

        assert results[0].score == 0.95
        assert results[1].score == 0.80
        assert results[2].score == 0.60


# ---------------------------------------------------------------------------
# Deletion tests
# ---------------------------------------------------------------------------


class TestDeletion:

    def test_delete_by_doc_id_removes_points(
        self, indexer, mock_qdrant_client
    ) -> None:
        """delete_by_doc_id calls qdrant.delete with correct point IDs."""
        mock_point = MagicMock()
        mock_point.id = "point-uuid-1"
        # scroll returns (results, next_offset); None offset means done
        mock_qdrant_client.scroll.return_value = ([mock_point], None)

        deleted = indexer.delete_by_doc_id("doc_001")

        assert deleted == 1
        mock_qdrant_client.delete.assert_called_once()

    def test_delete_by_doc_id_returns_zero_when_not_found(
        self, indexer, mock_qdrant_client
    ) -> None:
        """delete_by_doc_id returns 0 when no matching points exist."""
        mock_qdrant_client.scroll.return_value = ([], None)

        deleted = indexer.delete_by_doc_id("nonexistent_doc")
        assert deleted == 0
        mock_qdrant_client.delete.assert_not_called()


# ---------------------------------------------------------------------------
# Stats tests
# ---------------------------------------------------------------------------


class TestStats:

    def test_get_collection_stats_returns_dict(
        self, indexer, mock_qdrant_client
    ) -> None:
        """get_collection_stats returns a dict with expected keys."""
        stats = indexer.get_collection_stats()

        assert isinstance(stats, dict)
        assert "vectors_count" in stats
        assert "status" in stats
        assert "dimension" in stats
        assert stats["dimension"] == 768

    def test_get_collection_stats_error_returns_error_dict(
        self, indexer, mock_qdrant_client
    ) -> None:
        """get_collection_stats returns error dict on exception."""
        mock_qdrant_client.get_collection.side_effect = Exception("not found")

        stats = indexer.get_collection_stats()
        assert "error" in stats


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

INTEGRATION = pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS") != "1",
    reason="Set RUN_INTEGRATION_TESTS=1 and run: docker compose up -d qdrant",
)


@INTEGRATION
def test_qdrant_live_round_trip() -> None:
    """Write a chunk to real Qdrant, search for it, then clean up."""
    from src.vector.indexer import VectorIndexer

    vi = VectorIndexer(collection_name="test_integration_collection")
    assert vi.is_connected(), "Qdrant not reachable — is docker running?"

    # Setup
    vi.ensure_collection()

    doc = Document(
        doc_id="integration_test_doc",
        title="Metformin Treatment for Type 2 Diabetes",
        abstract=(
            "Metformin reduces blood glucose in patients with Type 2 Diabetes. "
            "It is the first-line pharmacological therapy recommended by guidelines."
        ),
        source="pubmed",
        source_url="https://pubmed.ncbi.nlm.nih.gov/99999/",
        publication_date=datetime(2023, 1, 1),
    )

    # Ingest
    count = vi.upsert_document(doc)
    assert count > 0, "No chunks were written"

    # Search
    results = vi.similarity_search("diabetes treatment metformin", top_k=3)
    assert len(results) > 0, "No search results returned"
    assert results[0].doc_id == "integration_test_doc"

    # Stats
    stats = vi.get_collection_stats()
    assert stats.get("vectors_count", 0) > 0

    # Cleanup
    deleted = vi.delete_by_doc_id("integration_test_doc")
    assert deleted > 0

    vi.delete_collection()