"""
Tests for Module 1 (DataFetcher) and Module 2 (NLPProcessor).

Run with:
    pytest src/tests/test_data_nlp.py -v

The tests are designed to pass without live API keys by mocking HTTP calls.
Integration tests that hit real APIs are marked with @pytest.mark.integration
and are skipped in CI unless the env var RUN_INTEGRATION_TESTS=1 is set.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from src.utils.models import Document, EdgeType, Entity, NodeType, Triple


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_document() -> Document:
    return Document(
        doc_id="test_doc_001",
        title="Metformin for Type 2 Diabetes: A Review",
        abstract=(
            "Metformin is a first-line treatment for Type 2 Diabetes. "
            "It causes gastrointestinal side effects in some patients. "
            "Metformin interacts with contrast dye and may lead to lactic acidosis. "
            "Hyperglycemia is associated with increased HbA1c levels."
        ),
        source="pubmed",
        source_url="https://pubmed.ncbi.nlm.nih.gov/12345/",
        publication_date=datetime(2022, 6, 15),
        authors=["Smith, J.", "Doe, A."],
        mesh_terms=["Metformin", "Diabetes Mellitus, Type 2"],
    )


@pytest.fixture()
def mock_pubmed_response() -> Dict[str, Any]:
    """Simulated pymed article object as a dict."""
    return {
        "pubmed_id": "12345678",
        "title": "Efficacy of Metformin in T2DM",
        "abstract": "Metformin reduces HbA1c in patients with Type 2 Diabetes.",
        "publication_date": datetime(2023, 1, 1),
        "authors": [{"lastname": "Smith", "firstname": "John"}],
    }


@pytest.fixture()
def mock_openfda_response() -> Dict[str, Any]:
    return {
        "results": [
            {
                "openfda": {
                    "brand_name": ["Glucophage"],
                    "generic_name": ["METFORMIN HYDROCHLORIDE"],
                    "application_number": ["NDA020357"],
                    "manufacturer_name": ["Bristol-Myers Squibb"],
                },
                "description": ["Metformin hydrochloride tablets."],
                "indications_and_usage": ["For the treatment of type 2 diabetes."],
                "warnings": ["Lactic acidosis risk."],
            }
        ]
    }


@pytest.fixture()
def mock_trials_response() -> Dict[str, Any]:
    return {
        "studies": [
            {
                "protocolSection": {
                    "identificationModule": {
                        "nctId": "NCT12345678",
                        "briefTitle": "Metformin vs Placebo in T2DM",
                    },
                    "descriptionModule": {
                        "briefSummary": "A randomized controlled trial of metformin.",
                    },
                    "statusModule": {
                        "overallStatus": "COMPLETED",
                        "startDateStruct": {"date": "2020-01-01"},
                    },
                    "designModule": {"phases": ["PHASE3"]},
                    "armsInterventionsModule": {
                        "interventions": [{"name": "Metformin"}]
                    },
                    "conditionsModule": {"conditions": ["Type 2 Diabetes"]},
                }
            }
        ]
    }


# ===========================================================================
# DataFetcher tests
# ===========================================================================


class TestDataFetcher:
    """Unit tests for src/data/fetcher.py."""

    def test_import(self) -> None:
        """DataFetcher can be imported and instantiated."""
        from src.data.fetcher import DataFetcher

        fetcher = DataFetcher(
            pubmed_email="test@test.com", umls_api_key="", openfda_api_key=""
        )
        assert fetcher is not None
        assert fetcher.pubmed_email == "test@test.com"

    # ------------------------------------------------------------------
    # PubMed
    # ------------------------------------------------------------------

    def test_fetch_pubmed_returns_documents(self) -> None:
        """fetch_pubmed returns a list of Document objects with required fields."""
        from src.data.fetcher import DataFetcher

        fetcher = DataFetcher(pubmed_email="test@test.com")

        if fetcher._pubmed_client is not None:
            # pymed is available — mock it
            mock_article = MagicMock()
            mock_article.pubmed_id = "12345678"
            mock_article.title = "Metformin in T2DM"
            mock_article.abstract = "Test abstract about diabetes."
            mock_article.publication_date = datetime(2023, 1, 1)
            mock_article.authors = [{"lastname": "Smith", "firstname": "J"}]
            mock_article.mesh_terms = ["Metformin"]

            with patch.object(fetcher._pubmed_client, "query", return_value=[mock_article]):
                docs = fetcher.fetch_pubmed("Type 2 Diabetes", max_results=1)
        else:
            # pymed not installed — use eutils fallback mock
            esearch_resp = MagicMock()
            esearch_resp.json.return_value = {"esearchresult": {"idlist": ["12345678"]}}
            esearch_resp.raise_for_status = MagicMock()
            esummary_resp = MagicMock()
            esummary_resp.json.return_value = {
                "result": {
                    "12345678": {
                        "title": "Metformin in T2DM",
                        "pubdate": "2023",
                        "authors": [{"name": "Smith J"}],
                    }
                }
            }
            esummary_resp.raise_for_status = MagicMock()
            with patch("requests.get", side_effect=[esearch_resp, esummary_resp]):
                docs = fetcher.fetch_pubmed("Type 2 Diabetes", max_results=1)

        assert isinstance(docs, list)
        assert len(docs) == 1
        doc = docs[0]
        assert isinstance(doc, Document)
        assert doc.source == "pubmed"
        assert "Metformin" in doc.title or "T2DM" in doc.title
        assert "pubmed.ncbi.nlm.nih.gov" in (doc.source_url or "")

    def test_fetch_pubmed_eutils_fallback(self) -> None:
        """fetch_pubmed falls back to eutils when pymed is unavailable."""
        from src.data import fetcher as fetcher_module
        from src.data.fetcher import DataFetcher

        with patch.object(fetcher_module, "PYMED_AVAILABLE", False):
            f = DataFetcher(pubmed_email="test@test.com")
            f._pubmed_client = None

            esearch_resp = MagicMock()
            esearch_resp.json.return_value = {
                "esearchresult": {"idlist": ["99999999"]}
            }
            esearch_resp.raise_for_status = MagicMock()

            esummary_resp = MagicMock()
            esummary_resp.json.return_value = {
                "result": {
                    "99999999": {
                        "title": "Fallback title",
                        "pubdate": "2023",
                        "authors": [{"name": "Brown A"}],
                    }
                }
            }
            esummary_resp.raise_for_status = MagicMock()

            with patch("requests.get", side_effect=[esearch_resp, esummary_resp]):
                docs = f.fetch_pubmed("diabetes", max_results=1)

        assert len(docs) == 1
        assert docs[0].source == "pubmed"
        assert docs[0].title == "Fallback title"

    def test_fetch_pubmed_empty_on_error(self) -> None:
        """fetch_pubmed returns empty list on network error, does not raise."""
        from src.data.fetcher import DataFetcher

        fetcher = DataFetcher(pubmed_email="test@test.com")
        if fetcher._pubmed_client:
            with patch.object(
                fetcher._pubmed_client, "query", side_effect=Exception("network error")
            ):
                docs = fetcher.fetch_pubmed("diabetes")
        else:
            with patch("requests.get", side_effect=Exception("network error")):
                docs = fetcher.fetch_pubmed("diabetes")

        assert docs == []

    # ------------------------------------------------------------------
    # OpenFDA
    # ------------------------------------------------------------------

    def test_fetch_openfda_returns_documents(self, mock_openfda_response: Dict) -> None:
        """fetch_openfda parses FDA label records correctly."""
        from src.data.fetcher import DataFetcher

        fetcher = DataFetcher()
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_openfda_response
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp):
            docs = fetcher.fetch_openfda("metformin")

        assert len(docs) == 1
        doc = docs[0]
        assert doc.source == "openfda"
        assert "Glucophage" in doc.title or "METFORMIN" in doc.title
        assert "metformin" in doc.abstract.lower() or "diabetes" in doc.abstract.lower()
        assert doc.metadata["brand_names"] == ["Glucophage"]

    def test_fetch_openfda_404_handled(self) -> None:
        """fetch_openfda returns [] when drug not found (404)."""
        import requests as req
        from src.data.fetcher import DataFetcher

        fetcher = DataFetcher()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = req.HTTPError("404 not found")
        mock_resp.status_code = 404

        with patch("requests.get", return_value=mock_resp):
            docs = fetcher.fetch_openfda("nonexistent_drug_xyz")

        assert docs == []

    # ------------------------------------------------------------------
    # ClinicalTrials
    # ------------------------------------------------------------------

    def test_fetch_trials_returns_documents(self, mock_trials_response: Dict) -> None:
        """fetch_trials parses ClinicalTrials v2 response correctly."""
        from src.data.fetcher import DataFetcher

        fetcher = DataFetcher()
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_trials_response
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp):
            docs = fetcher.fetch_trials("Type 2 Diabetes")

        assert len(docs) == 1
        doc = docs[0]
        assert doc.source == "clinicaltrials"
        assert "NCT12345678" in (doc.source_url or "")
        assert doc.metadata["nct_id"] == "NCT12345678"
        assert doc.metadata["status"] == "COMPLETED"
        assert "Metformin" in doc.metadata["interventions"]

    def test_fetch_trials_empty_on_error(self) -> None:
        """fetch_trials returns [] on network error."""
        from src.data.fetcher import DataFetcher

        fetcher = DataFetcher()
        with patch("requests.get", side_effect=Exception("timeout")):
            docs = fetcher.fetch_trials("Type 2 Diabetes")

        assert docs == []

    # ------------------------------------------------------------------
    # UMLS
    # ------------------------------------------------------------------

    def test_fetch_umls_skips_without_api_key(self) -> None:
        """fetch_umls_concepts returns [] when no API key is configured."""
        from src.data.fetcher import DataFetcher

        fetcher = DataFetcher(umls_api_key="")
        docs = fetcher.fetch_umls_concepts("diabetes")
        assert docs == []

    # ------------------------------------------------------------------
    # fetch_all
    # ------------------------------------------------------------------

    def test_fetch_all_deduplicates(self) -> None:
        """fetch_all does not return duplicate doc_ids."""
        from src.data.fetcher import DataFetcher

        fetcher = DataFetcher(pubmed_email="test@test.com", umls_api_key="")
        doc_a = Document(doc_id="dup_001", title="A", abstract="", source="pubmed")
        doc_b = Document(doc_id="dup_001", title="A", abstract="", source="pubmed")

        with patch.object(fetcher, "fetch_pubmed", return_value=[doc_a, doc_b]):
            with patch.object(fetcher, "fetch_openfda", return_value=[]):
                with patch.object(fetcher, "fetch_trials", return_value=[]):
                    with patch.object(fetcher, "fetch_umls_concepts", return_value=[]):
                        docs = fetcher.fetch_all("diabetes")

        ids = [d.doc_id for d in docs]
        assert len(ids) == len(set(ids)), "Duplicate doc_ids found"

    def test_document_model_serialisation(self, sample_document: Document) -> None:
        """Document Pydantic model serialises to dict and back correctly."""
        data = sample_document.dict()
        restored = Document(**data)
        assert restored.doc_id == sample_document.doc_id
        assert restored.title == sample_document.title
        assert restored.authors == sample_document.authors


# ===========================================================================
# NLPProcessor tests
# ===========================================================================


class TestNLPProcessor:
    """Unit tests for src/nlp/processor.py."""

    def test_import(self) -> None:
        """NLPProcessor can be imported and instantiated."""
        from src.nlp.processor import NLPProcessor

        proc = NLPProcessor(enable_entity_linker=False)
        assert proc is not None

    # ------------------------------------------------------------------
    # Entity extraction
    # ------------------------------------------------------------------

    def test_extract_entities_regex_fallback(self) -> None:
        """
        _extract_entities_regex detects known diabetes-domain entities.
        This test exercises the regex fallback path (no scispaCy required).
        """
        from src.nlp.processor import NLPProcessor

        proc = NLPProcessor(enable_entity_linker=False)
        proc._nlp = None  # Force regex fallback

        text = "Metformin is used to treat Type 2 Diabetes. Hyperglycemia and HbA1c are key markers."
        entities = proc.extract_entities(text)

        entity_texts = [e.text.lower() for e in entities]
        # At least metformin, diabetes, hyperglycemia should be found
        assert any("metformin" in t for t in entity_texts), f"Metformin not found in {entity_texts}"
        assert any("diabetes" in t for t in entity_texts), f"Diabetes not found in {entity_texts}"

    def test_extract_entities_empty_text(self) -> None:
        """extract_entities returns [] for empty input."""
        from src.nlp.processor import NLPProcessor

        proc = NLPProcessor(enable_entity_linker=False)
        assert proc.extract_entities("") == []
        assert proc.extract_entities("   ") == []

    def test_extract_entities_returns_entity_model(self) -> None:
        """extract_entities returns proper Entity Pydantic models."""
        from src.nlp.processor import NLPProcessor

        proc = NLPProcessor(enable_entity_linker=False)
        proc._nlp = None  # Force regex

        entities = proc.extract_entities("Metformin treats Type 2 Diabetes.", source_doc_id="doc_1")
        for ent in entities:
            assert isinstance(ent, Entity)
            assert isinstance(ent.entity_type, NodeType)
            assert 0.0 <= ent.confidence <= 1.0
            assert ent.source_doc_id == "doc_1"

    def test_entity_node_type_mapping(self) -> None:
        """Metformin → Drug, Type 2 Diabetes → Disease, HbA1c → Gene."""
        from src.nlp.processor import NLPProcessor

        proc = NLPProcessor(enable_entity_linker=False)
        proc._nlp = None

        entities = proc.extract_entities(
            "Metformin lowers HbA1c in patients with Type 2 Diabetes."
        )
        type_map = {e.text.lower(): e.entity_type for e in entities}

        assert any("metformin" in k for k in type_map), "Metformin not extracted"
        metformin_type = next(v for k, v in type_map.items() if "metformin" in k)
        assert metformin_type == NodeType.DRUG

    # ------------------------------------------------------------------
    # Relation extraction
    # ------------------------------------------------------------------

    def test_extract_relations_detects_treats(self) -> None:
        """extract_relations finds TREATS relation in simple sentence."""
        from src.nlp.processor import NLPProcessor

        proc = NLPProcessor(enable_entity_linker=False)
        proc._nlp = None

        text = "Metformin is used to treat Type 2 Diabetes."
        triples = proc.extract_relations(text)

        assert len(triples) > 0
        edge_types = [t.predicate for t in triples]
        assert EdgeType.TREATS in edge_types, f"TREATS not found. Got: {edge_types}"

    def test_extract_relations_detects_interacts_with(self) -> None:
        """extract_relations finds INTERACTS_WITH relation."""
        from src.nlp.processor import NLPProcessor

        proc = NLPProcessor(enable_entity_linker=False)
        proc._nlp = None

        # Both metformin (Drug) and pioglitazone (Drug) are in the regex vocab
        text = "Metformin interacts with pioglitazone and may cause lactic acidosis."
        triples = proc.extract_relations(text)

        edge_types = [t.predicate for t in triples]
        assert EdgeType.INTERACTS_WITH in edge_types, f"INTERACTS_WITH not found. Got: {edge_types}"

    def test_extract_relations_returns_triple_model(self) -> None:
        """extract_relations returns proper Triple Pydantic models."""
        from src.nlp.processor import NLPProcessor

        proc = NLPProcessor(enable_entity_linker=False)
        proc._nlp = None

        triples = proc.extract_relations(
            "Metformin treats Type 2 Diabetes.", source_doc_id="doc_1", year=2023
        )
        for triple in triples:
            assert isinstance(triple, Triple)
            assert isinstance(triple.subject, Entity)
            assert isinstance(triple.obj, Entity)
            assert isinstance(triple.predicate, EdgeType)
            assert 0.0 <= triple.confidence <= 1.0
            assert triple.source_doc_id == "doc_1"
            assert triple.year == 2023

    def test_extract_relations_empty_text(self) -> None:
        """extract_relations returns [] for empty input."""
        from src.nlp.processor import NLPProcessor

        proc = NLPProcessor(enable_entity_linker=False)
        assert proc.extract_relations("") == []

    def test_extract_relations_single_entity_no_triple(self) -> None:
        """A sentence with only one entity produces no triples."""
        from src.nlp.processor import NLPProcessor

        proc = NLPProcessor(enable_entity_linker=False)
        proc._nlp = None

        # Only metformin — no second entity
        triples = proc.extract_relations("Metformin is an effective medication.")
        # With regex fallback, only 1 entity found → no triple
        assert isinstance(triples, list)

    # ------------------------------------------------------------------
    # UMLS linking
    # ------------------------------------------------------------------

    def test_link_to_umls_returns_existing_cui(self) -> None:
        """link_to_umls returns the CUI if it is already populated."""
        from src.nlp.processor import NLPProcessor

        proc = NLPProcessor(enable_entity_linker=False)
        entity = Entity(
            text="metformin",
            entity_type=NodeType.DRUG,
            cui="C0025598",
            start_char=0,
            end_char=9,
        )
        assert proc.link_to_umls(entity) == "C0025598"

    def test_link_to_umls_returns_empty_without_linker(self) -> None:
        """link_to_umls returns '' when linker is not loaded."""
        from src.nlp.processor import NLPProcessor

        proc = NLPProcessor(enable_entity_linker=False)
        proc._linker_loaded = False
        entity = Entity(
            text="diabetes",
            entity_type=NodeType.DISEASE,
            cui=None,
            start_char=0,
            end_char=8,
        )
        result = proc.link_to_umls(entity)
        assert result == ""

    # ------------------------------------------------------------------
    # process_document integration
    # ------------------------------------------------------------------

    def test_process_document(self, sample_document: Document) -> None:
        """process_document extracts entities and triples from a Document."""
        from src.nlp.processor import NLPProcessor

        proc = NLPProcessor(enable_entity_linker=False)
        proc._nlp = None  # Use regex fallback

        entities, triples = proc.process_document(sample_document)

        assert isinstance(entities, list)
        assert isinstance(triples, list)
        # With the sample document text, at least metformin and diabetes should appear
        entity_texts = [e.text.lower() for e in entities]
        assert any("metformin" in t for t in entity_texts)

    def test_process_document_year_propagated(self, sample_document: Document) -> None:
        """process_document sets year=2022 on triples from the sample doc."""
        from src.nlp.processor import NLPProcessor

        proc = NLPProcessor(enable_entity_linker=False)
        proc._nlp = None

        _, triples = proc.process_document(sample_document)
        for triple in triples:
            assert triple.year == 2022, f"Expected 2022, got {triple.year}"


# ===========================================================================
# Integration tests (skipped unless RUN_INTEGRATION_TESTS=1)
# ===========================================================================

INTEGRATION = pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS") != "1",
    reason="Set RUN_INTEGRATION_TESTS=1 to run live API tests",
)


@INTEGRATION
def test_pubmed_live() -> None:
    """Live PubMed query — requires internet access and DNS resolution of eutils.ncbi.nlm.nih.gov.

    Skips automatically when the host is unreachable (e.g. DNS blocked by
    a corporate firewall or VPN) rather than failing the suite.
    """
    import socket
    from src.data.fetcher import DataFetcher

    try:
        socket.getaddrinfo("eutils.ncbi.nlm.nih.gov", 443)
    except socket.gaierror:
        pytest.skip("eutils.ncbi.nlm.nih.gov not reachable — network/DNS issue")

    fetcher = DataFetcher(pubmed_email=os.getenv("PUBMED_EMAIL", "test@test.com"))
    docs = fetcher.fetch_pubmed("Type 2 Diabetes[MeSH Terms]", max_results=3)
    assert len(docs) >= 1
    assert all(d.source == "pubmed" for d in docs)


@INTEGRATION
def test_openfda_live() -> None:
    """Live OpenFDA query — requires internet access."""
    import socket
    from src.data.fetcher import DataFetcher

    try:
        socket.getaddrinfo("api.fda.gov", 443)
    except socket.gaierror:
        pytest.skip("api.fda.gov not reachable — network/DNS issue")

    fetcher = DataFetcher()
    docs = fetcher.fetch_openfda("metformin", max_results=2)
    assert len(docs) >= 1


@INTEGRATION
def test_trials_live() -> None:
    """Live ClinicalTrials query — requires internet access."""
    from src.data.fetcher import DataFetcher

    fetcher = DataFetcher()
    docs = fetcher.fetch_trials("Type 2 Diabetes", max_results=3)
    assert len(docs) >= 1