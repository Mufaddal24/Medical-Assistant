"""
DataFetcher — Module 1 of the Medical Knowledge Assistant pipeline.

Responsibilities
----------------
* fetch_pubmed(query, max_results)  → List[Document]
* fetch_openfda(drug_name)          → List[Document]
* fetch_trials(condition)           → List[Document]
* fetch_umls_concepts(term)         → List[Document]   (concept normalisation)

All methods return a uniform List[Document] so the NLPProcessor can consume
them without knowing which source the data came from.

Environment variables (loaded from .env)
-----------------------------------------
PUBMED_EMAIL        e-mail address required by NCBI's E-utilities
UMLS_API_KEY        API key from https://uts.nlm.nih.gov/uts/signup-login
OPENFDA_API_KEY     optional – increases rate limit
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

# pymed provides a Pythonic wrapper around PubMed's E-utilities
try:
    from pymed import PubMed
    from pymed.article import PubMedArticle
    PYMED_AVAILABLE = True
except ImportError:  # pragma: no cover
    PYMED_AVAILABLE = False

from src.utils.models import Document

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPENFDA_BASE = "https://api.fda.gov/drug/label.json"
TRIALS_BASE = "https://clinicaltrials.gov/api/v2/studies"
UMLS_BASE = "https://uts-ws.nlm.nih.gov/rest"

_REQUEST_TIMEOUT = 30  # seconds
_RETRY_BACKOFF = [1, 2, 4]  # exponential back-off delays in seconds


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _make_doc_id(source: str, raw_id: str) -> str:
    """Return a deterministic, URL-safe document ID."""
    return hashlib.md5(f"{source}::{raw_id}".encode()).hexdigest()


def _get_with_retry(url: str, params: Dict[str, Any], timeout: int = _REQUEST_TIMEOUT) -> requests.Response:
    """
    Perform a GET request with exponential back-off retry on transient errors.

    Parameters
    ----------
    url:
        Target URL.
    params:
        Query-string parameters.
    timeout:
        Per-attempt timeout in seconds.

    Returns
    -------
    requests.Response
        Successful response object.

    Raises
    ------
    requests.HTTPError
        If all retry attempts fail.
    """
    last_exc: Optional[Exception] = None
    for delay in [0] + _RETRY_BACKOFF:
        if delay:
            logger.info("Retrying %s in %s s …", url, delay)
            time.sleep(delay)
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning("Request failed: %s", exc)
    raise requests.HTTPError(f"All retries exhausted for {url}") from last_exc


# ---------------------------------------------------------------------------
# DataFetcher class
# ---------------------------------------------------------------------------


class DataFetcher:
    """
    Fetches medical documents from PubMed, OpenFDA, ClinicalTrials.gov,
    and UMLS.

    Usage
    -----
    >>> fetcher = DataFetcher()
    >>> docs = fetcher.fetch_pubmed("Type 2 Diabetes", max_results=5)
    >>> docs[0].source
    'pubmed'
    """

    def __init__(
        self,
        pubmed_email: Optional[str] = None,
        umls_api_key: Optional[str] = None,
        openfda_api_key: Optional[str] = None,
    ) -> None:
        self.pubmed_email = pubmed_email or os.getenv("PUBMED_EMAIL", "user@example.com")
        self.umls_api_key = umls_api_key or os.getenv("UMLS_API_KEY", "")
        self.openfda_api_key = openfda_api_key or os.getenv("OPENFDA_API_KEY", "")

        if PYMED_AVAILABLE:
            self._pubmed_client = PubMed(tool="MedKGAssistant", email=self.pubmed_email)
        else:
            self._pubmed_client = None
            logger.warning("pymed not installed — PubMed fetcher will use E-utilities REST fallback")

        logger.info("DataFetcher initialised (email=%s)", self.pubmed_email)

    # ------------------------------------------------------------------
    # 1. PubMed
    # ------------------------------------------------------------------

    def fetch_pubmed(self, query: str, max_results: int = 10) -> List[Document]:
        """
        Fetch PubMed articles matching *query* and return them as Documents.

        Parameters
        ----------
        query:
            Free-text or MeSH query string,
            e.g. ``"Type 2 Diabetes[MeSH Terms]"``.
        max_results:
            Maximum number of articles to retrieve.

        Returns
        -------
        List[Document]
            One Document per PubMed article.
        """
        logger.info("Fetching PubMed articles for query=%r max=%d", query, max_results)

        if self._pubmed_client is not None:
            return self._fetch_pubmed_pymed(query, max_results)
        return self._fetch_pubmed_eutils(query, max_results)

    def _fetch_pubmed_pymed(self, query: str, max_results: int) -> List[Document]:
        """Internal: use pymed library."""
        docs: List[Document] = []
        try:
            results = self._pubmed_client.query(query, max_results=max_results)
            for article in results:
                try:
                    docs.append(self._pubmed_article_to_doc(article))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Skipping malformed PubMed article: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.error("PubMed fetch failed: %s", exc)
        logger.info("PubMed returned %d documents", len(docs))
        return docs

    def _pubmed_article_to_doc(self, article: "PubMedArticle") -> Document:
        """Convert a pymed PubMedArticle to our Document model."""
        pmid = str(article.pubmed_id or "").strip().split("\n")[0]
        title = str(article.title or "").strip()
        abstract = str(article.abstract or "").strip()

        pub_date: Optional[datetime] = None
        if article.publication_date:
            try:
                if isinstance(article.publication_date, datetime):
                    pub_date = article.publication_date
                else:
                    pub_date = datetime(
                        article.publication_date.year,
                        article.publication_date.month or 1,
                        article.publication_date.day or 1,
                    )
            except Exception:  # noqa: BLE001
                pub_date = None

        authors: List[str] = []
        if article.authors:
            for a in article.authors:
                if isinstance(a, dict):
                    lastname = a.get("lastname", "")
                    firstname = a.get("firstname", "")
                    name = f"{lastname}, {firstname}".strip(", ")
                    if name:
                        authors.append(name)

        mesh_terms: List[str] = []
        if hasattr(article, "mesh_terms") and article.mesh_terms:
            mesh_terms = [str(m) for m in article.mesh_terms if m]

        return Document(
            doc_id=_make_doc_id("pubmed", pmid),
            title=title or f"PubMed:{pmid}",
            abstract=abstract,
            source="pubmed",
            source_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
            publication_date=pub_date,
            authors=authors,
            mesh_terms=mesh_terms,
            metadata={"pmid": pmid},
        )

    def _fetch_pubmed_eutils(self, query: str, max_results: int) -> List[Document]:
        """Fallback: call NCBI E-utilities REST API directly."""
        docs: List[Document] = []
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        try:
            # Step 1: esearch to get PMIDs
            search_resp = _get_with_retry(
                f"{base}/esearch.fcgi",
                {
                    "db": "pubmed",
                    "term": query,
                    "retmax": max_results,
                    "retmode": "json",
                    "email": self.pubmed_email,
                },
            )
            pmids = search_resp.json().get("esearchresult", {}).get("idlist", [])
            if not pmids:
                return docs

            # Step 2: efetch summary for each PMID
            fetch_resp = _get_with_retry(
                f"{base}/esummary.fcgi",
                {
                    "db": "pubmed",
                    "id": ",".join(pmids),
                    "retmode": "json",
                    "email": self.pubmed_email,
                },
            )
            result = fetch_resp.json().get("result", {})
            for pmid in pmids:
                art = result.get(pmid, {})
                if not art:
                    continue
                title = art.get("title", "")
                pub_date_str = art.get("pubdate", "")
                pub_date: Optional[datetime] = None
                try:
                    pub_date = datetime.strptime(pub_date_str[:4], "%Y")
                except Exception:  # noqa: BLE001
                    pass
                authors = [a.get("name", "") for a in art.get("authors", [])]
                docs.append(
                    Document(
                        doc_id=_make_doc_id("pubmed", pmid),
                        title=title,
                        abstract="",  # esummary does not include abstract
                        source="pubmed",
                        source_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        publication_date=pub_date,
                        authors=[a for a in authors if a],
                        metadata={"pmid": pmid},
                    )
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("PubMed E-utilities fallback failed: %s", exc)
        logger.info("PubMed (eutils) returned %d documents", len(docs))
        return docs

    # ------------------------------------------------------------------
    # 2. OpenFDA
    # ------------------------------------------------------------------

    def fetch_openfda(self, drug_name: str, max_results: int = 5) -> List[Document]:
        """
        Fetch drug label information from the FDA openFDA API.

        Parameters
        ----------
        drug_name:
            Generic or brand name of the drug, e.g. ``"metformin"``.
        max_results:
            Maximum number of label records to return.

        Returns
        -------
        List[Document]
            One Document per FDA drug label record.
        """
        logger.info("Fetching OpenFDA labels for drug=%r", drug_name)
        docs: List[Document] = []
        params: Dict[str, Any] = {
            "search": f'openfda.generic_name:"{drug_name}"',
            "limit": max_results,
        }
        if self.openfda_api_key:
            params["api_key"] = self.openfda_api_key

        try:
            resp = _get_with_retry(OPENFDA_BASE, params)
            data = resp.json()
            for record in data.get("results", []):
                docs.append(self._openfda_record_to_doc(record, drug_name))
        except requests.HTTPError as exc:
            if "404" in str(exc):
                logger.warning("OpenFDA: no labels found for %r", drug_name)
            else:
                logger.error("OpenFDA fetch failed: %s", exc)

        logger.info("OpenFDA returned %d documents", len(docs))
        return docs

    def _openfda_record_to_doc(self, record: Dict[str, Any], drug_name: str) -> Document:
        """Convert an OpenFDA label record to our Document model."""
        openfda = record.get("openfda", {})
        brand_names: List[str] = openfda.get("brand_name", [])
        generic_names: List[str] = openfda.get("generic_name", [])

        title_parts = brand_names[:1] or generic_names[:1] or [drug_name]
        title = f"FDA Label: {title_parts[0]}"

        # Flatten the most informative text sections into abstract
        text_sections = []
        for section in [
            "description",
            "indications_and_usage",
            "warnings",
            "adverse_reactions",
            "drug_interactions",
            "mechanism_of_action",
        ]:
            val = record.get(section)
            if isinstance(val, list):
                text_sections.append(" ".join(val))
            elif isinstance(val, str):
                text_sections.append(val)

        abstract = " ".join(text_sections)[:4000]  # truncate for safety

        # Build a pseudo-URL from the application number if present
        app_numbers = openfda.get("application_number", [])
        source_url: Optional[str] = None
        if app_numbers:
            source_url = f"https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm?event=overview.process&ApplNo={app_numbers[0]}"

        raw_id = (app_numbers[:1] or brand_names[:1] or generic_names[:1] or [drug_name])[0]

        return Document(
            doc_id=_make_doc_id("openfda", raw_id),
            title=title,
            abstract=abstract,
            source="openfda",
            source_url=source_url,
            metadata={
                "brand_names": brand_names,
                "generic_names": generic_names,
                "application_numbers": app_numbers,
                "manufacturer": openfda.get("manufacturer_name", []),
            },
        )

    # ------------------------------------------------------------------
    # 3. ClinicalTrials.gov
    # ------------------------------------------------------------------

    def fetch_trials(self, condition: str, max_results: int = 10) -> List[Document]:
        """
        Fetch clinical trials from ClinicalTrials.gov v2 API.

        Parameters
        ----------
        condition:
            Disease or condition name, e.g. ``"Type 2 Diabetes"``.
        max_results:
            Maximum number of studies to return.

        Returns
        -------
        List[Document]
            One Document per clinical trial.
        """
        logger.info("Fetching ClinicalTrials for condition=%r", condition)
        docs: List[Document] = []
        params: Dict[str, Any] = {
            "query.cond": condition,
            "pageSize": max_results,
            "format": "json",
        }

        try:
            resp = _get_with_retry(TRIALS_BASE, params)
            data = resp.json()
            studies = data.get("studies", [])
            for study in studies:
                try:
                    docs.append(self._trial_to_doc(study))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Skipping malformed trial: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.error("ClinicalTrials fetch failed: %s", exc)

        logger.info("ClinicalTrials returned %d documents", len(docs))
        return docs

    def _trial_to_doc(self, study: Dict[str, Any]) -> Document:
        """Convert a ClinicalTrials v2 study record to our Document model."""
        # v2 API nests data under protocolSection
        protocol = study.get("protocolSection", study)
        ident = protocol.get("identificationModule", {})
        desc = protocol.get("descriptionModule", {})
        status = protocol.get("statusModule", {})
        design = protocol.get("designModule", {})
        arms = protocol.get("armsInterventionsModule", {})
        conditions_module = protocol.get("conditionsModule", {})

        nct_id: str = ident.get("nctId", "")
        title: str = ident.get("briefTitle") or ident.get("officialTitle") or nct_id

        abstract_parts: List[str] = []
        if desc.get("briefSummary"):
            abstract_parts.append(desc["briefSummary"])
        if desc.get("detailedDescription"):
            abstract_parts.append(desc["detailedDescription"])
        abstract = " ".join(abstract_parts)[:4000]

        start_date_str: str = status.get("startDateStruct", {}).get("date", "")
        pub_date: Optional[datetime] = None
        if start_date_str:
            for fmt in ("%Y-%m-%d", "%B %Y", "%Y"):
                try:
                    pub_date = datetime.strptime(start_date_str, fmt)
                    break
                except ValueError:
                    continue

        interventions: List[str] = [
            i.get("name", "") for i in arms.get("interventions", []) if i.get("name")
        ]
        conditions: List[str] = conditions_module.get("conditions", [])

        return Document(
            doc_id=_make_doc_id("trials", nct_id),
            title=title,
            abstract=abstract,
            source="clinicaltrials",
            source_url=f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else None,
            publication_date=pub_date,
            metadata={
                "nct_id": nct_id,
                "phase": design.get("phases", []),
                "status": status.get("overallStatus", ""),
                "interventions": interventions,
                "conditions": conditions,
            },
        )

    # ------------------------------------------------------------------
    # 4. UMLS concept lookup
    # ------------------------------------------------------------------

    def fetch_umls_concepts(self, term: str, max_results: int = 5) -> List[Document]:
        """
        Search the UMLS Metathesaurus for a term and return concept documents.

        Requires a valid UMLS_API_KEY in the environment.

        Parameters
        ----------
        term:
            Medical term to look up, e.g. ``"Type 2 Diabetes"``.
        max_results:
            Maximum number of concept records to return.

        Returns
        -------
        List[Document]
            One Document per UMLS concept.
        """
        if not self.umls_api_key:
            logger.warning("UMLS_API_KEY not set — skipping UMLS fetch for %r", term)
            return []

        logger.info("Fetching UMLS concepts for term=%r", term)
        docs: List[Document] = []

        try:
            # Step 1: obtain a TGT (Ticket-Granting Ticket)
            tgt = self._get_umls_tgt()
            if not tgt:
                return []

            # Step 2: get a service ticket
            ticket = self._get_umls_service_ticket(tgt)
            if not ticket:
                return []

            # Step 3: search
            resp = _get_with_retry(
                f"{UMLS_BASE}/search/current",
                {
                    "string": term,
                    "ticket": ticket,
                    "pageSize": max_results,
                    "returnIdType": "concept",
                },
            )
            data = resp.json()
            results_list = data.get("result", {}).get("results", [])

            for concept in results_list:
                cui = concept.get("ui", "")
                name = concept.get("name", term)
                docs.append(
                    Document(
                        doc_id=_make_doc_id("umls", cui),
                        title=f"UMLS Concept: {name}",
                        abstract=f"UMLS CUI {cui}: {name}",
                        source="umls",
                        source_url=f"https://uts.nlm.nih.gov/uts/umls/concept/{cui}",
                        metadata={"cui": cui, "name": name, "root_source": concept.get("rootSource", "")},
                    )
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("UMLS fetch failed: %s", exc)

        logger.info("UMLS returned %d concepts", len(docs))
        return docs

    def _get_umls_tgt(self) -> Optional[str]:
        """Obtain a UMLS Ticket-Granting Ticket."""
        try:
            resp = requests.post(
                f"{UMLS_BASE}/auth/authenticateUser",
                data={"apikey": self.umls_api_key},
                timeout=_REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            # TGT is embedded in the Location header or the response body
            location = resp.headers.get("location") or resp.text
            if "TGT" in location:
                tgt = location.split("/")[-1]
                return tgt
        except Exception as exc:  # noqa: BLE001
            logger.warning("UMLS TGT acquisition failed: %s", exc)
        return None

    def _get_umls_service_ticket(self, tgt: str) -> Optional[str]:
        """Obtain a single-use UMLS service ticket from a TGT."""
        try:
            resp = requests.post(
                f"{UMLS_BASE}/auth/ticket/{tgt}",
                data={"service": "http://umlsks.nlm.nih.gov"},
                timeout=_REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.text.strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("UMLS service ticket acquisition failed: %s", exc)
        return None

    # ------------------------------------------------------------------
    # Convenience: fetch all sources
    # ------------------------------------------------------------------

    def fetch_all(
        self,
        condition: str,
        drug_name: Optional[str] = None,
        max_results_per_source: int = 5,
    ) -> List[Document]:
        """
        Fetch documents from all sources for a given medical condition.

        Parameters
        ----------
        condition:
            The condition to query across sources (e.g. ``"Type 2 Diabetes"``).
        drug_name:
            Optional drug name to pass to OpenFDA (defaults to *condition*).
        max_results_per_source:
            Maximum results per data source.

        Returns
        -------
        List[Document]
            Combined documents from all sources, de-duplicated by doc_id.
        """
        all_docs: List[Document] = []
        seen: set[str] = set()

        sources = [
            lambda: self.fetch_pubmed(condition, max_results_per_source),
            lambda: self.fetch_openfda(drug_name or condition, max_results_per_source),
            lambda: self.fetch_trials(condition, max_results_per_source),
            lambda: self.fetch_umls_concepts(condition, max_results_per_source),
        ]

        for fetch_fn in sources:
            try:
                for doc in fetch_fn():
                    if doc.doc_id not in seen:
                        seen.add(doc.doc_id)
                        all_docs.append(doc)
            except Exception as exc:  # noqa: BLE001
                logger.error("Source fetch error: %s", exc)

        logger.info("fetch_all returned %d unique documents", len(all_docs))
        return all_docs