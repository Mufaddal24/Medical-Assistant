"""
NLPProcessor — Module 2 of the Medical Knowledge Assistant pipeline.

Responsibilities
----------------
* extract_entities(text)  → List[Entity]
    Uses scispaCy (en_core_sci_lg) for biomedical NER.
* extract_relations(text) → List[Triple]
    Rule-based + dependency-parse co-occurrence relations, with optional
    transformer-based classification for key edge types.
* link_to_umls(entity)    → CUI string
    Uses MedSpaCy / scispaCy's EntityLinker to map surface forms to
    UMLS Concept Unique Identifiers.

The processor is stateless per call; load it once and call repeatedly.

Dependencies (must be installed)
---------------------------------
    pip install scispacy medspacy
    pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz

Environment variables
---------------------
    SCISPACY_MODEL  (optional) override the default model name.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Dict, List, Optional, Tuple

from src.utils.models import Document, EdgeType, Entity, NodeType, Triple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt to import scispaCy / spaCy — allow graceful degradation for CI
# ---------------------------------------------------------------------------

try:
    import spacy
    from spacy.language import Language
    from spacy.tokens import Doc, Span
    SPACY_AVAILABLE = True
except ImportError:  # pragma: no cover
    SPACY_AVAILABLE = False
    logger.warning("spaCy not installed — NLPProcessor will use regex fallback")

# ---------------------------------------------------------------------------
# Mapping from scispaCy entity labels → our NodeType enum
# ---------------------------------------------------------------------------

_LABEL_MAP: Dict[str, NodeType] = {
    # scispaCy en_core_sci_lg labels
    "DISEASE": NodeType.DISEASE,
    "DISORDER": NodeType.DISEASE,
    "SIGN_OR_SYMPTOM": NodeType.SYMPTOM,
    "SYMPTOM": NodeType.SYMPTOM,
    "CHEMICAL": NodeType.DRUG,
    "DRUG": NodeType.DRUG,
    "SIMPLE_CHEMICAL": NodeType.DRUG,
    "GENE_OR_GENE_PRODUCT": NodeType.GENE,
    "GENE": NodeType.GENE,
    "PROTEIN": NodeType.GENE,
    "DNA": NodeType.GENE,
    "CELL": NodeType.GENE,         # fallback for cellular entities
    "ORGANISM": NodeType.DISEASE,  # may represent pathogen
    # Generic / catch-all
    "ORG": NodeType.DRUG,          # sometimes pharmaceutical companies
    "ENTITY": NodeType.DISEASE,
}

# ---------------------------------------------------------------------------
# Relation extraction patterns
# ---------------------------------------------------------------------------

# (pattern, EdgeType) — very conservative list to minimise false positives
_RELATION_PATTERNS: List[Tuple[re.Pattern[str], EdgeType]] = [
    (re.compile(r"\b(treats?|treatment of|therapy for|used to treat)\b", re.I), EdgeType.TREATS),
    (re.compile(r"\b(causes?|induces?|leads? to|results? in|associated with the development of)\b", re.I), EdgeType.CAUSES),
    (re.compile(r"\b(interacts? with|drug interaction|combined with)\b", re.I), EdgeType.INTERACTS_WITH),
    (re.compile(r"\b(associated with|linked to|correlated with|related to)\b", re.I), EdgeType.ASSOCIATED_WITH),
    (re.compile(r"\b(investigated in|studied in|enrolled in|clinical trial)\b", re.I), EdgeType.INVESTIGATED_IN),
    (re.compile(r"\b(cites?|cited by|referenced in|reported in)\b", re.I), EdgeType.CITED_BY),
]

# Sentence boundary pattern (simple, avoids NLTK dependency)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences (simple regex — spaCy senter is used when available)."""
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def _make_entity_id(text: str, entity_type: str) -> str:
    """Generate a stable ID for an entity based on canonical form + type."""
    canonical = text.lower().strip()
    return hashlib.md5(f"{entity_type}::{canonical}".encode()).hexdigest()[:16]


class NLPProcessor:
    """
    Extracts biomedical named entities, relations, and UMLS CUIs from text.

    Parameters
    ----------
    model_name:
        scispaCy model name. Defaults to ``en_core_sci_lg``.
    enable_entity_linker:
        Whether to load the UMLS entity linker component (slow to load,
        requires downloading the UMLS KB). Defaults to True when spaCy is
        available.
    linker_name:
        Name of the scispaCy linker to use. ``"umls"`` is the default.

    Example
    -------
    >>> proc = NLPProcessor()
    >>> entities = proc.extract_entities("Metformin is used to treat Type 2 Diabetes.")
    >>> entities[0].entity_type
    <NodeType.DRUG: 'Drug'>
    """

    def __init__(
        self,
        model_name: str = "en_core_sci_lg",
        enable_entity_linker: bool = True,
        linker_name: str = "umls",
    ) -> None:
        self._model_name = model_name
        self._linker_name = linker_name
        self._nlp: Optional["Language"] = None
        self._linker_loaded: bool = False

        if SPACY_AVAILABLE:
            self._load_model(enable_entity_linker)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, enable_linker: bool) -> None:
        """Load scispaCy model and optional entity linker."""
        logger.info("Loading scispaCy model: %s", self._model_name)
        try:
            import spacy  # noqa: PLC0415 — conditional import
            self._nlp = spacy.load(self._model_name)
            logger.info("scispaCy model loaded successfully")
        except OSError:
            logger.error(
                "scispaCy model '%s' not found. Install it with:\n"
                "  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/"
                "releases/v0.5.4/%s-0.5.4.tar.gz",
                self._model_name,
                self._model_name,
            )
            self._nlp = None
            return

        if enable_linker:
            self._load_entity_linker()

    def _load_entity_linker(self) -> None:
        """Attach scispaCy's UMLS entity linker to the pipeline."""
        if self._nlp is None:
            return
        try:
            # scispaCy >= 0.5 uses the spacy-entity-linker interface
            if "scispacy_linker" not in self._nlp.pipe_names:
                self._nlp.add_pipe(
                    "scispacy_linker",
                    config={"resolve_abbreviations": True, "linker_name": self._linker_name},
                )
            self._linker_loaded = True
            logger.info("UMLS entity linker loaded")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not load entity linker (UMLS KB may not be downloaded): %s", exc
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_entities(self, text: str, source_doc_id: Optional[str] = None) -> List[Entity]:
        """
        Extract biomedical named entities from *text*.

        Uses scispaCy's NER when available; falls back to a regex-based
        heuristic when scispaCy is not installed.

        Parameters
        ----------
        text:
            The raw text to process.
        source_doc_id:
            Optional document ID to attach to each extracted entity.

        Returns
        -------
        List[Entity]
            De-duplicated list of entities (merged by canonical lowercase form).
        """
        if not text or not text.strip():
            return []

        if self._nlp is not None:
            return self._extract_entities_spacy(text, source_doc_id)
        return self._extract_entities_regex(text, source_doc_id)

    def extract_relations(
        self,
        text: str,
        source_doc_id: Optional[str] = None,
        year: Optional[int] = None,
    ) -> List[Triple]:
        """
        Extract subject–predicate–object relation triples from *text*.

        Strategy:
        1. Extract entities in each sentence.
        2. Check for relation keywords between entity pairs within the same sentence.
        3. Assign the most specific matching EdgeType.

        Parameters
        ----------
        text:
            Source text (typically an abstract or section).
        source_doc_id:
            Document identifier for provenance.
        year:
            Publication year of the source document.

        Returns
        -------
        List[Triple]
            Extracted relation triples.
        """
        if not text or not text.strip():
            return []

        triples: List[Triple] = []

        if self._nlp is not None:
            doc = self._nlp(text[:50000])  # cap at 50k chars for performance
            sentences = list(doc.sents)
        else:
            sentences_text = _split_sentences(text)
            sentences = sentences_text  # type: ignore[assignment]

        for sent in sentences:
            sent_text = sent.text if hasattr(sent, "text") else str(sent)
            sent_entities: List[Entity] = self.extract_entities(sent_text, source_doc_id)
            if len(sent_entities) < 2:
                continue

            # Attempt to find a relation keyword in the sentence
            edge_type, confidence = self._classify_relation(sent_text)
            if edge_type is None:
                # Default to ASSOCIATED_WITH for co-occurrence within a sentence
                edge_type = EdgeType.ASSOCIATED_WITH
                confidence = 0.3

            # Pair entities: subject (first entity) → object (second entity)
            for i in range(len(sent_entities) - 1):
                subj = sent_entities[i]
                for j in range(i + 1, len(sent_entities)):
                    obj_ent = sent_entities[j]
                    if subj.entity_type == obj_ent.entity_type:
                        # Same type — still valid (e.g. drug–drug interaction)
                        pass
                    triples.append(
                        Triple(
                            subject=subj,
                            predicate=edge_type,
                            obj=obj_ent,
                            confidence=confidence,
                            source_doc_id=source_doc_id,
                            year=year,
                            evidence_text=sent_text[:500],
                        )
                    )

        logger.debug(
            "extract_relations found %d triples in doc %s", len(triples), source_doc_id
        )
        return triples

    def link_to_umls(self, entity: Entity) -> str:
        """
        Attempt to map an Entity to a UMLS Concept Unique Identifier.

        If the entity already has a CUI, it is returned immediately.
        Otherwise scispaCy's entity linker is consulted (requires the UMLS
        KB to have been downloaded).

        Parameters
        ----------
        entity:
            The Entity to link.

        Returns
        -------
        str
            The CUI string (e.g. ``"C0011860"``), or an empty string if
            linking fails.
        """
        if entity.cui:
            return entity.cui

        if not SPACY_AVAILABLE or self._nlp is None or not self._linker_loaded:
            logger.debug("Entity linker not available — returning empty CUI for %r", entity.text)
            return ""

        try:
            doc = self._nlp(entity.text)
            for ent in doc.ents:
                if hasattr(ent._, "kb_ents") and ent._.kb_ents:
                    top_cui, score = ent._.kb_ents[0]
                    logger.debug(
                        "Linked %r → CUI=%s (score=%.3f)", entity.text, top_cui, score
                    )
                    return top_cui
        except Exception as exc:  # noqa: BLE001
            logger.warning("UMLS linking failed for %r: %s", entity.text, exc)

        return ""

    def process_document(self, doc: Document) -> Tuple[List[Entity], List[Triple]]:
        """
        Convenience method to extract entities and relations from a Document.

        Processes the title + abstract together for better context.

        Parameters
        ----------
        doc:
            A Document returned by DataFetcher.

        Returns
        -------
        Tuple[List[Entity], List[Triple]]
            (entities, triples) extracted from the document.
        """
        full_text = f"{doc.title}. {doc.abstract}".strip()
        year: Optional[int] = None
        if doc.publication_date:
            year = doc.publication_date.year

        entities = self.extract_entities(full_text, source_doc_id=doc.doc_id)

        # Enrich entities with UMLS CUIs
        for entity in entities:
            if not entity.cui:
                entity.cui = self.link_to_umls(entity)

        triples = self.extract_relations(full_text, source_doc_id=doc.doc_id, year=year)

        logger.info(
            "Processed doc %s: %d entities, %d triples",
            doc.doc_id,
            len(entities),
            len(triples),
        )
        return entities, triples

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_entities_spacy(
        self, text: str, source_doc_id: Optional[str]
    ) -> List[Entity]:
        """Use scispaCy NER pipeline to extract entities."""
        assert self._nlp is not None  # noqa: S101 — guarded by caller
        doc = self._nlp(text[:50000])
        seen: Dict[str, Entity] = {}

        for ent in doc.ents:
            node_type = _LABEL_MAP.get(ent.label_, NodeType.DISEASE)
            canonical = ent.text.lower().strip()

            # Attempt to get UMLS CUI from linker
            cui: str = ""
            if self._linker_loaded and hasattr(ent._, "kb_ents") and ent._.kb_ents:
                cui = ent._.kb_ents[0][0]

            entity_id = _make_entity_id(canonical, node_type.value)

            if entity_id not in seen:
                seen[entity_id] = Entity(
                    text=ent.text,
                    entity_type=node_type,
                    cui=cui or None,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    confidence=1.0,
                    canonical_name=ent.text,
                    source_doc_id=source_doc_id,
                )

        return list(seen.values())

    def _extract_entities_regex(
        self, text: str, source_doc_id: Optional[str]
    ) -> List[Entity]:
        """
        Fallback regex-based NER using a curated list of medical terms.
        Used only when scispaCy is unavailable (e.g. CI environments).
        """
        # Minimal vocabulary for demonstration — extend as needed
        patterns: List[Tuple[re.Pattern[str], NodeType]] = [
            (re.compile(r"\b(type\s*[12]\s*diabetes|diabetes mellitus|T2DM|T1DM)\b", re.I), NodeType.DISEASE),
            (re.compile(r"\b(metformin|insulin|glipizide|pioglitazone|sitagliptin|empagliflozin|liraglutide|semaglutide)\b", re.I), NodeType.DRUG),
            (re.compile(r"\b(hyperglycemia|hypoglycemia|polyuria|polydipsia|neuropathy|retinopathy|nephropathy)\b", re.I), NodeType.SYMPTOM),
            (re.compile(r"\b(HbA1c|GLUT4|GLUT2|IRS-1|TCF7L2|KCNJ11|PPARG|INS|GCK)\b"), NodeType.GENE),
        ]

        seen: Dict[str, Entity] = {}
        for pattern, node_type in patterns:
            for match in pattern.finditer(text):
                canonical = match.group(0).lower().strip()
                entity_id = _make_entity_id(canonical, node_type.value)
                if entity_id not in seen:
                    seen[entity_id] = Entity(
                        text=match.group(0),
                        entity_type=node_type,
                        cui=None,
                        start_char=match.start(),
                        end_char=match.end(),
                        confidence=0.7,
                        canonical_name=match.group(0),
                        source_doc_id=source_doc_id,
                    )
        return list(seen.values())

    def _classify_relation(self, text: str) -> Tuple[Optional[EdgeType], float]:
        """
        Match relation patterns against text, choosing the match that appears
        earliest (lowest start position) to avoid false positives from
        incidental later patterns.

        Returns
        -------
        Tuple[Optional[EdgeType], float]
            (edge_type, confidence) — (None, 0.0) if no pattern matches.
        """
        best: Optional[Tuple[int, EdgeType]] = None  # (match_start, edge_type)
        for pattern, edge_type in _RELATION_PATTERNS:
            m = pattern.search(text)
            if m:
                if best is None or m.start() < best[0]:
                    best = (m.start(), edge_type)
        if best is None:
            return None, 0.0
        edge_type = best[1]
        confidence = 0.6 if edge_type != EdgeType.ASSOCIATED_WITH else 0.4
        return edge_type, confidence
