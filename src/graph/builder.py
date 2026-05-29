"""
GraphBuilder — Module 3 of the Medical Knowledge Assistant pipeline.

Responsibilities
----------------
* add_node(entity, props)         — MERGE a single node into Neo4j
* add_edge(src, relation, tgt)    — MERGE a single relationship into Neo4j
* upsert_batch(triples)           — idempotent bulk write of Triple objects
* get_subgraph(query_cuis, hops)  — retrieve a multi-hop subgraph for retrieval
* get_node_by_cui(cui)            — fetch a single node by UMLS CUI
* create_indexes()                — create Neo4j indexes on first run

All Cypher writes use MERGE (never CREATE) so the method is safe to call
multiple times with the same data — the graph stays consistent.

Confidence propagation
----------------------
Multi-hop path confidence = product of all edge confidence values along
the path. This is computed in get_subgraph() and attached to GraphSubgraph.

Environment variables (loaded from .env)
-----------------------------------------
NEO4J_URI       bolt://localhost:7687
NEO4J_USER      neo4j
NEO4J_PASSWORD  medkg_password
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, AuthError

from src.utils.models import (
    EdgeType,
    Entity,
    GraphEdge,
    GraphNode,
    GraphSubgraph,
    NodeType,
    Triple,
)

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cypher templates
# ---------------------------------------------------------------------------

# MERGE on id (CUI) so re-running ingestion never duplicates nodes
_MERGE_NODE_CQL = """
MERGE (n:{label} {{id: $id}})
ON CREATE SET
    n.name            = $name,
    n.type            = $type,
    n.source_url      = $source_url,
    n.last_updated    = $last_updated,
    n.confidence_score = $confidence_score
ON MATCH SET
    n.name            = CASE WHEN $name <> '' THEN $name ELSE n.name END,
    n.source_url      = CASE WHEN $source_url IS NOT NULL THEN $source_url ELSE n.source_url END,
    n.last_updated    = $last_updated
RETURN n
"""

# MERGE on (source_id, target_id, relation_type) to avoid duplicate edges
_MERGE_EDGE_CQL = """
MATCH (a {{id: $source_id}})
MATCH (b {{id: $target_id}})
MERGE (a)-[r:{rel_type}]->(b)
ON CREATE SET
    r.confidence     = $confidence,
    r.source_doc_id  = $source_doc_id,
    r.year           = $year,
    r.relation_type  = $relation_type
ON MATCH SET
    r.confidence     = CASE WHEN $confidence > r.confidence THEN $confidence ELSE r.confidence END,
    r.source_doc_id  = $source_doc_id
RETURN r
"""

# Multi-hop subgraph retrieval — returns nodes and edges up to `hops` away
_SUBGRAPH_CQL = """
MATCH path = (start)-[*1..{hops}]-(neighbor)
WHERE start.id IN $cui_list
WITH nodes(path) AS ns, relationships(path) AS rs
UNWIND ns AS n
WITH COLLECT(DISTINCT n) AS all_nodes, rs
UNWIND rs AS r
RETURN all_nodes,
       COLLECT(DISTINCT {{
           source_id:    startNode(r).id,
           target_id:    endNode(r).id,
           edge_type:    type(r),
           confidence:   r.confidence,
           source_doc_id: r.source_doc_id,
           year:         r.year,
           relation_type: r.relation_type
       }}) AS all_edges
"""

_GET_NODE_CQL = """
MATCH (n {id: $cui})
RETURN n
"""

_CREATE_INDEXES_CQL = [
    "CREATE INDEX node_id_index IF NOT EXISTS FOR (n:Disease)   ON (n.id)",
    "CREATE INDEX drug_id_index IF NOT EXISTS FOR (n:Drug)      ON (n.id)",
    "CREATE INDEX gene_id_index IF NOT EXISTS FOR (n:Gene)      ON (n.id)",
    "CREATE INDEX sym_id_index  IF NOT EXISTS FOR (n:Symptom)   ON (n.id)",
    "CREATE INDEX trial_id_idx  IF NOT EXISTS FOR (n:ClinicalTrial) ON (n.id)",
    "CREATE INDEX paper_id_idx  IF NOT EXISTS FOR (n:Paper)     ON (n.id)",
    "CREATE INDEX node_name_idx IF NOT EXISTS FOR (n:Disease)   ON (n.name)",
]


# ---------------------------------------------------------------------------
# GraphBuilder
# ---------------------------------------------------------------------------


class GraphBuilder:
    """
    Manages all reads and writes to the Neo4j knowledge graph.

    Parameters
    ----------
    uri:
        Bolt URI of the Neo4j instance, e.g. ``bolt://localhost:7687``.
    user:
        Neo4j username.
    password:
        Neo4j password.
    database:
        Neo4j database name (default ``neo4j``).

    Example
    -------
    >>> gb = GraphBuilder()
    >>> gb.create_indexes()
    >>> gb.upsert_batch(triples)
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: str = "neo4j",
    ) -> None:
        self._uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self._user = user or os.getenv("NEO4J_USER", "neo4j")
        self._password = password or os.getenv("NEO4J_PASSWORD", "medkg_password")
        self._database = database
        self._driver: Optional[Driver] = None
        self._connect()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        """Establish connection to Neo4j."""
        try:
            self._driver = GraphDatabase.driver(
                self._uri,
                auth=(self._user, self._password),
            )
            # Verify connectivity
            self._driver.verify_connectivity()
            logger.info("Connected to Neo4j at %s", self._uri)
        except ServiceUnavailable:
            logger.error(
                "Neo4j not reachable at %s — is the container running? "
                "Run: docker compose up -d neo4j",
                self._uri,
            )
            self._driver = None
        except AuthError:
            logger.error(
                "Neo4j authentication failed for user=%s — check NEO4J_PASSWORD in .env",
                self._user,
            )
            self._driver = None
        except Exception as exc:  # noqa: BLE001
            logger.error("Neo4j connection error: %s", exc)
            self._driver = None

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self._driver:
            self._driver.close()
            logger.info("Neo4j connection closed")

    def is_connected(self) -> bool:
        """Return True if the driver is connected and responsive."""
        if not self._driver:
            return False
        try:
            self._driver.verify_connectivity()
            return True
        except Exception:  # noqa: BLE001
            return False

    def _session(self) -> Session:
        """Return a new Neo4j session, raising if not connected."""
        if not self._driver:
            raise RuntimeError(
                "GraphBuilder is not connected to Neo4j. "
                "Check that the container is running and credentials are correct."
            )
        return self._driver.session(database=self._database)

    # ------------------------------------------------------------------
    # Schema / indexes
    # ------------------------------------------------------------------

    def create_indexes(self) -> None:
        """
        Create Neo4j indexes for all node labels on the ``id`` property.

        Safe to call multiple times — uses ``IF NOT EXISTS``.
        Should be called once on application startup.
        """
        if not self._driver:
            logger.warning("Skipping index creation — not connected to Neo4j")
            return

        with self._session() as session:
            for cql in _CREATE_INDEXES_CQL:
                try:
                    session.run(cql)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Index creation skipped: %s", exc)

        logger.info("Neo4j indexes created / verified")

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_node(
        self,
        entity: Entity,
        extra_props: Optional[Dict[str, Any]] = None,
    ) -> Optional[GraphNode]:
        """
        MERGE a single entity node into Neo4j.

        Uses the entity's CUI as the unique node id. If CUI is absent a
        hash-based synthetic id is generated from the entity text + type.

        Parameters
        ----------
        entity:
            The Entity to persist.
        extra_props:
            Optional additional properties to set on the node.

        Returns
        -------
        Optional[GraphNode]
            The persisted node, or None if the write failed.
        """
        if not self._driver:
            logger.warning("add_node skipped — not connected")
            return None

        node_id = self._entity_id(entity)
        label = entity.entity_type.value
        now = datetime.now(timezone.utc).isoformat()

        params: Dict[str, Any] = {
            "id": node_id,
            "name": entity.canonical_name or entity.text,
            "type": label,
            "source_url": None,
            "last_updated": now,
            "confidence_score": entity.confidence,
        }
        if extra_props:
            params.update(extra_props)

        cql = _MERGE_NODE_CQL.format(label=label)

        try:
            with self._session() as session:
                result = session.run(cql, **params)
                record = result.single()
                if record:
                    n = record["n"]
                    logger.debug("Upserted node id=%s label=%s", node_id, label)
                    return GraphNode(
                        id=n["id"],
                        name=n["name"],
                        node_type=entity.entity_type,
                        source_url=n.get("source_url"),
                        confidence_score=n.get("confidence_score", 1.0),
                    )
        except Exception as exc:  # noqa: BLE001
            logger.error("add_node failed for %r: %s", entity.text, exc)

        return None

    def get_node_by_cui(self, cui: str) -> Optional[GraphNode]:
        """
        Retrieve a single node by its UMLS CUI / id.

        Parameters
        ----------
        cui:
            The node id to look up.

        Returns
        -------
        Optional[GraphNode]
            The matching node, or None if not found.
        """
        if not self._driver:
            return None

        try:
            with self._session() as session:
                result = session.run(_GET_NODE_CQL, cui=cui)
                record = result.single()
                if record:
                    n = record["n"]
                    label = list(n.labels)[0] if n.labels else "Disease"
                    node_type = NodeType(label) if label in NodeType._value2member_map_ else NodeType.DISEASE
                    return GraphNode(
                        id=n["id"],
                        name=n.get("name", ""),
                        node_type=node_type,
                        source_url=n.get("source_url"),
                        confidence_score=n.get("confidence_score", 1.0),
                        last_updated=datetime.fromisoformat(n["last_updated"])
                        if n.get("last_updated")
                        else datetime.utcnow(),
                    )
        except Exception as exc:  # noqa: BLE001
            logger.error("get_node_by_cui failed for cui=%s: %s", cui, exc)

        return None

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_edge(
        self,
        source_entity: Entity,
        relation: EdgeType,
        target_entity: Entity,
        confidence: float = 0.5,
        source_doc_id: Optional[str] = None,
        year: Optional[int] = None,
    ) -> bool:
        """
        MERGE a single directed relationship between two entities.

        Both nodes must already exist (or will be created by add_node first).

        Parameters
        ----------
        source_entity:
            The subject Entity.
        relation:
            The EdgeType (e.g. ``EdgeType.TREATS``).
        target_entity:
            The object Entity.
        confidence:
            Confidence score for this relation (0–1).
        source_doc_id:
            ID of the document this relation was extracted from.
        year:
            Publication year of the source document.

        Returns
        -------
        bool
            True if the edge was written successfully.
        """
        if not self._driver:
            logger.warning("add_edge skipped — not connected")
            return False

        source_id = self._entity_id(source_entity)
        target_id = self._entity_id(target_entity)
        cql = _MERGE_EDGE_CQL.format(rel_type=relation.value)

        try:
            with self._session() as session:
                session.run(
                    cql,
                    source_id=source_id,
                    target_id=target_id,
                    confidence=confidence,
                    source_doc_id=source_doc_id,
                    year=year,
                    relation_type=relation.value,
                )
                logger.debug(
                    "Upserted edge %s -[%s]-> %s",
                    source_entity.text,
                    relation.value,
                    target_entity.text,
                )
                return True
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "add_edge failed %r -[%s]-> %r: %s",
                source_entity.text,
                relation.value,
                target_entity.text,
                exc,
            )
            return False

    # ------------------------------------------------------------------
    # Batch upsert
    # ------------------------------------------------------------------

    def upsert_batch(self, triples: List[Triple]) -> Tuple[int, int]:
        """
        Idempotent bulk write of a list of Triple objects.

        For each triple:
        1. MERGE the subject node
        2. MERGE the object node
        3. MERGE the relationship

        Parameters
        ----------
        triples:
            List of Triple objects from NLPProcessor.extract_relations().

        Returns
        -------
        Tuple[int, int]
            (nodes_written, edges_written) counts.
        """
        if not triples:
            return 0, 0

        nodes_written = 0
        edges_written = 0

        for triple in triples:
            # Upsert subject
            subj_node = self.add_node(triple.subject)
            if subj_node:
                nodes_written += 1

            # Upsert object
            obj_node = self.add_node(triple.obj)
            if obj_node:
                nodes_written += 1

            # Upsert edge (only if both nodes succeeded)
            if subj_node and obj_node:
                success = self.add_edge(
                    source_entity=triple.subject,
                    relation=triple.predicate,
                    target_entity=triple.obj,
                    confidence=triple.confidence,
                    source_doc_id=triple.source_doc_id,
                    year=triple.year,
                )
                if success:
                    edges_written += 1

        logger.info(
            "upsert_batch complete: %d nodes, %d edges written from %d triples",
            nodes_written,
            edges_written,
            len(triples),
        )
        return nodes_written, edges_written

    # ------------------------------------------------------------------
    # Subgraph retrieval
    # ------------------------------------------------------------------

    def get_subgraph(
        self,
        query_cuis: List[str],
        hops: int = 2,
    ) -> GraphSubgraph:
        """
        Retrieve a multi-hop subgraph centred on the given CUI list.

        Parameters
        ----------
        query_cuis:
            List of UMLS CUIs (node ids) to start traversal from.
        hops:
            Number of relationship hops to traverse (default 2).

        Returns
        -------
        GraphSubgraph
            Nodes, edges, and the product-of-confidences path score.
        """
        if not self._driver or not query_cuis:
            return GraphSubgraph()

        cql = _SUBGRAPH_CQL.format(hops=hops)

        try:
            with self._session() as session:
                result = session.run(cql, cui_list=query_cuis)
                record = result.single()

                if not record:
                    return GraphSubgraph(query_node_ids=query_cuis)

                raw_nodes = record["all_nodes"]
                raw_edges = record["all_edges"]

                nodes: List[GraphNode] = []
                for n in raw_nodes:
                    label = list(n.labels)[0] if n.labels else "Disease"
                    node_type = (
                        NodeType(label)
                        if label in NodeType._value2member_map_
                        else NodeType.DISEASE
                    )
                    nodes.append(
                        GraphNode(
                            id=n["id"],
                            name=n.get("name", ""),
                            node_type=node_type,
                            source_url=n.get("source_url"),
                            confidence_score=n.get("confidence_score", 1.0),
                            last_updated=datetime.fromisoformat(n["last_updated"])
                            if n.get("last_updated")
                            else datetime.utcnow(),
                        )
                    )

                edges: List[GraphEdge] = []
                path_confidence = 1.0
                for e in raw_edges:
                    edge_type_str = e.get("edge_type", "ASSOCIATED_WITH")
                    try:
                        edge_type = EdgeType(edge_type_str)
                    except ValueError:
                        edge_type = EdgeType.ASSOCIATED_WITH

                    conf = float(e.get("confidence") or 0.5)
                    path_confidence *= conf

                    edges.append(
                        GraphEdge(
                            source_id=e["source_id"],
                            target_id=e["target_id"],
                            edge_type=edge_type,
                            confidence=conf,
                            source_doc_id=e.get("source_doc_id"),
                            year=e.get("year"),
                            relation_type=e.get("relation_type"),
                        )
                    )

                logger.info(
                    "get_subgraph: %d nodes, %d edges, path_confidence=%.4f",
                    len(nodes),
                    len(edges),
                    path_confidence,
                )

                return GraphSubgraph(
                    nodes=nodes,
                    edges=edges,
                    query_node_ids=query_cuis,
                    path_confidence=round(path_confidence, 6),
                )

        except Exception as exc:  # noqa: BLE001
            logger.error("get_subgraph failed: %s", exc)
            return GraphSubgraph(query_node_ids=query_cuis)

    def search_nodes_by_name(
        self,
        name: str,
        node_type: Optional[NodeType] = None,
        limit: int = 10,
    ) -> List[GraphNode]:
        """
        Full-text search for nodes whose name contains *name*.

        Parameters
        ----------
        name:
            Partial or full node name to search for.
        node_type:
            Optional filter by node label.
        limit:
            Maximum number of nodes to return.

        Returns
        -------
        List[GraphNode]
            Matching nodes ordered by name.
        """
        if not self._driver:
            return []

        if node_type:
            cql = f"""
                MATCH (n:{node_type.value})
                WHERE toLower(n.name) CONTAINS toLower($name)
                RETURN n LIMIT $limit
            """
        else:
            cql = """
                MATCH (n)
                WHERE toLower(n.name) CONTAINS toLower($name)
                RETURN n LIMIT $limit
            """

        nodes: List[GraphNode] = []
        try:
            with self._session() as session:
                result = session.run(cql, name=name, limit=limit)
                for record in result:
                    n = record["n"]
                    label = list(n.labels)[0] if n.labels else "Disease"
                    node_type_enum = (
                        NodeType(label)
                        if label in NodeType._value2member_map_
                        else NodeType.DISEASE
                    )
                    nodes.append(
                        GraphNode(
                            id=n["id"],
                            name=n.get("name", ""),
                            node_type=node_type_enum,
                            source_url=n.get("source_url"),
                            confidence_score=n.get("confidence_score", 1.0),
                        )
                    )
        except Exception as exc:  # noqa: BLE001
            logger.error("search_nodes_by_name failed: %s", exc)

        return nodes

    def get_graph_stats(self) -> Dict[str, Any]:
        """
        Return basic graph statistics: node counts per label, edge counts.

        Returns
        -------
        Dict[str, Any]
            Dictionary with node/edge counts.
        """
        if not self._driver:
            return {}

        stats: Dict[str, Any] = {}
        try:
            with self._session() as session:
                # Node counts per label
                for label in NodeType:
                    result = session.run(
                        f"MATCH (n:{label.value}) RETURN count(n) AS cnt"
                    )
                    record = result.single()
                    stats[label.value] = record["cnt"] if record else 0

                # Total edge count
                result = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt")
                record = result.single()
                stats["total_edges"] = record["cnt"] if record else 0

        except Exception as exc:  # noqa: BLE001
            logger.error("get_graph_stats failed: %s", exc)

        return stats

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "GraphBuilder":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _entity_id(entity: Entity) -> str:
        """
        Return the canonical node id for an entity.

        Uses the UMLS CUI when available, otherwise falls back to a
        hash of (type, lowercase_text).
        """
        if entity.cui:
            return entity.cui

        import hashlib  # noqa: PLC0415
        canonical = f"{entity.entity_type.value}::{entity.text.lower().strip()}"
        return "SYN_" + hashlib.md5(canonical.encode()).hexdigest()[:12]
