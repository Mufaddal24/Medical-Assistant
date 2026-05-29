"""
Tests for Module 3 — GraphBuilder.

All Neo4j calls are mocked so these tests run without a live database.
Integration tests (marked @pytest.mark.integration) require:
  - docker compose up -d neo4j
  - RUN_INTEGRATION_TESTS=1 set in environment
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.utils.models import EdgeType, Entity, GraphSubgraph, NodeType, Triple


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def drug_entity() -> Entity:
    return Entity(
        text="Metformin",
        entity_type=NodeType.DRUG,
        cui="C0025598",
        start_char=0,
        end_char=9,
        confidence=0.95,
        canonical_name="Metformin",
    )


@pytest.fixture()
def disease_entity() -> Entity:
    return Entity(
        text="Type 2 Diabetes",
        entity_type=NodeType.DISEASE,
        cui="C0011860",
        start_char=20,
        end_char=35,
        confidence=0.98,
        canonical_name="Diabetes Mellitus, Type 2",
    )


@pytest.fixture()
def sample_triple(drug_entity: Entity, disease_entity: Entity) -> Triple:
    return Triple(
        subject=drug_entity,
        predicate=EdgeType.TREATS,
        obj=disease_entity,
        confidence=0.85,
        source_doc_id="doc_001",
        year=2023,
        evidence_text="Metformin is used to treat Type 2 Diabetes.",
    )


@pytest.fixture()
def mock_driver():
    """Return a MagicMock that mimics neo4j.Driver behaviour."""
    driver = MagicMock()
    driver.verify_connectivity.return_value = None

    # session() returns a context manager
    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    driver.session.return_value = mock_session

    return driver, mock_session


@pytest.fixture()
def graph_builder(mock_driver):
    """GraphBuilder with a mocked Neo4j driver."""
    from src.graph.builder import GraphBuilder

    driver, _ = mock_driver
    with patch("src.graph.builder.GraphDatabase.driver", return_value=driver):
        gb = GraphBuilder(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="test",
        )
    return gb


# ---------------------------------------------------------------------------
# Connection tests
# ---------------------------------------------------------------------------


class TestGraphBuilderConnection:

    def test_is_connected_true(self, graph_builder) -> None:
        """is_connected returns True when driver.verify_connectivity succeeds."""
        assert graph_builder.is_connected() is True

    def test_is_connected_false_when_no_driver(self) -> None:
        """is_connected returns False when driver is None."""
        from src.graph.builder import GraphBuilder

        with patch("src.graph.builder.GraphDatabase.driver", side_effect=Exception("fail")):
            gb = GraphBuilder()
        assert gb.is_connected() is False

    def test_close(self, graph_builder, mock_driver) -> None:
        """close() calls driver.close()."""
        driver, _ = mock_driver
        graph_builder.close()
        driver.close.assert_called_once()

    def test_context_manager(self, mock_driver) -> None:
        """GraphBuilder works as a context manager."""
        from src.graph.builder import GraphBuilder

        driver, _ = mock_driver
        with patch("src.graph.builder.GraphDatabase.driver", return_value=driver):
            with GraphBuilder() as gb:
                assert gb.is_connected()
        driver.close.assert_called_once()


# ---------------------------------------------------------------------------
# Node tests
# ---------------------------------------------------------------------------


class TestAddNode:

    def test_add_node_uses_cui_as_id(
        self, graph_builder, mock_driver, drug_entity: Entity
    ) -> None:
        """add_node uses the entity CUI as the node id."""
        _, session = mock_driver

        mock_result = MagicMock()
        mock_node = MagicMock()
        mock_node.__getitem__ = lambda self, key: {
            "id": "C0025598",
            "name": "Metformin",
            "source_url": None,
            "confidence_score": 0.95,
            "last_updated": datetime.utcnow().isoformat(),
        }[key]
        mock_node.get = lambda key, default=None: {
            "source_url": None,
            "confidence_score": 0.95,
            "last_updated": datetime.utcnow().isoformat(),
        }.get(key, default)
        mock_result.single.return_value = {"n": mock_node}
        session.run.return_value = mock_result

        node = graph_builder.add_node(drug_entity)

        # Verify session.run was called
        session.run.assert_called_once()
        call_kwargs = session.run.call_args

        # The CUI should appear as the id parameter
        assert "C0025598" in str(call_kwargs)

    def test_add_node_generates_synthetic_id_without_cui(
        self, graph_builder, mock_driver
    ) -> None:
        """add_node generates a SYN_ id when entity has no CUI."""
        _, session = mock_driver
        entity = Entity(
            text="Unknown Compound",
            entity_type=NodeType.DRUG,
            cui=None,
            start_char=0,
            end_char=16,
        )

        mock_result = MagicMock()
        mock_node = MagicMock()
        mock_node.__getitem__ = lambda self, key: {
            "id": "SYN_abc123",
            "name": "Unknown Compound",
            "source_url": None,
            "confidence_score": 1.0,
            "last_updated": datetime.utcnow().isoformat(),
        }[key]
        mock_node.get = lambda key, default=None: default
        mock_result.single.return_value = {"n": mock_node}
        session.run.return_value = mock_result

        graph_builder.add_node(entity)

        call_str = str(session.run.call_args)
        assert "SYN_" in call_str

    def test_add_node_returns_none_when_disconnected(
        self, drug_entity: Entity
    ) -> None:
        """add_node returns None gracefully when not connected."""
        from src.graph.builder import GraphBuilder

        with patch("src.graph.builder.GraphDatabase.driver", side_effect=Exception):
            gb = GraphBuilder()
        result = gb.add_node(drug_entity)
        assert result is None

    def test_add_node_returns_none_on_query_error(
        self, graph_builder, mock_driver, drug_entity: Entity
    ) -> None:
        """add_node returns None if the Cypher query raises."""
        _, session = mock_driver
        session.run.side_effect = Exception("Cypher error")

        result = graph_builder.add_node(drug_entity)
        assert result is None


# ---------------------------------------------------------------------------
# Edge tests
# ---------------------------------------------------------------------------


class TestAddEdge:

    def test_add_edge_returns_true_on_success(
        self,
        graph_builder,
        mock_driver,
        drug_entity: Entity,
        disease_entity: Entity,
    ) -> None:
        """add_edge returns True when the Cypher write succeeds."""
        _, session = mock_driver
        session.run.return_value = MagicMock()

        result = graph_builder.add_edge(
            source_entity=drug_entity,
            relation=EdgeType.TREATS,
            target_entity=disease_entity,
            confidence=0.85,
            source_doc_id="doc_001",
            year=2023,
        )
        assert result is True

    def test_add_edge_uses_correct_relation_type(
        self,
        graph_builder,
        mock_driver,
        drug_entity: Entity,
        disease_entity: Entity,
    ) -> None:
        """add_edge embeds the correct relationship type in the Cypher query."""
        _, session = mock_driver
        session.run.return_value = MagicMock()

        graph_builder.add_edge(
            source_entity=drug_entity,
            relation=EdgeType.TREATS,
            target_entity=disease_entity,
        )

        call_str = str(session.run.call_args)
        assert "TREATS" in call_str

    def test_add_edge_returns_false_on_error(
        self,
        graph_builder,
        mock_driver,
        drug_entity: Entity,
        disease_entity: Entity,
    ) -> None:
        """add_edge returns False if the query raises."""
        _, session = mock_driver
        session.run.side_effect = Exception("Node not found")

        result = graph_builder.add_edge(drug_entity, EdgeType.TREATS, disease_entity)
        assert result is False

    def test_add_edge_returns_false_when_disconnected(
        self, drug_entity: Entity, disease_entity: Entity
    ) -> None:
        """add_edge returns False gracefully when not connected."""
        from src.graph.builder import GraphBuilder

        with patch("src.graph.builder.GraphDatabase.driver", side_effect=Exception):
            gb = GraphBuilder()

        result = gb.add_edge(drug_entity, EdgeType.TREATS, disease_entity)
        assert result is False


# ---------------------------------------------------------------------------
# Batch upsert tests
# ---------------------------------------------------------------------------


class TestUpsertBatch:

    def test_upsert_batch_empty_list(self, graph_builder) -> None:
        """upsert_batch returns (0, 0) for an empty triple list."""
        nodes, edges = graph_builder.upsert_batch([])
        assert nodes == 0
        assert edges == 0

    def test_upsert_batch_calls_add_node_and_add_edge(
        self, graph_builder, sample_triple: Triple
    ) -> None:
        """upsert_batch calls add_node twice and add_edge once per triple."""
        from src.utils.models import GraphNode

        mock_node = GraphNode(
            id="C0025598",
            name="Metformin",
            node_type=NodeType.DRUG,
        )

        with patch.object(graph_builder, "add_node", return_value=mock_node) as mock_add_node:
            with patch.object(graph_builder, "add_edge", return_value=True) as mock_add_edge:
                nodes, edges = graph_builder.upsert_batch([sample_triple])

        # add_node called for subject and object
        assert mock_add_node.call_count == 2
        # add_edge called once
        assert mock_add_edge.call_count == 1
        assert nodes == 2
        assert edges == 1

    def test_upsert_batch_skips_edge_if_node_fails(
        self, graph_builder, sample_triple: Triple
    ) -> None:
        """upsert_batch does not write edge if a node write fails."""
        with patch.object(graph_builder, "add_node", return_value=None):
            with patch.object(graph_builder, "add_edge") as mock_add_edge:
                nodes, edges = graph_builder.upsert_batch([sample_triple])

        mock_add_edge.assert_not_called()
        assert edges == 0

    def test_upsert_batch_multiple_triples(
        self,
        graph_builder,
        drug_entity: Entity,
        disease_entity: Entity,
    ) -> None:
        """upsert_batch processes all triples in the list."""
        from src.utils.models import GraphNode

        triples = [
            Triple(subject=drug_entity, predicate=EdgeType.TREATS, obj=disease_entity, confidence=0.8),
            Triple(subject=drug_entity, predicate=EdgeType.CAUSES, obj=disease_entity, confidence=0.4),
        ]
        mock_node = GraphNode(id="X", name="X", node_type=NodeType.DRUG)

        with patch.object(graph_builder, "add_node", return_value=mock_node):
            with patch.object(graph_builder, "add_edge", return_value=True):
                nodes, edges = graph_builder.upsert_batch(triples)

        assert edges == 2


# ---------------------------------------------------------------------------
# Subgraph retrieval tests
# ---------------------------------------------------------------------------


class TestGetSubgraph:

    def test_get_subgraph_returns_empty_when_disconnected(self) -> None:
        """get_subgraph returns empty GraphSubgraph when not connected."""
        from src.graph.builder import GraphBuilder

        with patch("src.graph.builder.GraphDatabase.driver", side_effect=Exception):
            gb = GraphBuilder()

        result = gb.get_subgraph(["C0025598"])
        assert isinstance(result, GraphSubgraph)
        assert result.nodes == []
        assert result.edges == []

    def test_get_subgraph_returns_empty_for_no_cuis(
        self, graph_builder
    ) -> None:
        """get_subgraph returns empty GraphSubgraph for empty CUI list."""
        result = graph_builder.get_subgraph([])
        assert isinstance(result, GraphSubgraph)
        assert result.nodes == []

    def test_get_subgraph_path_confidence_product(
        self, graph_builder, mock_driver
    ) -> None:
        """get_subgraph computes path_confidence as product of edge confidences."""
        _, session = mock_driver

        mock_node = MagicMock()
        mock_node.__iter__ = MagicMock(return_value=iter(["Disease"]))
        mock_node.labels = {"Disease"}
        mock_node.__getitem__ = lambda self, key: {
            "id": "C0011860",
            "name": "Type 2 Diabetes",
            "source_url": None,
            "confidence_score": 1.0,
            "last_updated": datetime.utcnow().isoformat(),
        }[key]
        mock_node.get = lambda key, default=None: {
            "source_url": None,
            "confidence_score": 1.0,
            "last_updated": datetime.utcnow().isoformat(),
        }.get(key, default)

        mock_result = MagicMock()
        mock_result.single.return_value = {
            "all_nodes": [mock_node],
            "all_edges": [
                {
                    "source_id": "C0025598",
                    "target_id": "C0011860",
                    "edge_type": "TREATS",
                    "confidence": 0.8,
                    "source_doc_id": "doc_1",
                    "year": 2023,
                    "relation_type": "TREATS",
                },
                {
                    "source_id": "C0011860",
                    "target_id": "C0099999",
                    "edge_type": "CAUSES",
                    "confidence": 0.5,
                    "source_doc_id": "doc_2",
                    "year": 2022,
                    "relation_type": "CAUSES",
                },
            ],
        }
        session.run.return_value = mock_result

        subgraph = graph_builder.get_subgraph(["C0025598"], hops=2)

        # 0.8 * 0.5 = 0.4
        assert abs(subgraph.path_confidence - 0.4) < 1e-6
        assert len(subgraph.edges) == 2


# ---------------------------------------------------------------------------
# Helper / stats tests
# ---------------------------------------------------------------------------


class TestHelpers:

    def test_entity_id_uses_cui(self, drug_entity: Entity) -> None:
        """_entity_id returns the CUI when present."""
        from src.graph.builder import GraphBuilder

        assert GraphBuilder._entity_id(drug_entity) == "C0025598"

    def test_entity_id_generates_syn_without_cui(self) -> None:
        """_entity_id returns a SYN_ prefixed hash when CUI is absent."""
        from src.graph.builder import GraphBuilder

        entity = Entity(
            text="Unknown Drug",
            entity_type=NodeType.DRUG,
            cui=None,
            start_char=0,
            end_char=12,
        )
        result = GraphBuilder._entity_id(entity)
        assert result.startswith("SYN_")

    def test_entity_id_deterministic(self) -> None:
        """_entity_id returns the same id for the same input."""
        from src.graph.builder import GraphBuilder

        entity = Entity(
            text="Metformin",
            entity_type=NodeType.DRUG,
            cui=None,
            start_char=0,
            end_char=9,
        )
        assert GraphBuilder._entity_id(entity) == GraphBuilder._entity_id(entity)

    def test_get_graph_stats_returns_dict(
        self, graph_builder, mock_driver
    ) -> None:
        """get_graph_stats returns a dictionary with node/edge counts."""
        _, session = mock_driver

        mock_result = MagicMock()
        mock_result.single.return_value = {"cnt": 5}
        session.run.return_value = mock_result

        stats = graph_builder.get_graph_stats()
        assert isinstance(stats, dict)
        assert "total_edges" in stats

    def test_create_indexes_runs_without_error(
        self, graph_builder, mock_driver
    ) -> None:
        """create_indexes runs all index statements without raising."""
        _, session = mock_driver
        session.run.return_value = MagicMock()

        graph_builder.create_indexes()
        assert session.run.call_count >= 1


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

INTEGRATION = pytest.mark.skipif(
    __import__("os").getenv("RUN_INTEGRATION_TESTS") != "1",
    reason="Set RUN_INTEGRATION_TESTS=1 and run: docker compose up -d neo4j",
)


@INTEGRATION
def test_neo4j_live_round_trip() -> None:
    """Write a node and edge to real Neo4j, read it back, then clean up."""
    from src.graph.builder import GraphBuilder

    gb = GraphBuilder()
    assert gb.is_connected(), "Neo4j not reachable — is docker running?"

    gb.create_indexes()

    drug = Entity(
        text="Metformin",
        entity_type=NodeType.DRUG,
        cui="TEST_C0025598",
        start_char=0,
        end_char=9,
        confidence=0.95,
        canonical_name="Metformin",
    )
    disease = Entity(
        text="Type 2 Diabetes",
        entity_type=NodeType.DISEASE,
        cui="TEST_C0011860",
        start_char=0,
        end_char=15,
        confidence=0.98,
        canonical_name="Diabetes Mellitus, Type 2",
    )
    triple = Triple(
        subject=drug,
        predicate=EdgeType.TREATS,
        obj=disease,
        confidence=0.9,
        source_doc_id="integration_test",
        year=2024,
    )

    nodes, edges = gb.upsert_batch([triple])
    assert nodes == 2
    assert edges == 1

    # Read back
    node = gb.get_node_by_cui("TEST_C0025598")
    assert node is not None
    assert node.name == "Metformin"

    # Cleanup
    with gb._session() as session:
        session.run("MATCH (n {id: 'TEST_C0025598'}) DETACH DELETE n")
        session.run("MATCH (n {id: 'TEST_C0011860'}) DETACH DELETE n")

    gb.close()
