"""
Tests for Module 8 — GraphVisualizer (src/graph/visualizer.py)

All tests are pure unit tests — no live Neo4j required.
pyvis IS installed in this project (pyvis>=0.3.2), so the HTML generation
tests use the real pyvis library. The only thing mocked is file I/O where
we don't want to create files on disk during normal test runs.

Compatibility notes (real models.py)
--------------------------------------
- NodeType members: DISEASE, DRUG, GENE, SYMPTOM, CLINICAL_TRIAL, PAPER (UPPERCASE)
- EdgeType members: TREATS, CAUSES, ... (UPPERCASE)
- GraphNode.confidence_score: float (default 1.0)
- GraphNode.properties: Dict[str, Any] (default {})
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from src.graph.visualizer import (
    HIGHLIGHT_BORDER,
    NODE_COLORS,
    GraphVisualizer,
    _inject_legend,
    _LEGEND_HTML,
    build_edge_tooltip,
    build_node_tooltip,
    get_node_color,
)
from src.utils.models import (
    EdgeType,
    GraphEdge,
    GraphNode,
    GraphSubgraph,
    NodeType,
)


# ---------------------------------------------------------------------------
# Dynamic enum resolution
# ---------------------------------------------------------------------------

def _nt(value: str) -> NodeType:
    for m in NodeType:
        if m.value.lower() == value.lower():
            return m
    return list(NodeType)[0]


def _et(value: str) -> EdgeType:
    for m in EdgeType:
        if m.value.lower() == value.lower():
            return m
    return list(EdgeType)[0]


_DRUG     = _nt("Drug")
_DISEASE  = _nt("Disease")
_GENE     = _nt("Gene")
_SYMPTOM  = _nt("Symptom")
_CLINICAL = _nt("ClinicalTrial")
_PAPER    = _nt("Paper")
_TREATS   = _et("TREATS")
_CAUSES   = _et("CAUSES")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def viz() -> GraphVisualizer:
    return GraphVisualizer()


def _node(
    node_id: str,
    name: str,
    node_type: Optional[NodeType] = None,
    source_url: Optional[str] = None,
    confidence: float = 0.9,
) -> GraphNode:
    return GraphNode(
        id=node_id,
        name=name,
        node_type=node_type or _DRUG,
        source_url=source_url,
        last_updated=datetime.now(tz=timezone.utc),
        confidence_score=confidence,
    )


def _edge(
    src: str,
    tgt: str,
    edge_type: Optional[EdgeType] = None,
    confidence: float = 0.8,
    year: Optional[int] = 2023,
    doc_id: Optional[str] = "doc_001",
) -> GraphEdge:
    return GraphEdge(
        source_id=src,
        target_id=tgt,
        edge_type=edge_type or _TREATS,
        confidence=confidence,
        source_doc_id=doc_id,
        year=year,
    )


def _make_subgraph(
    highlight_ids: Optional[List[str]] = None,
) -> GraphSubgraph:
    drug    = _node("C001", "Metformin",     _DRUG,    "https://pubmed.ncbi.nlm.nih.gov/1/")
    disease = _node("C002", "Type 2 Diabetes", _DISEASE)
    gene    = _node("C003", "TCF7L2",        _GENE)
    e1 = _edge("C001", "C002", _TREATS)
    e2 = _edge("C003", "C002", _et("ASSOCIATED_WITH"), confidence=0.5)
    return GraphSubgraph(
        nodes=[drug, disease, gene],
        edges=[e1, e2],
        query_node_ids=highlight_ids or ["C002"],
        path_confidence=0.72,
    )


# ---------------------------------------------------------------------------
# 1. Constructor
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_output_path(self, viz: GraphVisualizer) -> None:
        assert viz.output_path == "graph_visualization.html"

    def test_default_height(self, viz: GraphVisualizer) -> None:
        assert viz.height == "700px"

    def test_default_width(self, viz: GraphVisualizer) -> None:
        assert viz.width == "100%"

    def test_default_bgcolor(self, viz: GraphVisualizer) -> None:
        assert viz.bgcolor == "#f8f9fa"

    def test_custom_params(self) -> None:
        v = GraphVisualizer(
            output_path="out/graph.html",
            height="500px",
            width="80%",
            bgcolor="#ffffff",
        )
        assert v.output_path == "out/graph.html"
        assert v.height == "500px"
        assert v.width == "80%"
        assert v.bgcolor == "#ffffff"


# ---------------------------------------------------------------------------
# 2. get_node_color (module-level helper)
# ---------------------------------------------------------------------------

class TestGetNodeColor:
    def test_drug_normal_background(self) -> None:
        c = get_node_color("Drug", False)
        assert c["background"] == "#3498db"

    def test_drug_normal_border(self) -> None:
        c = get_node_color("Drug", False)
        assert c["border"] == "#2980b9"

    def test_drug_highlight_background_is_brighter(self) -> None:
        normal  = get_node_color("Drug", False)["background"]
        bright  = get_node_color("Drug", True)["background"]
        assert normal != bright

    def test_highlight_border_is_gold(self) -> None:
        c = get_node_color("Drug", True)
        assert c["border"] == HIGHLIGHT_BORDER

    def test_all_six_types_have_distinct_normal_colors(self) -> None:
        types = ["Disease", "Drug", "Gene", "Symptom", "ClinicalTrial", "Paper"]
        colors = [get_node_color(t, False)["background"] for t in types]
        assert len(set(colors)) == 6, "Each NodeType must have a unique base colour"

    def test_all_six_types_have_distinct_highlight_colors(self) -> None:
        types = ["Disease", "Drug", "Gene", "Symptom", "ClinicalTrial", "Paper"]
        colors = [get_node_color(t, True)["background"] for t in types]
        assert len(set(colors)) == 6

    def test_disease_is_red_family(self) -> None:
        c = get_node_color("Disease", False)
        assert c["background"].startswith("#e") or c["background"].startswith("#c"), \
            f"Disease colour should be red-family, got {c['background']}"

    def test_unknown_type_returns_default(self) -> None:
        c = get_node_color("UnknownType", False)
        assert "background" in c
        assert "border" in c

    def test_highlight_sub_keys_present(self) -> None:
        c = get_node_color("Gene", True)
        assert "highlight" in c
        assert "hover" in c
        assert "background" in c["highlight"]
        assert "border" in c["highlight"]

    @pytest.mark.parametrize("node_type", ["Disease", "Drug", "Gene",
                                            "Symptom", "ClinicalTrial", "Paper"])
    def test_node_type_keys_match_enum_values(self, node_type: str) -> None:
        """Every NodeType value must be in NODE_COLORS."""
        assert node_type in NODE_COLORS


# ---------------------------------------------------------------------------
# 3. build_node_tooltip (module-level helper)
# ---------------------------------------------------------------------------

class TestBuildNodeTooltip:
    def test_contains_node_name(self) -> None:
        n = _node("C001", "Metformin", _DRUG, "https://example.com")
        tip = build_node_tooltip(n, False)
        assert "Metformin" in tip

    def test_contains_node_type(self) -> None:
        n = _node("C001", "Metformin", _DRUG)
        tip = build_node_tooltip(n, False)
        assert _DRUG.value in tip

    def test_contains_node_id(self) -> None:
        n = _node("C0025598", "Metformin", _DRUG)
        tip = build_node_tooltip(n, False)
        assert "C0025598" in tip

    def test_contains_source_url(self) -> None:
        n = _node("C001", "Metformin", _DRUG, "https://pubmed.ncbi.nlm.nih.gov/1/")
        tip = build_node_tooltip(n, False)
        assert "pubmed" in tip.lower()

    def test_no_url_shows_fallback(self) -> None:
        n = _node("C001", "Metformin", _DRUG, source_url=None)
        tip = build_node_tooltip(n, False)
        assert "No source URL" in tip or "source" in tip.lower()

    def test_highlighted_contains_answer_path_marker(self) -> None:
        n = _node("C001", "Metformin", _DRUG)
        tip = build_node_tooltip(n, True)
        assert "Answer path" in tip

    def test_not_highlighted_has_no_star_marker(self) -> None:
        n = _node("C001", "Metformin", _DRUG)
        tip = build_node_tooltip(n, False)
        assert "Answer path" not in tip

    def test_contains_confidence(self) -> None:
        n = _node("C001", "Metformin", _DRUG, confidence=0.87)
        tip = build_node_tooltip(n, False)
        assert "0.87" in tip

    def test_returns_html_string(self) -> None:
        n = _node("C001", "Metformin", _DRUG)
        tip = build_node_tooltip(n, False)
        assert "<div" in tip and "</div>" in tip


# ---------------------------------------------------------------------------
# 4. build_edge_tooltip (module-level helper)
# ---------------------------------------------------------------------------

class TestBuildEdgeTooltip:
    def test_contains_edge_type(self) -> None:
        e = _edge("C001", "C002", _TREATS)
        tip = build_edge_tooltip(e, "Metformin", "Diabetes")
        assert "TREATS" in tip

    def test_contains_node_names(self) -> None:
        e = _edge("C001", "C002", _TREATS)
        tip = build_edge_tooltip(e, "Metformin", "Diabetes")
        assert "Metformin" in tip
        assert "Diabetes" in tip

    def test_contains_confidence(self) -> None:
        e = _edge("C001", "C002", confidence=0.65)
        tip = build_edge_tooltip(e, "A", "B")
        assert "0.65" in tip

    def test_contains_year(self) -> None:
        e = _edge("C001", "C002", year=2023)
        tip = build_edge_tooltip(e, "A", "B")
        assert "2023" in tip

    def test_no_year_no_year_in_tip(self) -> None:
        e = _edge("C001", "C002", year=None)
        tip = build_edge_tooltip(e, "A", "B")
        assert "2023" not in tip

    def test_contains_doc_id(self) -> None:
        e = _edge("C001", "C002", doc_id="doc_abc")
        tip = build_edge_tooltip(e, "A", "B")
        assert "doc_abc" in tip

    def test_returns_html_string(self) -> None:
        e = _edge("C001", "C002")
        tip = build_edge_tooltip(e, "A", "B")
        assert "<div" in tip


# ---------------------------------------------------------------------------
# 5. _inject_legend
# ---------------------------------------------------------------------------

class TestInjectLegend:
    def test_appends_before_body_close(self) -> None:
        html = "<html><body><p>test</p></body></html>"
        result = _inject_legend(html)
        assert "kg-legend" in result
        assert result.endswith("</body></html>")

    def test_appends_at_end_when_no_body_tag(self) -> None:
        html = "<p>no body tag</p>"
        result = _inject_legend(html)
        assert "kg-legend" in result

    def test_legend_contains_color_entries(self) -> None:
        assert "Disease" in _LEGEND_HTML
        assert "Drug" in _LEGEND_HTML
        assert "Gene" in _LEGEND_HTML

    def test_legend_contains_answer_path_note(self) -> None:
        assert "Answer path" in _LEGEND_HTML


# ---------------------------------------------------------------------------
# 6. build_pyvis_graph — full HTML generation
# ---------------------------------------------------------------------------

class TestBuildPyvisGraph:
    def test_returns_string(self, viz: GraphVisualizer) -> None:
        html = viz.build_pyvis_graph(_make_subgraph())
        assert isinstance(html, str)

    def test_html_starts_with_html_tag(self, viz: GraphVisualizer) -> None:
        html = viz.build_pyvis_graph(_make_subgraph())
        assert html.strip().startswith("<html>")

    def test_contains_node_names(self, viz: GraphVisualizer) -> None:
        html = viz.build_pyvis_graph(_make_subgraph())
        assert "Metformin" in html
        assert "Type 2 Diabetes" in html
        assert "TCF7L2" in html

    def test_contains_edge_label(self, viz: GraphVisualizer) -> None:
        html = viz.build_pyvis_graph(_make_subgraph())
        assert "TREATS" in html

    def test_contains_legend(self, viz: GraphVisualizer) -> None:
        html = viz.build_pyvis_graph(_make_subgraph())
        assert "kg-legend" in html

    def test_is_self_contained(self, viz: GraphVisualizer) -> None:
        """vis.js should be embedded inline (not just CDN links)."""
        html = viz.build_pyvis_graph(_make_subgraph())
        # vis.js creates the network via `new vis.Network`
        assert "vis.Network" in html or "visjs" in html.lower()

    def test_empty_subgraph_returns_valid_html(self, viz: GraphVisualizer) -> None:
        empty = GraphSubgraph()
        html = viz.build_pyvis_graph(empty)
        assert isinstance(html, str)
        assert "<html" in html.lower()

    def test_empty_subgraph_contains_no_data_message(self, viz: GraphVisualizer) -> None:
        html = viz.build_pyvis_graph(GraphSubgraph())
        assert "No graph data" in html

    def test_none_highlight_does_not_raise(self, viz: GraphVisualizer) -> None:
        html = viz.build_pyvis_graph(_make_subgraph(), highlight_node_ids=None)
        assert "<html" in html.lower()

    def test_empty_highlight_list_does_not_raise(self, viz: GraphVisualizer) -> None:
        html = viz.build_pyvis_graph(_make_subgraph(), highlight_node_ids=[])
        assert "<html" in html.lower()

    def test_highlighted_nodes_trigger_distinct_rendering(self, viz: GraphVisualizer) -> None:
        """HTML with highlights should differ from HTML without."""
        html_plain = viz.build_pyvis_graph(_make_subgraph(), highlight_node_ids=[])
        html_hl    = viz.build_pyvis_graph(_make_subgraph(), highlight_node_ids=["C002"])
        assert html_plain != html_hl

    def test_highlight_answer_path_marker_present(self, viz: GraphVisualizer) -> None:
        html = viz.build_pyvis_graph(_make_subgraph(), highlight_node_ids=["C002"])
        assert "Answer path" in html

    def test_dangling_edge_skipped_gracefully(self, viz: GraphVisualizer) -> None:
        """Edge referencing a node not in the subgraph must not raise."""
        n = _node("C001", "Metformin", _DRUG)
        e_valid   = _edge("C001", "C001")   # self-loop is fine
        e_dangling = _edge("C001", "C999")  # C999 not in subgraph
        sg = GraphSubgraph(nodes=[n], edges=[e_valid, e_dangling])
        html = viz.build_pyvis_graph(sg, highlight_node_ids=[])
        assert isinstance(html, str)

    def test_single_node_no_edges(self, viz: GraphVisualizer) -> None:
        n = _node("C001", "Metformin", _DRUG)
        sg = GraphSubgraph(nodes=[n], edges=[])
        html = viz.build_pyvis_graph(sg)
        assert "Metformin" in html

    def test_all_node_types_render(self, viz: GraphVisualizer) -> None:
        nodes = [
            _node("N1", "Disease Node",   _DISEASE),
            _node("N2", "Drug Node",      _DRUG),
            _node("N3", "Gene Node",      _GENE),
            _node("N4", "Symptom Node",   _SYMPTOM),
            _node("N5", "Trial Node",     _CLINICAL),
            _node("N6", "Paper Node",     _PAPER),
        ]
        sg = GraphSubgraph(nodes=nodes, edges=[])
        html = viz.build_pyvis_graph(sg)
        for node in nodes:
            assert node.name in html


# ---------------------------------------------------------------------------
# 7. save()
# ---------------------------------------------------------------------------

class TestSave:
    def test_creates_file(self, viz: GraphVisualizer) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_graph.html")
            html = "<html><body>test</body></html>"
            returned = viz.save(html, path)
            assert os.path.isfile(returned)

    def test_returns_absolute_path(self, viz: GraphVisualizer) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.html")
            returned = viz.save("<html></html>", path)
            assert os.path.isabs(returned)

    def test_file_content_matches_input(self, viz: GraphVisualizer) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.html")
            html = "<html><body>content here</body></html>"
            viz.save(html, path)
            with open(path, encoding="utf-8") as fh:
                assert fh.read() == html

    def test_creates_parent_directories(self, viz: GraphVisualizer) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "a", "b", "c", "graph.html")
            viz.save("<html></html>", nested)
            assert os.path.isfile(nested)

    def test_default_path_used_when_none_given(self, viz: GraphVisualizer) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            viz.output_path = os.path.join(tmpdir, "default.html")
            returned = viz.save("<html></html>")
            assert "default.html" in returned


# ---------------------------------------------------------------------------
# 8. build_and_save()
# ---------------------------------------------------------------------------

class TestBuildAndSave:
    def test_returns_absolute_path(self, viz: GraphVisualizer) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.html")
            returned = viz.build_and_save(_make_subgraph(), path=path)
            assert os.path.isabs(returned)

    def test_file_is_created(self, viz: GraphVisualizer) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.html")
            viz.build_and_save(_make_subgraph(), path=path)
            assert os.path.isfile(path)

    def test_file_content_is_html(self, viz: GraphVisualizer) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.html")
            viz.build_and_save(_make_subgraph(), path=path)
            with open(path, encoding="utf-8") as fh:
                content = fh.read()
            assert content.strip().startswith("<html>")

    def test_highlights_propagated_to_file(self, viz: GraphVisualizer) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.html")
            viz.build_and_save(_make_subgraph(), highlight_node_ids=["C002"], path=path)
            with open(path, encoding="utf-8") as fh:
                content = fh.read()
            assert "Answer path" in content

    def test_empty_subgraph_still_creates_file(self, viz: GraphVisualizer) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "empty.html")
            viz.build_and_save(GraphSubgraph(), path=path)
            assert os.path.isfile(path)


# ---------------------------------------------------------------------------
# 9. Fallback when pyvis is not installed
# ---------------------------------------------------------------------------

class TestFallback:
    def test_fallback_html_contains_node_names(self, viz: GraphVisualizer) -> None:
        sg = _make_subgraph()
        html = viz._fallback_html(sg, {"C002"})
        assert "Metformin" in html
        assert "Type 2 Diabetes" in html

    def test_fallback_html_marks_highlighted(self, viz: GraphVisualizer) -> None:
        sg = _make_subgraph()
        html = viz._fallback_html(sg, {"C002"})
        assert "★" in html

    def test_fallback_html_contains_edge_types(self, viz: GraphVisualizer) -> None:
        sg = _make_subgraph()
        html = viz._fallback_html(sg, set())
        assert "TREATS" in html

    def test_build_pyvis_graph_uses_fallback_without_pyvis(self, viz: GraphVisualizer) -> None:
        sg = _make_subgraph()
        with patch("src.graph.visualizer.PYVIS_AVAILABLE", False):
            html = viz.build_pyvis_graph(sg, highlight_node_ids=["C002"])
        assert "Metformin" in html
        assert isinstance(html, str)

    def test_empty_fallback_html_is_valid(self, viz: GraphVisualizer) -> None:
        html = viz._fallback_html(GraphSubgraph(), set())
        assert "<html" in html
        assert "</html>" in html


# ---------------------------------------------------------------------------
# 10. _empty_html
# ---------------------------------------------------------------------------

class TestEmptyHtml:
    def test_returns_valid_html(self, viz: GraphVisualizer) -> None:
        html = viz._empty_html()
        assert "<html" in html
        assert "</html>" in html

    def test_contains_no_data_message(self, viz: GraphVisualizer) -> None:
        html = viz._empty_html()
        assert "No graph data" in html

    def test_contains_legend(self, viz: GraphVisualizer) -> None:
        html = viz._empty_html()
        assert "kg-legend" in html
