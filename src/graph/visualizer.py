"""
Module 8 — GraphVisualizer
Converts a GraphSubgraph into a self-contained interactive HTML file using pyvis.

Responsibilities
----------------
* build_pyvis_graph(subgraph, highlight_node_ids) → self-contained HTML string
* save(html, path)                               → write HTML to disk, return path
* build_and_save(subgraph, highlight_node_ids, path) → convenience wrapper

Color map (spec)
----------------
  Disease        → red     (#e74c3c  /  highlight: #ff7675)
  Drug           → blue    (#3498db  /  highlight: #74b9ff)
  Gene           → green   (#2ecc71  /  highlight: #55efc4)
  Symptom        → orange  (#e67e22  /  highlight: #fdcb6e)
  ClinicalTrial  → purple  (#9b59b6  /  highlight: #a29bfe)
  Paper          → gray    (#95a5a6  /  highlight: #dfe6e9)

Nodes in highlight_node_ids (answer path) use the brighter colour variant and
a gold (#f1c40f) border with increased thickness. Edges where BOTH endpoints
are highlighted are drawn thicker and darker.

Output is a single self-contained HTML string — vis.js is embedded inline
(cdn_resources='in_line') so the file works offline in any browser.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple

from src.utils.models import GraphEdge, GraphNode, GraphSubgraph

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional pyvis import
# ---------------------------------------------------------------------------

try:
    from pyvis.network import Network as PyvisNetwork
    PYVIS_AVAILABLE = True
except ImportError:  # pragma: no cover
    PyvisNetwork = None  # type: ignore[assignment,misc]
    PYVIS_AVAILABLE = False
    logger.warning("pyvis not installed — GraphVisualizer will return minimal HTML")

# ---------------------------------------------------------------------------
# Colour constants
# Tuple layout: (normal_bg, normal_border, highlight_bg, highlight_border)
# ---------------------------------------------------------------------------

NODE_COLORS: Dict[str, Tuple[str, str, str, str]] = {
    "Disease":       ("#e74c3c", "#c0392b", "#ff7675", "#d63031"),
    "Drug":          ("#3498db", "#2980b9", "#74b9ff", "#0984e3"),
    "Gene":          ("#2ecc71", "#27ae60", "#55efc4", "#00b894"),
    "Symptom":       ("#e67e22", "#d35400", "#fdcb6e", "#e17055"),
    "ClinicalTrial": ("#9b59b6", "#8e44ad", "#a29bfe", "#6c5ce7"),
    "Paper":         ("#95a5a6", "#7f8c8d", "#dfe6e9", "#b2bec3"),
}
_DEFAULT_COLOR: Tuple[str, str, str, str] = ("#bdc3c7", "#95a5a6", "#ecf0f1", "#bdc3c7")

HIGHLIGHT_BORDER = "#f1c40f"   # gold — marks answer-path nodes
_FONT_COLOR = "#2d3436"

_EDGE_COLOR_NORMAL    = {"color": "#636e72", "highlight": "#2d3436", "hover": "#2d3436"}
_EDGE_COLOR_HIGHLIGHT = {"color": "#2d3436", "highlight": "#000000", "hover": "#000000"}

_NODE_SIZE_NORMAL    = 22
_NODE_SIZE_HIGHLIGHT = 32
_BORDER_WIDTH_NORMAL    = 2
_BORDER_WIDTH_HIGHLIGHT = 5
_EDGE_WIDTH_NORMAL    = 2
_EDGE_WIDTH_HIGHLIGHT = 5

# ---------------------------------------------------------------------------
# Physics / layout options (vis.js JSON)
# ---------------------------------------------------------------------------

_PHYSICS_OPTIONS = """
{
  "nodes": {
    "font": {"size": 13, "face": "Arial", "color": "#2d3436"},
    "shadow": {"enabled": true, "size": 4, "x": 2, "y": 2}
  },
  "edges": {
    "font": {"size": 10, "face": "Arial", "align": "middle", "color": "#636e72"},
    "smooth": {"type": "curvedCW", "roundness": 0.15},
    "shadow": {"enabled": true, "size": 3, "x": 1, "y": 1}
  },
  "physics": {
    "enabled": true,
    "stabilization": {"iterations": 150, "updateInterval": 25},
    "barnesHut": {
      "gravitationalConstant": -9000,
      "centralGravity": 0.3,
      "springLength": 180,
      "springConstant": 0.04,
      "damping": 0.09
    }
  },
  "interaction": {
    "hover": true,
    "tooltipDelay": 150,
    "navigationButtons": true,
    "keyboard": {"enabled": true}
  }
}
"""

# ---------------------------------------------------------------------------
# Legend HTML — injected before </body>
# ---------------------------------------------------------------------------

_LEGEND_HTML = """
<div id="kg-legend" style="
    position:fixed;bottom:18px;right:18px;
    background:rgba(255,255,255,0.96);
    padding:14px 18px;
    border:1px solid #dfe6e9;
    border-radius:10px;
    font-family:Arial,sans-serif;
    font-size:12px;
    color:#2d3436;
    z-index:9999;
    box-shadow:0 3px 12px rgba(0,0,0,0.15);
    line-height:1.9;
">
  <div style="font-weight:bold;margin-bottom:6px;font-size:13px;">Node Types</div>
  <div><span style="color:#e74c3c;font-size:18px;">&#9679;</span>&nbsp;Disease</div>
  <div><span style="color:#3498db;font-size:18px;">&#9679;</span>&nbsp;Drug</div>
  <div><span style="color:#2ecc71;font-size:18px;">&#9679;</span>&nbsp;Gene</div>
  <div><span style="color:#e67e22;font-size:18px;">&#9679;</span>&nbsp;Symptom</div>
  <div><span style="color:#9b59b6;font-size:18px;">&#9679;</span>&nbsp;Clinical Trial</div>
  <div><span style="color:#95a5a6;font-size:18px;">&#9679;</span>&nbsp;Paper</div>
  <div style="margin-top:8px;border-top:1px solid #dfe6e9;padding-top:8px;">
    <span style="display:inline-block;width:13px;height:13px;
      background:#f1c40f;border-radius:50%;vertical-align:middle;
      margin-right:5px;"></span><b>Answer path</b>
  </div>
  <div style="margin-top:5px;font-size:11px;color:#636e72;">
    Scroll / drag to navigate<br>Hover nodes for details
  </div>
</div>
"""

# ---------------------------------------------------------------------------
# GraphVisualizer
# ---------------------------------------------------------------------------


class GraphVisualizer:
    """Convert a Neo4j GraphSubgraph into a self-contained interactive HTML.

    Parameters
    ----------
    output_path:
        Default file path used by :meth:`save` when no explicit path is given.
    height:
        CSS height of the vis.js canvas (default ``"700px"``).
    width:
        CSS width of the vis.js canvas (default ``"100%"``).
    bgcolor:
        Canvas background colour (default ``"#f8f9fa"``).

    Example
    -------
    >>> viz = GraphVisualizer()
    >>> html = viz.build_pyvis_graph(subgraph, highlight_node_ids=["C0025598"])
    >>> path = viz.save(html)         # writes graph_visualization.html
    """

    def __init__(
        self,
        output_path: str = "graph_visualization.html",
        height: str = "700px",
        width: str = "100%",
        bgcolor: str = "#f8f9fa",
    ) -> None:
        self.output_path = output_path
        self.height = height
        self.width = width
        self.bgcolor = bgcolor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_pyvis_graph(
        self,
        subgraph: GraphSubgraph,
        highlight_node_ids: Optional[List[str]] = None,
    ) -> str:
        """Build a self-contained pyvis HTML string from *subgraph*.

        Nodes whose ``id`` appears in *highlight_node_ids* are rendered with
        brighter colours and a bold gold border. Edges where BOTH endpoints
        are highlighted are drawn thicker and darker to mark the answer path.

        Parameters
        ----------
        subgraph:
            The ``GraphSubgraph`` to visualise — may be empty.
        highlight_node_ids:
            Node IDs that appear in the answer path. Pass ``[]`` or ``None``
            to render all nodes in normal style.

        Returns
        -------
        str
            Self-contained HTML string with vis.js embedded inline (~700 KB).
            Write directly to ``graph_visualization.html`` and open in any
            browser without an internet connection.
        """
        highlighted: Set[str] = set(highlight_node_ids or [])

        if not PYVIS_AVAILABLE:
            logger.warning("pyvis not installed — returning text-only fallback HTML")
            return self._fallback_html(subgraph, highlighted)

        if not subgraph.nodes:
            logger.info("GraphVisualizer: empty subgraph — returning minimal HTML")
            return self._empty_html()

        net = self._create_network()
        node_ids_in_graph: Set[str] = {n.id for n in subgraph.nodes}
        id_to_name: Dict[str, str] = {n.id: n.name for n in subgraph.nodes}

        # --- Nodes ---
        for node in subgraph.nodes:
            is_hl = node.id in highlighted
            net.add_node(
                node.id,
                label=node.name,
                color=get_node_color(node.node_type.value, is_hl),
                title=build_node_tooltip(node, is_hl),
                size=_NODE_SIZE_HIGHLIGHT if is_hl else _NODE_SIZE_NORMAL,
                borderWidth=_BORDER_WIDTH_HIGHLIGHT if is_hl else _BORDER_WIDTH_NORMAL,
                borderWidthSelected=(
                    _BORDER_WIDTH_HIGHLIGHT + 2 if is_hl else _BORDER_WIDTH_NORMAL + 2
                ),
                font={"size": 13 if is_hl else 11, "bold": is_hl},
            )

        # --- Edges ---
        for edge in subgraph.edges:
            # Skip edges that reference nodes not present in the subgraph
            if edge.source_id not in node_ids_in_graph or edge.target_id not in node_ids_in_graph:
                logger.debug(
                    "Skipping edge %s→%s — endpoint not in subgraph",
                    edge.source_id, edge.target_id,
                )
                continue

            both_hl = edge.source_id in highlighted and edge.target_id in highlighted
            src_name = id_to_name.get(edge.source_id, edge.source_id)
            tgt_name = id_to_name.get(edge.target_id, edge.target_id)

            net.add_edge(
                edge.source_id,
                edge.target_id,
                label=edge.edge_type.value,
                color=_EDGE_COLOR_HIGHLIGHT if both_hl else _EDGE_COLOR_NORMAL,
                width=_EDGE_WIDTH_HIGHLIGHT if both_hl else _EDGE_WIDTH_NORMAL,
                title=build_edge_tooltip(edge, src_name, tgt_name),
                arrows="to",
                font={"size": 10, "strokeWidth": 0},
            )

        self._apply_options(net)
        html = net.generate_html(notebook=False)
        html = _inject_legend(html)

        logger.info(
            "GraphVisualizer: %d nodes, %d edges, %d highlighted — %d chars",
            len(subgraph.nodes),
            len(subgraph.edges),
            len(highlighted & node_ids_in_graph),
            len(html),
        )
        return html

    def save(
        self,
        html: str,
        path: Optional[str] = None,
    ) -> str:
        """Write *html* to *path* and return the absolute path.

        Creates parent directories if they do not exist.

        Parameters
        ----------
        html:
            HTML string returned by :meth:`build_pyvis_graph`.
        path:
            Output file path. Defaults to :attr:`output_path`.

        Returns
        -------
        str
            Absolute path of the written file.
        """
        target = os.path.abspath(path or self.output_path)
        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(target, "w", encoding="utf-8") as fh:
            fh.write(html)
        logger.info("GraphVisualizer: saved to %s (%d chars)", target, len(html))
        return target

    def build_and_save(
        self,
        subgraph: GraphSubgraph,
        highlight_node_ids: Optional[List[str]] = None,
        path: Optional[str] = None,
    ) -> str:
        """Build the pyvis graph and save it to disk in one call.

        Parameters
        ----------
        subgraph:
            The ``GraphSubgraph`` to visualise.
        highlight_node_ids:
            Node IDs that appear in the answer path.
        path:
            Output file path. Defaults to :attr:`output_path`.

        Returns
        -------
        str
            Absolute path of the saved HTML file.
        """
        html = self.build_pyvis_graph(subgraph, highlight_node_ids)
        return self.save(html, path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_network(self) -> Any:
        """Instantiate and return a configured pyvis Network."""
        return PyvisNetwork(
            height=self.height,
            width=self.width,
            bgcolor=self.bgcolor,
            font_color=_FONT_COLOR,
            directed=True,
            notebook=False,
            cdn_resources="in_line",  # embeds vis.js — works offline
        )

    def _apply_options(self, net: Any) -> None:
        """Apply physics and interaction options to the network."""
        try:
            net.set_options(_PHYSICS_OPTIONS)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not apply pyvis options: %s", exc)

    def _empty_html(self) -> str:
        """Return a minimal valid HTML page for an empty subgraph."""
        return (
            "<html><head><meta charset='utf-8'><title>Knowledge Graph</title></head>"
            "<body style='font-family:Arial;display:flex;align-items:center;"
            "justify-content:center;height:100vh;margin:0;background:#f8f9fa;'>"
            "<p style='color:#636e72;font-size:18px;'>No graph data available.</p>"
            f"{_LEGEND_HTML}</body></html>"
        )

    def _fallback_html(self, subgraph: GraphSubgraph, highlighted: Set[str]) -> str:
        """Text-only HTML when pyvis is not installed."""
        id_to_name = {n.id: n.name for n in subgraph.nodes}
        node_rows = "".join(
            f"<li>[{n.node_type.value}] {n.name}{'  ★' if n.id in highlighted else ''}"
            f" &nbsp;<code style='color:#636e72'>{n.id}</code></li>"
            for n in subgraph.nodes
        )
        edge_rows = "".join(
            f"<li>{id_to_name.get(e.source_id, e.source_id)} "
            f"—[{e.edge_type.value}]→ "
            f"{id_to_name.get(e.target_id, e.target_id)} "
            f"(conf={e.confidence:.2f})</li>"
            for e in subgraph.edges
        )
        return (
            "<html><head><meta charset='utf-8'>"
            "<title>Knowledge Graph (text mode)</title></head>"
            "<body style='font-family:Arial;padding:24px;'>"
            "<h2>Knowledge Graph</h2>"
            "<p style='color:#e17055'>pyvis not installed — showing text view</p>"
            f"<h3>Nodes ({len(subgraph.nodes)})</h3>"
            f"<ul>{node_rows or '<li>none</li>'}</ul>"
            f"<h3>Edges ({len(subgraph.edges)})</h3>"
            f"<ul>{edge_rows or '<li>none</li>'}</ul>"
            "</body></html>"
        )


# ---------------------------------------------------------------------------
# Module-level helper functions (public — tested independently)
# ---------------------------------------------------------------------------


def get_node_color(node_type_value: str, is_highlighted: bool) -> Dict[str, Any]:
    """Return the pyvis colour dict for a given ``NodeType`` value string.

    Parameters
    ----------
    node_type_value:
        The ``.value`` of a ``NodeType`` member (e.g. ``"Drug"``).
    is_highlighted:
        Whether to use the brighter answer-path colour variant.

    Returns
    -------
    Dict
        pyvis ``color`` dict with ``background``, ``border``,
        ``highlight``, and ``hover`` sub-keys.
    """
    normal_bg, normal_bdr, bright_bg, bright_bdr = NODE_COLORS.get(
        node_type_value, _DEFAULT_COLOR
    )
    if is_highlighted:
        return {
            "background": bright_bg,
            "border": HIGHLIGHT_BORDER,
            "highlight": {"background": bright_bg, "border": HIGHLIGHT_BORDER},
            "hover":     {"background": bright_bg, "border": HIGHLIGHT_BORDER},
        }
    return {
        "background": normal_bg,
        "border": normal_bdr,
        "highlight": {"background": bright_bg, "border": bright_bdr},
        "hover":     {"background": bright_bg, "border": bright_bdr},
    }


def build_node_tooltip(node: GraphNode, is_highlighted: bool) -> str:
    """Build an HTML tooltip string for a graph node.

    Parameters
    ----------
    node:
        A ``GraphNode`` instance.
    is_highlighted:
        Whether the node is in the answer path.

    Returns
    -------
    str
        HTML safe for the pyvis ``title`` parameter (shown on hover).
    """
    badge = " &nbsp;<b style='color:#f1c40f'>★ Answer path</b>" if is_highlighted else ""
    url_line = (
        f"<a href='{node.source_url}' target='_blank' style='color:#0984e3;'>"
        f"Source ↗</a>"
        if node.source_url
        else "<span style='color:#b2bec3'>No source URL</span>"
    )
    updated = (
        node.last_updated.strftime("%Y-%m-%d") if node.last_updated else "unknown"
    )
    return (
        f"<div style='font-family:Arial;font-size:12px;max-width:280px;"
        f"line-height:1.6;padding:4px;'>"
        f"<b style='font-size:13px;'>{node.name}</b>{badge}<br>"
        f"<span style='color:#636e72;font-size:11px;'>{node.node_type.value}</span><br>"
        f"ID:&nbsp;<code style='font-size:11px;'>{node.id}</code><br>"
        f"Confidence:&nbsp;{node.confidence_score:.2f}<br>"
        f"Updated:&nbsp;{updated}<br>"
        f"{url_line}"
        f"</div>"
    )


def build_edge_tooltip(edge: GraphEdge, src_name: str, tgt_name: str) -> str:
    """Build an HTML tooltip string for a graph edge.

    Parameters
    ----------
    edge:
        A ``GraphEdge`` instance.
    src_name:
        Display name of the source node.
    tgt_name:
        Display name of the target node.

    Returns
    -------
    str
        HTML safe for the pyvis edge ``title`` parameter.
    """
    year_part = f"&nbsp;({edge.year})" if edge.year else ""
    doc_part = (
        f"<br>Doc:&nbsp;<code style='font-size:10px;'>{edge.source_doc_id}</code>"
        if edge.source_doc_id
        else ""
    )
    return (
        f"<div style='font-family:Arial;font-size:12px;line-height:1.6;padding:4px;'>"
        f"<b>{src_name}</b> → <b>{tgt_name}</b><br>"
        f"Relation:&nbsp;<b>{edge.edge_type.value}</b>{year_part}<br>"
        f"Confidence:&nbsp;{edge.confidence:.2f}"
        f"{doc_part}"
        f"</div>"
    )


def _inject_legend(html: str) -> str:
    """Inject the colour-coding legend div before the closing body tag."""
    if "</body>" in html:
        return html.replace("</body>", f"{_LEGEND_HTML}</body>", 1)
    return html + _LEGEND_HTML
