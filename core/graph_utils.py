"""Deterministic cell graphs and header-aware HTML parsing for UniGSA."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from bs4 import BeautifulSoup


# ---- Edge Type Registry ----

EDGE_TYPE_MAP = {
    "PAD": 0,            # padding (used in collator)
    "PARENT_CHILD": 1,   # hierarchy tree parent ↔ child (within a table)
    "SIBLING": 2,        # same parent in hierarchy tree (within a table)
    "CROSS_TABLE": 3,    # cross-table link (e.g. shared header across tables)
}
NUM_EDGE_TYPES = 4


# ---- Data structures ----

@dataclass
class GraphNode:
    row: int
    col: int
    value: str
    node_id: int = 0
    depth: int = 0       # hierarchy depth (1-based; 0 reserved for non-hier)
    parent_id: int = 0   # parent's node_id + 1 (0 = no parent)
    table_id: int = 0    # 0 = single-table or non-table; 1..N for multi-table


@dataclass
class GraphEdge:
    src: int
    dst: int
    edge_type: str  # "SAME_ROW", "SAME_COL", "PARENT_CHILD", "SIBLING", "CROSS_TABLE"


@dataclass
class HierEdge:
    """Edge for GAT message passing (uses EDGE_TYPE_MAP integer IDs)."""
    src: int
    dst: int
    edge_type: int


@dataclass
class TableGraph:
    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)
    hier_edges: List[HierEdge] = field(default_factory=list)
    num_rows: int = 0
    num_cols: int = 0
    top_depth: int = 0
    left_depth: int = 0
    num_tables: int = 1
    # Per-table metadata (for multi-table graphs)
    table_shapes: List[Tuple[int, int]] = field(default_factory=list)  # [(rows, cols), ...]
    table_depths: List[Tuple[int, int]] = field(default_factory=list)  # [(top_d, left_d), ...]
    _pos_to_node: Dict[Tuple[int, int, int], int] = field(default_factory=dict)
    # ↑ key = (table_id, row, col) — table_id 0 for single-table compatibility


# ---- Single-table graph construction ----

def _add_grid_nodes_and_edges(
    graph: TableGraph,
    grid: List[List[str]],
    table_id: int = 0,
) -> Tuple[int, int]:
    """
    Append the grid nodes. Typed hierarchy edges are constructed separately.
    Returns: (start_node_id, end_node_id_exclusive) for newly added nodes.
    """
    if not grid or not grid[0]:
        return len(graph.nodes), len(graph.nodes)

    num_rows, num_cols = len(grid), len(grid[0])
    start_nid = len(graph.nodes)

    # Add nodes
    for r in range(num_rows):
        for c in range(num_cols):
            val = str(grid[r][c]) if grid[r][c] is not None else ""
            nid = len(graph.nodes)
            graph.nodes.append(GraphNode(row=r, col=c, value=val,
                                         node_id=nid, table_id=table_id))
            graph._pos_to_node[(table_id, r, c)] = nid



    return start_nid, len(graph.nodes)


def _add_hierarchy_for_grid(
    graph: TableGraph,
    grid: List[List[str]],
    top_depth: int,
    left_depth: int,
    table_id: int = 0,
) -> None:
    """Compute per-node depth/parent + add PARENT_CHILD / SIBLING edges
    for nodes belonging to ``table_id``."""
    if not grid or not grid[0]:
        return
    num_rows, num_cols = len(grid), len(grid[0])
    if not (0 <= top_depth <= num_rows and 0 <= left_depth <= num_cols):
        raise ValueError("Header depths must lie within the table dimensions")

    # ---- per-node depth ----
    for node in graph.nodes:
        if node.table_id != table_id:
            continue
        r, c = node.row, node.col
        if r < top_depth and c < left_depth:
            node.depth = max(r, c) + 1
        elif r < top_depth:
            node.depth = r + 1
        elif c < left_depth:
            node.depth = c + 1
        else:
            node.depth = max(top_depth, left_depth) + 1

    # Paper parent identifiers: top headers take precedence in the corner.
    # Root headers use the reserved NULL identifier (0).
    table_nodes = [n for n in graph.nodes if n.table_id == table_id]
    for node in table_nodes:
        r, c = node.row, node.col
        parent_pos = None
        if r < top_depth:
            if r > 0:
                parent_pos = (r - 1, c)
        elif c < left_depth:
            if c > 0:
                parent_pos = (r, c - 1)
        elif top_depth > 0:
            parent_pos = (top_depth - 1, c)
        node.parent_id = (graph._pos_to_node[(table_id, *parent_pos)] + 1
                          if parent_pos is not None else 0)

    def connect(src, dst, kind):
        if src == dst:
            return
        for u, v in ((src, dst), (dst, src)):
            graph.hier_edges.append(HierEdge(u, v, EDGE_TYPE_MAP[kind]))
        graph.edges.append(GraphEdge(src, dst, kind))

    # Include the bottom-most header -> data-cell link, not only header levels.
    for node in table_nodes:
        if node.parent_id:
            connect(node.parent_id - 1, node.node_id, "PARENT_CHILD")
    # Preserve both header trees in the overlapping top-left region.
    for r in range(num_rows):
        for c in range(1, left_depth):
            connect(graph._pos_to_node[(table_id, r, c - 1)],
                    graph._pos_to_node[(table_id, r, c)], "PARENT_CHILD")

    # Literal parent-identity rule in the paper, including shared NULL roots.
    # Distinct copies of a merged header are NOT silently merged by text.
    groups = {}
    for node in table_nodes:
        if node.row < top_depth or node.col < left_depth:
            groups.setdefault(node.parent_id, []).append(node)
    for siblings in groups.values():
        for i, u in enumerate(siblings):
            for v in siblings[i + 1:]:
                if u.value != v.value:
                    connect(u.node_id, v.node_id, "SIBLING")


def _dedup_hier_edges(graph: TableGraph) -> None:
    seen, deduped = set(), []
    for e in graph.hier_edges:
        key = (e.src, e.dst, e.edge_type)
        if key not in seen:
            seen.add(key)
            deduped.append(e)
    graph.hier_edges = deduped


# ---- Public single-table builders (table_id = 0 → backward compatible) ----

def table_to_graph(grid: List[List[str]]) -> TableGraph:
    """Build a basic single-table node set without unused all-pairs grid edges."""
    graph = TableGraph()
    if not grid or not grid[0]:
        return graph
    _add_grid_nodes_and_edges(graph, grid, table_id=0)
    graph.num_rows = len(grid)
    graph.num_cols = len(grid[0])
    graph.num_tables = 1
    graph.table_shapes = [(graph.num_rows, graph.num_cols)]
    graph.table_depths = [(0, 0)]
    return graph


def table_to_graph_hierarchical(
    grid: List[List[str]],
    top_depth: int = 0,
    left_depth: int = 0,
) -> TableGraph:
    """Build single-table hierarchical graph (PARENT_CHILD + SIBLING)."""
    graph = table_to_graph(grid)
    if not graph.nodes:
        return graph
    graph.top_depth = top_depth
    graph.left_depth = left_depth
    graph.table_depths = [(top_depth, left_depth)]
    _add_hierarchy_for_grid(graph, grid, top_depth, left_depth, table_id=0)
    _dedup_hier_edges(graph)
    return graph


# ---- Multi-table graph builder ----

def table_to_graph_multi(
    grids: List[List[List[str]]],
    table_depths: Optional[List[Tuple[int, int]]] = None,
    add_cross_table_edges: bool = True,
) -> TableGraph:
    """
    Combine tables; IDs are 1..N for multiple tables and 0 for one table.

    Args:
        grids: list of 2D grids (one per table)
        table_depths: optional list of (top_depth, left_depth) per table
        add_cross_table_edges: if True, link nodes across tables whose stripped
            text values match (heuristic for shared headers / row labels).
    """
    graph = TableGraph()
    if not grids:
        return graph

    if table_depths is None:
        table_depths = [(1 if g and g[0] else 0, 0) for g in grids]
    if len(table_depths) != len(grids):
        raise ValueError("One (top_depth, left_depth) pair is required per table")

    graph.num_tables = len(grids)
    graph.table_shapes = []
    graph.table_depths = list(table_depths)

    for index, grid in enumerate(grids):
        ti = index + 1 if len(grids) > 1 else 0
        if not grid or not grid[0]:
            graph.table_shapes.append((0, 0))
            continue
        _add_grid_nodes_and_edges(graph, grid, table_id=ti)
        graph.table_shapes.append((len(grid), len(grid[0])))
        td, ld = table_depths[index]
        _add_hierarchy_for_grid(graph, grid, td, ld, table_id=ti)

    # Cross-table edges: all header-node pairs with equal stripped/lowercase text
    # in the header region of two different tables.
    if add_cross_table_edges and len(grids) > 1:
        ct_type = EDGE_TYPE_MAP["CROSS_TABLE"]

        # Collect header nodes per table:
        # treat all rows < top_depth and all cols < left_depth as "header" cells.
        header_buckets: Dict[str, List[int]] = {}
        for node in graph.nodes:
            if node.table_id == 0:
                continue
            ti = node.table_id
            td, ld = table_depths[ti - 1] if ti - 1 < len(table_depths) else (0, 0)
            if not (node.row < td or node.col < ld):
                continue
            v = node.value.strip().lower()
            header_buckets.setdefault(v, []).append(node.node_id)

        for v, nids in header_buckets.items():
            if len(nids) < 2:
                continue
            # Only connect across different tables
            by_table: Dict[int, List[int]] = {}
            for nid in nids:
                by_table.setdefault(graph.nodes[nid].table_id, []).append(nid)
            tbl_keys = sorted(by_table)
            for i in range(len(tbl_keys)):
                for j in range(i + 1, len(tbl_keys)):
                    for a in by_table[tbl_keys[i]]:
                        for b in by_table[tbl_keys[j]]:
                            graph.hier_edges.append(HierEdge(a, b, ct_type))
                            graph.hier_edges.append(HierEdge(b, a, ct_type))
                            graph.edges.append(GraphEdge(a, b, "CROSS_TABLE"))

    _dedup_hier_edges(graph)
    # Aggregate top-level shape (use max for safety)
    if graph.table_shapes:
        graph.num_rows = max((rs for rs, _ in graph.table_shapes), default=0)
        graph.num_cols = max((cs for _, cs in graph.table_shapes), default=0)
    return graph


# ---- Edge / depth / parent extraction ----

def get_hier_edge_arrays(graph: TableGraph) -> Tuple[List[int], List[int], List[int]]:
    return ([e.src for e in graph.hier_edges],
            [e.dst for e in graph.hier_edges],
            [e.edge_type for e in graph.hier_edges])


def get_node_depths(graph: TableGraph) -> Dict[int, int]:
    return {n.node_id: n.depth for n in graph.nodes}


def get_node_parents(graph: TableGraph) -> Dict[int, int]:
    return {n.node_id: n.parent_id for n in graph.nodes}


def get_node_tables(graph: TableGraph) -> Dict[int, int]:
    return {n.node_id: n.table_id for n in graph.nodes}


# ---- HiTab kg → 2D grid ----

def _get_leaf_paths(node: Dict, path: Optional[List[str]] = None):
    if path is None:
        path = []
    val = node.get("value", "")
    current_path = path if val in ("<TOP>", "<LEFT>") else path + [str(val)]
    children = node.get("children_dict", [])
    if not children:
        return [(node.get("line_idx"), current_path)]
    results = []
    for c in children:
        results.extend(_get_leaf_paths(c, current_path))
    return results


def reconstruct_hitab_from_kg(kg: Dict) -> Tuple[List[List[str]], str, int, int]:
    """Reconstruct 2D grid from HiTab kg format.
    Returns: (grid, title, top_depth, left_depth)"""
    top_leaves = _get_leaf_paths(kg["top_root"])
    left_leaves = _get_leaf_paths(kg["left_root"])
    data = kg.get("data", [])
    title = kg.get("title", "")
    top_depth = max((len(p) for _, p in top_leaves), default=0)
    left_depth = max((len(p) for _, p in left_leaves), default=0)
    n_data_rows = len(data)
    n_data_cols = len(data[0]) if data else 0
    total_rows = top_depth + n_data_rows
    total_cols = left_depth + n_data_cols
    if total_rows == 0 or total_cols == 0:
        return [[]], title, top_depth, left_depth
    grid = [["" for _ in range(total_cols)] for _ in range(total_rows)]
    for col_idx, path in top_leaves:
        if col_idx is None:
            continue
        ac = left_depth + col_idx
        if ac < total_cols:
            for d, v in enumerate(path):
                if d < top_depth:
                    grid[d][ac] = str(v)
    for row_idx, path in left_leaves:
        if row_idx is None:
            continue
        ar = top_depth + row_idx
        if ar < total_rows:
            for d, v in enumerate(path):
                if d < left_depth:
                    grid[ar][d] = str(v)
    for ri, row in enumerate(data):
        for ci, cell in enumerate(row):
            val = cell.get("value", "") if isinstance(cell, dict) else cell
            ar, ac = top_depth + ri, left_depth + ci
            if ar < total_rows and ac < total_cols:
                grid[ar][ac] = str(val)
    return grid, title, top_depth, left_depth


def load_hitab_tables_jsonl(path: str) -> Dict[str, Dict]:
    tables = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                t = json.loads(line)
                tables[str(t["name"])] = t
    return tables


# ---- HTML table parser ----

def parse_html_table_with_depths(html: str):
    """Expand rowspan/colspan and infer header depths from HTML markup.

    Prefer thead/column th and row-scope th. For unannotated HTML use a
    one-row top header and no left header; this fallback is an implementation
    convention, not a hierarchy detector specified by the paper.
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        return [], (0, 0)
    rows = [r for r in table.find_all("tr") if r.find_parent("table") is table]
    occupied, kinds = {}, {}
    top_depth = 0
    for ri, row in enumerate(rows):
        cells = row.find_all(["th", "td"], recursive=False)
        if row.find_parent("thead") is not None:
            top_depth = max(top_depth, ri + 1)
        elif ri == top_depth and cells and all(
                c.name == "th" and c.get("scope") not in ("row", "rowgroup") for c in cells):
            top_depth = ri + 1
        ci = 0
        for cell in cells:
            rs = max(1, int(cell.get("rowspan", 1)))
            cs = max(1, int(cell.get("colspan", 1)))
            while any((ri, ci + dc) in occupied for dc in range(cs)):
                ci += 1
            text = cell.get_text(" ", strip=True)
            for dr in range(rs):
                for dc in range(cs):
                    if ri + dr < len(rows):
                        occupied[(ri + dr, ci + dc)] = text
                        kinds[(ri + dr, ci + dc)] = (cell.name, cell.get("scope", ""))
            ci += cs
    width = max((c + 1 for _, c in occupied), default=0)
    if not width:
        return [], (0, 0)
    grid = [[occupied.get((r, c), "") for c in range(width)] for r in range(len(rows))]
    top_depth = min(top_depth or 1, len(grid))
    left_counts = []
    for r in range(top_depth, len(grid)):
        count = 0
        for c in range(width):
            name, scope = kinds.get((r, c), ("td", ""))
            if name != "th" or scope in ("col", "colgroup"):
                break
            count += 1
        left_counts.append(count)
    left_depth = min(left_counts, default=0)
    return grid, (top_depth, left_depth)


def parse_html_table(html: str) -> List[List[str]]:
    return parse_html_table_with_depths(html)[0]
