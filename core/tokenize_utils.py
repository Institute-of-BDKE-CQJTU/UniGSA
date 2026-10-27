"""
Unified tokenization helpers for UniGSA.

Public API:
    SPECIAL_TOKENS, IGNORE_INDEX, get_all_special_tokens
    serialize_graph_compact(graph)
    compute_graph_ids(input_ids, tids, ...)
    tokenize_sample(prompt, answer, graph, tokenizer, max_length)
        → unified tokenizer that returns the standard sample dict (input_ids,
        attention_mask, labels, row_ids, col_ids, node_ids, depth_ids,
        parent_ids, table_ids, edge_src/edge_dst/edge_types).
    estimate_token_length(prompt, answer, tokenizer)
    split_table_for_max_length(...)  — single-table chunking helper
    GraphQADataset
    load_reasoning_jsonl, build_reasoning_answer
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizer

from UniGSA.core.graph_utils import (
    TableGraph,
    table_to_graph,
    table_to_graph_hierarchical,
    get_hier_edge_arrays,
    get_node_depths,
    get_node_parents,
    get_node_tables,
)

logger = logging.getLogger(__name__)


# ---- Special tokens (added [TABLE] for multi-table support) ----

SPECIAL_TOKENS = {
    "GRAPH": "[GRAPH]", "TABLE": "[TABLE]", "ROW": "[ROW]",
    "CELL": "[CELL]", "END_GRAPH": "[END_GRAPH]",
}
SERIALIZATION_VERSION = "unigsa-paper-v1"
IGNORE_INDEX = -100


def get_all_special_tokens() -> List[str]:
    return list(SPECIAL_TOKENS.values())


def _get_special_token_ids(tokenizer) -> Dict[str, int]:
    return {k: tokenizer.convert_tokens_to_ids(v) for k, v in SPECIAL_TOKENS.items()}


# ============================================================================
# Serialization
# ============================================================================

def serialize_graph_compact(graph: TableGraph) -> str:
    """Paper equation: paired TABLE/ROW/CELL markers and END_GRAPH.

    Literal marker strings inside cell values are escaped with fullwidth
    brackets to prevent cell text from altering the structural grammar.
    """
    parts = [SPECIAL_TOKENS["GRAPH"]]
    tables = {}
    for node in graph.nodes:
        tables.setdefault(node.table_id, []).append(node)
    for tid in sorted(tables):
        parts.append(SPECIAL_TOKENS["TABLE"])
        rows = {}
        for node in tables[tid]:
            rows.setdefault(node.row, []).append(node)
        for row in sorted(rows):
            parts.append(SPECIAL_TOKENS["ROW"])
            for node in sorted(rows[row], key=lambda n: n.col):
                value = node.value
                for marker in SPECIAL_TOKENS.values():
                    value = value.replace(marker, marker.replace("[", "［").replace("]", "］"))
                parts.extend([SPECIAL_TOKENS["CELL"], value, SPECIAL_TOKENS["CELL"]])
            parts.append(SPECIAL_TOKENS["ROW"])
        parts.append(SPECIAL_TOKENS["TABLE"])
    parts.append(SPECIAL_TOKENS["END_GRAPH"])
    return " ".join(parts)

# ============================================================================
# Token-sequence node ID mapping
# ============================================================================

def _build_node_to_token_map(graph: TableGraph) -> Dict[int, int]:
    """
    Build mapping from graph.node_id → token-sequence node_id (1-indexed).

    Token node_id starts from 1 and increments in serialization order
    (table_id ascending, then row ascending, then col ascending).
    """
    if not graph.nodes:
        return {}

    mapping: Dict[int, int] = {}
    tables: Dict[int, List] = {}
    for n in graph.nodes:
        tables.setdefault(n.table_id, []).append(n)
    token_nid = 0
    for ti in sorted(tables.keys()):
        nodes_in_table = tables[ti]
        max_row = max(n.row for n in nodes_in_table)
        for r in range(max_row + 1):
            row_nodes = sorted((n for n in nodes_in_table if n.row == r),
                               key=lambda n: n.col)
            for n in row_nodes:
                token_nid += 1
                mapping[n.node_id] = token_nid
    return mapping


def _remap_hier_edges(graph: TableGraph) -> Tuple[List[int], List[int], List[int]]:
    """Map hier_edges from graph node IDs to token-sequence node IDs."""
    g2t = _build_node_to_token_map(graph)
    src_raw, dst_raw, type_raw = get_hier_edge_arrays(graph)
    edge_src, edge_dst, edge_types = [], [], []
    for s, d, t in zip(src_raw, dst_raw, type_raw):
        ts, td = g2t.get(s), g2t.get(d)
        if ts is not None and td is not None:
            edge_src.append(ts)
            edge_dst.append(td)
            edge_types.append(t)
    return edge_src, edge_dst, edge_types


def _compute_token_node_meta(
    graph: TableGraph,
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
    """Compute token-node-id → (depth, parent, table_id) dicts."""
    g2t = _build_node_to_token_map(graph)
    g_depths = get_node_depths(graph)
    g_parents = get_node_parents(graph)
    g_tables = get_node_tables(graph)

    token_depths: Dict[int, int] = {}
    token_parents: Dict[int, int] = {}
    token_tables: Dict[int, int] = {}
    for grid_nid, tok_nid in g2t.items():
        token_depths[tok_nid] = g_depths.get(grid_nid, 0)
        token_tables[tok_nid] = g_tables.get(grid_nid, 0)
        # Re-map parent (parent stored as grid_nid + 1; 0 means none)
        parent_grid_plus1 = g_parents.get(grid_nid, 0)
        if parent_grid_plus1 > 0:
            token_parents[tok_nid] = g2t.get(parent_grid_plus1 - 1, 0)
        else:
            token_parents[tok_nid] = 0
    return token_depths, token_parents, token_tables


# ============================================================================
# Per-token structural ID computation
# ============================================================================

def compute_graph_ids(
    input_ids: List[int], graph_tid: int, row_tid: int, col_tid: int,
    cell_tid: int, node_tid: int, edge_tid: int, table_tid: int = -1,
    node_depths=None, node_parents=None, node_tables=None, end_graph_tid: int = -1,
    _return_node_count: bool = False,
):
    """Align cell CONTENT only; every delimiter and non-cell token gets zero.

    col_tid/node_tid/edge_tid are retained as unused positional compatibility
    slots. Internal callers supply -1; they are not vocabulary additions.
    """
    fields = [[0] * len(input_ids) for _ in range(6)]
    in_graph = in_table = in_row = in_cell = False
    row = col = node = table = 0
    finished = False
    markers = {graph_tid, row_tid, cell_tid, table_tid, end_graph_tid} - {-1}
    for i, tid in enumerate(input_ids):
        if finished:
            continue  # Questions/answers cannot open a second graph.
        if tid == graph_tid and not in_graph:
            in_graph = True
        elif in_graph and tid == end_graph_tid:
            in_graph = in_table = in_row = in_cell = False
            finished = True
        elif in_graph and tid == table_tid:
            in_table = not in_table
            in_row = in_cell = False
            if in_table:
                table += 1
                row = col = 0
        elif in_graph and in_table and tid == row_tid:
            in_row = not in_row
            in_cell = False
            if in_row:
                row += 1
                col = 0
        elif in_graph and in_row and tid == cell_tid:
            in_cell = not in_cell
            if in_cell:
                col += 1
                node += 1
        elif in_graph and in_cell and tid not in markers:
            vals = (row, col, node, (node_depths or {}).get(node, 0),
                    (node_parents or {}).get(node, 0),
                    (node_tables or {}).get(node, table))
            for dest, value in zip(fields, vals):
                dest[i] = value
    return (tuple(fields), node) if _return_node_count else tuple(fields)


def graph_input_fields(input_ids, graph, tokenizer):
    tids = _get_special_token_ids(tokenizer)
    depths, parents, tables = _compute_token_node_meta(graph)
    values, retained_nodes = compute_graph_ids(
        input_ids, tids["GRAPH"], tids["ROW"], -1, tids["CELL"], -1, -1,
        table_tid=tids["TABLE"], end_graph_tid=tids["END_GRAPH"],
        node_depths=depths, node_parents=parents, node_tables=tables,
        _return_node_count=True,
    )
    names = ("row_ids", "col_ids", "node_ids", "depth_ids", "parent_ids", "table_ids")
    result = dict(zip(names, values))
    result["parent_ids"] = [p if p <= retained_nodes else 0 for p in result["parent_ids"]]
    g2t = _build_node_to_token_map(graph)
    # Keep nodes whose opening CELL marker survived prefix truncation, including
    # empty cells. Removed nodes must not send messages into the retained graph.
    metadata = [[0] * (retained_nodes + 1) for _ in range(4)]
    for node in graph.nodes:
        nid = g2t[node.node_id]
        if nid > retained_nodes:
            continue
        for field, value in zip(metadata, (node.row + 1, node.col + 1, node.depth, node.table_id)):
            field[nid] = value
    result.update(dict(zip(("graph_rows", "graph_cols", "graph_depths", "graph_tables"), metadata)))
    src, dst, types = _remap_hier_edges(graph)
    kept_edges = [(s, d, t) for s, d, t in zip(src, dst, types)
                  if s <= retained_nodes and d <= retained_nodes]
    result.update(edge_src=[e[0] for e in kept_edges],
                  edge_dst=[e[1] for e in kept_edges],
                  edge_types=[e[2] for e in kept_edges])
    return result

# ============================================================================
# Unified tokenize_sample
# ============================================================================

def tokenize_sample(
    prompt: str,
    answer: str,
    graph: TableGraph,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> Optional[Dict]:
    """
    Tokenize (prompt, answer, graph) into the standard training sample dict.

    The graph is used purely to derive structural IDs and edges — it must
    correspond to the same tables that were rendered into ``prompt`` via
    ``serialize_graph_compact``.
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    eos = tokenizer.eos_token_id or 0
    input_ids = prompt_ids + answer_ids + [eos]
    labels = [IGNORE_INDEX] * len(prompt_ids) + answer_ids + [eos]

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    # Causal LM shifts labels by one. An entirely truncated answer has no loss
    # targets and otherwise produces an all-ignored/NaN loss in common backbones.
    if not any(label != IGNORE_INDEX for label in labels[1:]):
        logger.warning("Skipping sample: no supervised answer tokens remain after truncation to %s tokens", max_length)
        return None
    return {
        "input_ids": input_ids, "attention_mask": [1] * len(input_ids),
        "labels": labels, **graph_input_fields(input_ids, graph, tokenizer),
    }


def estimate_token_length(prompt: str, answer: str, tokenizer: PreTrainedTokenizer) -> int:
    pids = tokenizer.encode(prompt, add_special_tokens=True)
    aids = tokenizer.encode(answer, add_special_tokens=False)
    return len(pids) + len(aids) + 1


# ============================================================================
# Single-table splitting (for oversized HiTab / TAT-QA / FinQA samples)
# ============================================================================

def _extract_sub_grid(grid, header_rows, header_cols,
                     data_row_start, data_row_end,
                     data_col_start, data_col_end):
    sub_grid = []
    for r in range(header_rows):
        row = [grid[r][c] for c in range(header_cols)]
        row += [grid[r][c] for c in range(data_col_start, data_col_end)]
        sub_grid.append(row)
    for r in range(data_row_start, data_row_end):
        row = [grid[r][c] for c in range(header_cols)]
        row += [grid[r][c] for c in range(data_col_start, data_col_end)]
        sub_grid.append(row)
    return sub_grid


def check_answer_in_grid(grid: List[List[str]], answer: str) -> bool:
    cell_values = set()
    for row in grid:
        for cell in row:
            cs = str(cell).strip().lower()
            if cs:
                cell_values.add(cs)
    al = answer.strip().lower()
    if al in cell_values:
        return True
    if ", " in answer:
        parts = [p.strip().lower() for p in answer.split(", ")]
        if all(p in cell_values for p in parts):
            return True
    for cv in cell_values:
        if al in cv:
            return True
    return False


def _estimate_grid_sample_length(
    sub_grid: List[List[str]], question: str, answer: str,
    title: str, tokenizer, prompt_builder,
) -> int:
    """Estimate sample token length for a candidate sub-grid."""
    graph = table_to_graph(sub_grid)
    prompt, _ = prompt_builder(graph, question, caption=title)
    return estimate_token_length(prompt, answer, tokenizer)


def split_table_for_max_length(
    grid: List[List[str]], question: str, answer: str, title: str,
    tokenizer, max_length: int,
    prompt_builder,
    header_rows: int = 1, header_cols: int = 0,
    no_answer_text: str = "no answer",
    max_no_answer_ratio: float = 1.0,
) -> List[Tuple[List[List[str]], str]]:
    """Split an oversized single-table sample into smaller sub-tables.

    ``prompt_builder(graph, question, caption=...)`` is called to estimate
    the length of each candidate; this allows different datasets to use
    different prompt formats (table-only / text+table / multi-table)."""
    total_rows = len(grid)
    total_cols = len(grid[0]) if grid else 0
    header_rows = min(header_rows, total_rows)
    header_cols = min(header_cols, total_cols)

    if _estimate_grid_sample_length(grid, question, answer, title, tokenizer, prompt_builder) <= max_length:
        return [(grid, answer)]

    drs, dre = header_rows, total_rows
    dcs, dce = header_cols, total_cols
    if dre - drs <= 0 or dce - dcs <= 0:
        return [(grid, answer)]

    # Binary-search column chunks first
    col_chunks = []
    c = dcs
    while c < dce:
        lo, hi, best = 1, dce - c, 1
        while lo <= hi:
            mid = (lo + hi) // 2
            sub = _extract_sub_grid(grid, header_rows, header_cols, drs, dre, c, c + mid)
            if _estimate_grid_sample_length(sub, question, answer, title, tokenizer, prompt_builder) <= max_length:
                best = mid; lo = mid + 1
            else:
                hi = mid - 1
        col_chunks.append((c, c + best))
        c += best

    sub_tables = []
    for cs_, ce_ in col_chunks:
        sub = _extract_sub_grid(grid, header_rows, header_cols, drs, dre, cs_, ce_)
        if _estimate_grid_sample_length(sub, question, answer, title, tokenizer, prompt_builder) <= max_length:
            sa = answer if check_answer_in_grid(sub, answer) else no_answer_text
            sub_tables.append((sub, sa))
        else:
            r = drs
            while r < dre:
                lo, hi, best = 1, dre - r, 1
                while lo <= hi:
                    mid = (lo + hi) // 2
                    sub_r = _extract_sub_grid(grid, header_rows, header_cols, r, r + mid, cs_, ce_)
                    if _estimate_grid_sample_length(sub_r, question, answer, title, tokenizer, prompt_builder) <= max_length:
                        best = mid; lo = mid + 1
                    else:
                        hi = mid - 1
                sub_r = _extract_sub_grid(grid, header_rows, header_cols, r, r + best, cs_, ce_)
                sa = answer if check_answer_in_grid(sub_r, answer) else no_answer_text
                sub_tables.append((sub_r, sa))
                r += best

    has = sum(1 for _, a in sub_tables if a != no_answer_text)
    no = sum(1 for _, a in sub_tables if a == no_answer_text)
    if has > 0 and max_no_answer_ratio >= 0:
        max_no = max(int(has * max_no_answer_ratio), 0)
        if no > max_no:
            kept, result = 0, []
            for sub, sa in sub_tables:
                if sa != no_answer_text:
                    result.append((sub, sa))
                elif kept < max_no:
                    result.append((sub, sa))
                    kept += 1
            sub_tables = result

    return sub_tables or [(grid, answer)]


# ============================================================================
# Reasoning helpers
# ============================================================================

def load_reasoning_jsonl(path: str) -> Dict[str, Dict]:
    """Load reasoning JSONL keyed by ``id`` (only matched entries)."""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            sid = item.get("id", "")
            if item.get("status") == "matched" and sid:
                out[sid] = {"reasoning": item.get("reasoning", "")}
    return out


def build_reasoning_answer(reasoning_text: str, original_answer: str) -> str:
    """Default builder: '<reasoning>. So the final answer is <answer>'."""
    reasoning_text = reasoning_text.strip()
    if reasoning_text.endswith("."):
        reasoning_text = reasoning_text[:-1]
    return f"{reasoning_text}. So the final answer is {original_answer}"


def build_reasoning_answer_text_extraction(reasoning_text: str, original_answer: str) -> str:
    """For TTQA span/multi-span: keeps full text answer below the reasoning."""
    reasoning_text = reasoning_text.strip()
    return f"{reasoning_text}\n\nThe extracted answer is: {original_answer}"


# ============================================================================
# Dataset wrapper
# ============================================================================

class GraphQADataset:
    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
