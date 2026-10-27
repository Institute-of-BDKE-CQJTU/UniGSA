"""Unified inference helpers."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from UniGSA.core.graph_utils import (
    TableGraph, parse_html_table,
    table_to_graph_hierarchical, table_to_graph_multi,
)
from UniGSA.core.tokenize_utils import (
    SPECIAL_TOKENS, get_all_special_tokens,
    compute_graph_ids, graph_input_fields,
    _remap_hier_edges, _compute_token_node_meta,
)
from UniGSA.core.model import GraphLoRAModel
from UniGSA.inference.reasoning import extract_answer_from_reasoning

logger = logging.getLogger(__name__)


# ============================================================================
# Prepare model inputs from a graph + prompt
# ============================================================================

def prepare_inputs(
    tokenizer,
    graph: TableGraph,
    prompt: str,
    device: str = "cpu",
    max_length: Optional[int] = None,
):
    """Build all model input tensors from a TableGraph + prompt string."""
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    if max_length is not None:
        ids = ids[:max_length]
    fields = graph_input_fields(ids, graph, tokenizer)
    edge_src, edge_dst, edge_types = (fields.pop(k) for k in ("edge_src", "edge_dst", "edge_types"))
    t = lambda x: torch.tensor([x], dtype=torch.long, device=device)

    if len(edge_src) == 0:
        edge_src_t = torch.zeros(1, 1, dtype=torch.long, device=device)
        edge_dst_t = torch.zeros(1, 1, dtype=torch.long, device=device)
        edge_types_t = torch.zeros(1, 1, dtype=torch.long, device=device)
        edge_mask_t = torch.zeros(1, 1, dtype=torch.bool, device=device)
    else:
        edge_src_t = t(edge_src)
        edge_dst_t = t(edge_dst)
        edge_types_t = t(edge_types)
        edge_mask_t = torch.ones(1, len(edge_src), dtype=torch.bool, device=device)

    return {
        "input_ids": t(ids),
        "attention_mask": torch.ones(1, len(ids), device=device, dtype=torch.long),
        **{key: t(value) for key, value in fields.items()},
        "edge_src": edge_src_t,
        "edge_dst": edge_dst_t,
        "edge_types": edge_types_t,
        "edge_mask": edge_mask_t,
    }


# ============================================================================
# Single-prediction helper
# ============================================================================

def predict_single(
    model,
    tokenizer,
    graph: TableGraph,
    prompt: str,
    max_new_tokens: int = 256,
    device: str = "cpu",
    use_reasoning: bool = False,
    max_length: int = None,
    postprocess: bool = True,
):
    """Run greedy generation. If ``use_reasoning`` returns
    (extracted_answer, raw_output)."""
    if not graph.nodes:
        return ("", "") if use_reasoning else ""
    inputs = prepare_inputs(tokenizer, graph, prompt, device=device, max_length=max_length)
    model.eval()
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens)
    raw = tokenizer.decode(gen[0], skip_special_tokens=True).strip() if gen.numel() > 0 else ""
    from UniGSA.eval.base import finalize_answer
    answer = extract_answer_from_reasoning(raw) if use_reasoning else raw
    final = finalize_answer(answer) if postprocess else answer
    return (final, raw) if use_reasoning else final


# ============================================================================
# Convenience: build graph from raw input for each dataset family
# ============================================================================

def grid_to_graph(grid, top_depth: int = 0, left_depth: int = 0) -> TableGraph:
    if isinstance(grid, str):
        grid = parse_html_table(grid)
    return table_to_graph_hierarchical(grid, top_depth, left_depth)


def grids_to_multi_graph(grids, table_depths=None, add_cross_table_edges: bool = True) -> TableGraph:
    from UniGSA.core.graph_utils import parse_html_table_with_depths
    parsed, depths = [], []
    for i, value in enumerate(grids):
        grid, inferred = parse_html_table_with_depths(value) if isinstance(value, str) else (value, (1, 0))
        if grid and grid[0]:
            parsed.append(grid)
            depths.append(table_depths[i] if table_depths is not None else inferred)
    return table_to_graph_multi(parsed, table_depths=depths, add_cross_table_edges=add_cross_table_edges)


# ============================================================================
# Model loading
# ============================================================================

def load_model(model_path, graphlora_path, tokenizer_path=None, device="cpu", fp16=False):
    tok_path = tokenizer_path or graphlora_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": get_all_special_tokens()})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if fp16 else torch.float32,
        trust_remote_code=True,
    )
    base.resize_token_embeddings(len(tokenizer))
    model = GraphLoRAModel.load_graphlora(base, tokenizer, graphlora_path)
    model = model.to(device).eval()
    return model, tokenizer
