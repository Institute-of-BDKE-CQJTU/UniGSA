"""
MultiHiertt dataset loader (multi-table + text + program QA).

Dataset format:
    {
      "uid": str,
      "paragraphs": [str, ...],
      "tables": [HTML_str, ...],          # 2-7 HTML tables per sample
      "table_description": {              # "tableIdx-row-col" → sentence
          "0-2-1": "Table 0 shows ... is $446 ."
      },
      "qa": {
          "question": str,
          "answer": str | float,          # span_selection: str, arithmetic: float
          "question_type": "span_selection" | "arithmetic",
          "table_evidence": ["0-2-4", ...],
          "text_evidence": [int, ...],
          "program": "" | "subtract(1733,1912), divide(#0,1912), ..."
      }
    }

Training answer:
  - span_selection → answer string
  - arithmetic     → math expression derived from program (multi-step
                     #N references are resolved via convert_program_to_expr)
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizer

from UniGSA.core.graph_utils import parse_html_table, parse_html_table_with_depths, table_to_graph_multi
from UniGSA.core.tokenize_utils import (
    tokenize_sample, estimate_token_length,
    load_reasoning_jsonl, build_reasoning_answer,
)
from UniGSA.prompts.multi_table import build_qa_prompt_mtqa
from UniGSA.datasets.base import DatasetSpec, register_dataset
from UniGSA.datasets.program_utils import convert_program_to_expr

logger = logging.getLogger(__name__)


# ---- Answer formatting ----

def format_multihiertt_answer(qa: Dict) -> str:
    qtype = qa.get("question_type", "")
    answer = qa.get("answer", "")
    program = qa.get("program", "")

    if qtype == "arithmetic":
        if program:
            return convert_program_to_expr(program)
        # fall through to raw answer if program missing
        return _stringify(answer)
    # span_selection / unknown
    return _stringify(answer)


def _stringify(answer) -> str:
    if isinstance(answer, list):
        return ", ".join(str(a) for a in answer)
    if isinstance(answer, float):
        return str(answer) if answer != int(answer) else str(int(answer))
    return str(answer)


# ---- Evidence-based context selection ----

def _parse_table_evidence_keys(keys: List[str]) -> Dict[int, List[Tuple[int, int]]]:
    """Parse ['0-2-4', '1-3-1', ...] → {0: [(2,4)], 1: [(3,1)]}."""
    out: Dict[int, List[Tuple[int, int]]] = {}
    for k in keys:
        try:
            ti, r, c = k.split("-")
            out.setdefault(int(ti), []).append((int(r), int(c)))
        except ValueError:
            continue
    return out


def _select_tables_by_evidence(
    grids: List[List[List[str]]],
    table_evidence: List[str],
    keep_all_when_empty: bool = True,
) -> Tuple[List[List[List[str]]], List[int]]:
    """Filter tables to those referenced by table_evidence; keep originals
    if no evidence is present.

    Returns: (filtered_grids, original_indices_kept)
    """
    if not table_evidence:
        return (grids, list(range(len(grids)))) if keep_all_when_empty else ([], [])
    refs = _parse_table_evidence_keys(table_evidence)
    keep_idx = sorted(refs.keys())
    keep_idx = [i for i in keep_idx if 0 <= i < len(grids)]
    if not keep_idx:
        return (grids, list(range(len(grids)))) if keep_all_when_empty else ([], [])
    return [grids[i] for i in keep_idx], keep_idx


def _select_paragraphs_by_evidence(
    paragraphs: List[str],
    text_evidence: List[int],
    keep_all_when_empty: bool = True,
) -> List[str]:
    if not text_evidence:
        return paragraphs if keep_all_when_empty else []
    sel = sorted(set(int(i) for i in text_evidence if isinstance(i, (int, str))
                     and str(i).lstrip("-").isdigit()))
    sel = [i for i in sel if 0 <= i < len(paragraphs)]
    return [paragraphs[i] for i in sel] or (paragraphs if keep_all_when_empty else [])


def parse_multihiertt_tables(tables_html, evidence=None, table_depths=None):
    """Use the same HTML-derived header metadata in training and evaluation.

    Optional explicit per-table depths override inference. Gold filtering is
    only used when explicitly requested; indices refer to the ORIGINAL input.
    """
    selected = set(_parse_table_evidence_keys(evidence or []))
    grids, depths = [], []
    for index, html in enumerate(tables_html):
        if selected and index not in selected:
            continue
        grid, inferred = parse_html_table_with_depths(html)
        if grid and grid[0]:
            grids.append(grid)
            depths.append(tuple(table_depths[index]) if table_depths is not None else inferred)
    return grids, depths


# ---- Loader ----

def load_multihiertt_dataset(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 4096,
    max_samples: int = -1,
    use_gold_evidence: bool = False,
    add_cross_table_edges: bool = True,
    reasoning_path: Optional[str] = None,
    **_,
) -> List[Dict]:
    use_reasoning = reasoning_path is not None
    reasoning_map = load_reasoning_jsonl(reasoning_path) if use_reasoning else {}
    if use_reasoning:
        logger.info(f"MultiHiertt: loaded {len(reasoning_map)} matched reasoning entries")

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples: List[Dict] = []
    skipped, reasoning_skipped, shrunk = 0, 0, 0

    for item in data:
        if 0 < max_samples <= len(samples):
            break

        uid = item.get("uid", "")
        tables_html = item.get("tables", [])
        paragraphs = item.get("paragraphs", [])
        qa = item.get("qa", {})
        question = qa.get("question", "")
        answer = format_multihiertt_answer(qa)

        if not tables_html or not question or answer == "":
            skipped += 1; continue

        grids, table_depths = parse_multihiertt_tables(
            tables_html, qa.get("table_evidence", []) if use_gold_evidence else None,
            item.get("table_depths"),
        )
        if not grids:
            skipped += 1; continue

        if use_reasoning:
            if uid not in reasoning_map:
                reasoning_skipped += 1; continue
            answer = build_reasoning_answer(reasoning_map[uid]["reasoning"], answer)

        # Apply gold evidence selection (training-time only by default)
        if use_gold_evidence:
            paragraphs = _select_paragraphs_by_evidence(
                paragraphs, qa.get("text_evidence", []), keep_all_when_empty=False
            )

        graph = table_to_graph_multi(
            grids, table_depths=table_depths, add_cross_table_edges=add_cross_table_edges
        )
        prompt, _ = build_qa_prompt_mtqa(graph, question, paragraphs=paragraphs,
                                         use_reasoning=use_reasoning)
        sample = tokenize_sample(prompt, answer, graph, tokenizer, max_length)
        if sample is not None:
            samples.append(sample)

    if skipped:
        logger.warning(f"MultiHiertt: skipped {skipped} samples (empty / invalid)")
    if reasoning_skipped:
        logger.info(f"MultiHiertt: skipped {reasoning_skipped} samples (no matched reasoning)")
    if shrunk:
        logger.info(f"MultiHiertt: shrunk context for {shrunk} samples")
    logger.info(f"MultiHiertt: loaded {len(samples)} samples "
                f"(use_reasoning={use_reasoning}, gold_evidence={use_gold_evidence})")
    return samples


def _multihiertt_evaluator(*args, **kwargs):
    from UniGSA.eval.multihiertt import run_multihiertt_eval
    return run_multihiertt_eval(*args, **kwargs)


register_dataset(DatasetSpec(
    name="multihiertt",
    loader=load_multihiertt_dataset,
    evaluator=_multihiertt_evaluator,
    primary_metric="em",
    needs_text_context=True,
    multi_table=True,
    default_max_length=4096,
    description="MultiHiertt — multi-table + text + program QA",
))
