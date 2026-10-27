"""HiTab dataset loader."""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

from transformers import PreTrainedTokenizer

from UniGSA.core.graph_utils import (
    table_to_graph_hierarchical,
    reconstruct_hitab_from_kg,
    load_hitab_tables_jsonl,
)
from UniGSA.core.tokenize_utils import (
    tokenize_sample,
    split_table_for_max_length,
    load_reasoning_jsonl,
    build_reasoning_answer,
    check_answer_in_grid,
)
from UniGSA.prompts.table_only import build_qa_prompt
from UniGSA.datasets.base import DatasetSpec, register_dataset

logger = logging.getLogger(__name__)


def _normalize_answer_element(val) -> str:
    """Normalize a single answer element for consistent training targets.
    Converts floats like 51.0 → '51', keeps 52.1 → '52.1'."""
    if isinstance(val, float):
        if val == int(val):
            return str(int(val))
        return str(val)
    return str(val)


def load_hitab_dataset(
    samples_path: str,
    tokenizer: PreTrainedTokenizer,
    tables_path: Optional[str] = None,
    max_length: int = 2048,
    max_samples: int = -1,
    split_oversized: bool = False,
    max_no_answer_ratio: float = 1.0,
    no_answer_text: str = "no answer",
    reasoning_path: Optional[str] = None,
    enhanced_prompt: bool = False,
    **_,
) -> List[Dict]:
    """Load HiTab dataset for single-table QA training."""
    if tables_path is None:
        # Try standard layout: <hitab_dir>/{split}_samples_*.jsonl + <hitab_dir>/tables.jsonl
        tables_path = os.path.join(os.path.dirname(samples_path), "tables.jsonl")

    tables = load_hitab_tables_jsonl(tables_path)
    use_reasoning = reasoning_path is not None
    reasoning_map = load_reasoning_jsonl(reasoning_path) if use_reasoning else {}
    if use_reasoning:
        logger.info(f"HiTab: loaded {len(reasoning_map)} matched reasoning entries")

    samples: List[Dict] = []
    skipped, split_count, reasoning_skipped = 0, 0, 0

    with open(samples_path, "r", encoding="utf-8") as f:
        for line in f:
            if 0 < max_samples <= len(samples):
                break
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            sid = item.get("id", "")
            tname = str(item.get("context", ""))
            question = item.get("question", "")
            ans_raw = item.get("answer", "")
            # Normalize answer elements: 51.0 → "51", 52.1 → "52.1"
            answer = (", ".join(_normalize_answer_element(a) for a in ans_raw)
                      if isinstance(ans_raw, list) else str(ans_raw))

            # If this sample has a generated arithmetic expression
            # (HiTab v2 format), use the EXPRESSION itself as the training
            # target — mirroring FinQA's program-as-answer formulation.
            # Evaluation will execute the predicted expression and compare
            # numerically with the gold answer.  Direct/selection samples
            # keep their literal answer as the target.
            expression = item.get("expression")
            if expression:
                training_target = expression
            else:
                training_target = answer

            if use_reasoning:
                if sid not in reasoning_map:
                    reasoning_skipped += 1
                    continue
                full_answer = build_reasoning_answer(reasoning_map[sid]["reasoning"], training_target)
            else:
                full_answer = training_target



            if tname not in tables:
                skipped += 1
                continue
            kg = tables[tname].get("kg", {})
            grid, title, top_depth, left_depth = reconstruct_hitab_from_kg(kg)
            if not grid or not grid[0]:
                skipped += 1
                continue

            if split_oversized:
                sub_tables = split_table_for_max_length(
                    grid, question, full_answer, title, tokenizer, max_length,
                    prompt_builder=build_qa_prompt,
                    header_rows=top_depth, header_cols=left_depth,
                    no_answer_text=no_answer_text,
                    max_no_answer_ratio=max_no_answer_ratio,
                )
                if len(sub_tables) > 1:
                    split_count += 1
                if use_reasoning:
                    new_st = []
                    for sg, sa in sub_tables:
                        if sa == no_answer_text:
                            new_st.append((sg, no_answer_text))
                        else:
                            new_st.append((sg, full_answer if check_answer_in_grid(sg, answer) else no_answer_text))
                    sub_tables = new_st
            else:
                sub_tables = [(grid, full_answer)]

            for sg, sa in sub_tables:
                graph = table_to_graph_hierarchical(sg, top_depth, left_depth)
                prompt, _ = build_qa_prompt(graph, question, caption=title,
                                            use_reasoning=use_reasoning,
                                            enhanced=enhanced_prompt)
                sample = tokenize_sample(prompt, sa, graph, tokenizer, max_length)
                if sample is not None:
                    samples.append(sample)

    if skipped:
        logger.warning(f"HiTab: skipped {skipped} samples (table not found / empty)")
    if reasoning_skipped:
        logger.info(f"HiTab: skipped {reasoning_skipped} samples (no matched reasoning)")
    if split_count:
        logger.info(f"HiTab: split {split_count} oversized samples (total {len(samples)})")
    logger.info(f"HiTab: loaded {len(samples)} training samples (use_reasoning={use_reasoning})")
    return samples


# Lazy register: evaluator imported when needed (to avoid circular imports
# during package initialization)
def _hitab_evaluator(*args, **kwargs):
    from UniGSA.eval.hitab import run_hitab_eval
    return run_hitab_eval(*args, **kwargs)


register_dataset(DatasetSpec(
    name="hitab",
    loader=load_hitab_dataset,
    evaluator=_hitab_evaluator,
    primary_metric="em",
    needs_text_context=False,
    multi_table=False,
    default_max_length=2048,
    description="HiTab — hierarchical single-table QA",
))
