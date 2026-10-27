"""Generic single-table QA loader (HTML / 2D-list inputs)."""

from __future__ import annotations

import json
import logging
from typing import Dict, List

from transformers import PreTrainedTokenizer

from UniGSA.core.graph_utils import table_to_graph_hierarchical, parse_html_table
from UniGSA.core.tokenize_utils import (
    tokenize_sample, split_table_for_max_length,
)
from UniGSA.prompts.table_only import build_qa_prompt
from UniGSA.datasets.base import DatasetSpec, register_dataset

logger = logging.getLogger(__name__)


def load_generic_dataset(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 1024,
    max_samples: int = -1,
    split_oversized: bool = False,
    max_no_answer_ratio: float = 1.0,
    no_answer_text: str = "no answer",
    **_,
) -> List[Dict]:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if 0 < max_samples:
        data = data[:max_samples]

    samples: List[Dict] = []
    split_count = 0
    for item in data:
        question = item["question"]
        ans = item["answer"]
        if isinstance(ans, list):
            ans = ", ".join(str(a) for a in ans)
        td = item["table"]
        grid = parse_html_table(td) if isinstance(td, str) else td
        if not grid:
            continue
        caption = item.get("caption", "")
        header_rows = item.get("header_rows", 1)
        header_cols = item.get("header_cols", 0)

        if split_oversized:
            sub_tables = split_table_for_max_length(
                grid, question, ans, caption, tokenizer, max_length,
                prompt_builder=build_qa_prompt,
                header_rows=header_rows, header_cols=header_cols,
                no_answer_text=no_answer_text,
                max_no_answer_ratio=max_no_answer_ratio,
            )
            if len(sub_tables) > 1:
                split_count += 1
        else:
            sub_tables = [(grid, ans)]

        for sg, sa in sub_tables:
            graph = table_to_graph_hierarchical(sg, header_rows, header_cols)
            prompt, _ = build_qa_prompt(graph, question, caption=caption)
            sample = tokenize_sample(prompt, sa, graph, tokenizer, max_length)
            if sample is not None:
                samples.append(sample)

    if split_count:
        logger.info(f"Generic: split {split_count} oversized samples")
    logger.info(f"Generic: loaded {len(samples)} samples")
    return samples


def _generic_evaluator(*args, **kwargs):
    from UniGSA.eval.hitab import run_hitab_eval
    raise NotImplementedError("Generic evaluator not implemented (use the HiTab "
                              "evaluator with a compatible JSONL).")


register_dataset(DatasetSpec(
    name="generic",
    loader=load_generic_dataset,
    evaluator=_generic_evaluator,
    primary_metric="em",
    needs_text_context=False,
    multi_table=False,
    default_max_length=1024,
    description="Generic 2D-grid / HTML table QA loader",
))
