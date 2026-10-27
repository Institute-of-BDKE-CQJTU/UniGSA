"""FinQA dataset loader."""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

from transformers import PreTrainedTokenizer

from UniGSA.core.graph_utils import table_to_graph_hierarchical, table_to_graph
from UniGSA.core.tokenize_utils import (
    tokenize_sample, split_table_for_max_length,
    estimate_token_length, load_reasoning_jsonl, build_reasoning_answer,
)
from UniGSA.prompts.text_table import build_qa_prompt_ttqa
from UniGSA.datasets.base import DatasetSpec, register_dataset
from UniGSA.datasets.program_utils import convert_program_to_expr

logger = logging.getLogger(__name__)


def format_finqa_answer(qa_item: Dict) -> str:
    program = qa_item.get("program_re", "") or qa_item.get("program", "")
    if program:
        return convert_program_to_expr(program)
    return str(qa_item.get("answer", ""))


def _extract_finqa_gold_text(pre_text, post_text, gold_inds):
    text_entries = []
    for key, text_val in (gold_inds or {}).items():
        if key.startswith("text_"):
            try:
                idx = int(key.split("_")[1])
                text_entries.append((idx, text_val))
            except (ValueError, IndexError):
                text_entries.append((999, text_val))
    text_entries.sort(key=lambda x: x[0])
    if text_entries:
        return [t for _, t in text_entries]
    return pre_text[:3] if pre_text else []


def load_finqa_dataset(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    max_samples: int = -1,
    split_oversized: bool = False,
    max_no_answer_ratio: float = 1.0,
    no_answer_text: str = "no answer",
    use_gold_text: bool = False,
    reasoning_path: Optional[str] = None,
    **_,
) -> List[Dict]:
    use_reasoning = reasoning_path is not None
    reasoning_map = load_reasoning_jsonl(reasoning_path) if use_reasoning else {}
    if use_reasoning:
        logger.info(f"FinQA: loaded {len(reasoning_map)} matched reasoning entries")

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples: List[Dict] = []
    skipped, split_count, reasoning_skipped = 0, 0, 0

    for item in data:
        if 0 < max_samples <= len(samples):
            break

        grid = item.get("table", [])
        if not grid or not grid[0]:
            skipped += 1; continue

        pre_text = item.get("pre_text", [])
        post_text = item.get("post_text", [])

        qa = item.get("qa", {})
        question = qa.get("question", "")
        answer = format_finqa_answer(qa)
        sid = item.get("id", "")

        if not answer or not question:
            skipped += 1; continue

        if use_reasoning:
            if sid not in reasoning_map:
                reasoning_skipped += 1; continue
            answer = build_reasoning_answer(reasoning_map[sid]["reasoning"], answer)

        if use_gold_text:
            paragraphs = _extract_finqa_gold_text(pre_text, post_text, qa.get("gold_inds", {}))
        else:
            paragraphs = pre_text + post_text

        caption = ""

        def _builder(graph, q, caption=""):
            return build_qa_prompt_ttqa(graph, q, paragraphs=paragraphs, caption=caption,
                                        use_reasoning=use_reasoning)

        if split_oversized:
            graph_tmp = table_to_graph(grid)
            prompt_tmp, _ = _builder(graph_tmp, question)
            est = estimate_token_length(prompt_tmp, answer, tokenizer)
            if est > max_length:
                sub_tables = split_table_for_max_length(
                    grid, question, answer, caption, tokenizer, max_length,
                    prompt_builder=_builder,
                    header_rows=1, header_cols=0,
                    no_answer_text=no_answer_text,
                    max_no_answer_ratio=max_no_answer_ratio,
                )
                if len(sub_tables) > 1:
                    split_count += 1
            else:
                sub_tables = [(grid, answer)]
        else:
            sub_tables = [(grid, answer)]

        for sg, sa in sub_tables:
            graph = table_to_graph_hierarchical(sg, 1, 0)
            prompt, _ = _builder(graph, question)
            sample = tokenize_sample(prompt, sa, graph, tokenizer, max_length)
            if sample is not None:
                samples.append(sample)

    if skipped:
        logger.warning(f"FinQA: skipped {skipped} samples")
    if reasoning_skipped:
        logger.info(f"FinQA: skipped {reasoning_skipped} samples (no matched reasoning)")
    if split_count:
        logger.info(f"FinQA: split {split_count} oversized samples")
    logger.info(f"FinQA: loaded {len(samples)} samples (use_reasoning={use_reasoning})")
    return samples


def _finqa_evaluator(*args, **kwargs):
    from UniGSA.eval.finqa import run_finqa_eval
    return run_finqa_eval(*args, **kwargs)


register_dataset(DatasetSpec(
    name="finqa",
    loader=load_finqa_dataset,
    evaluator=_finqa_evaluator,
    primary_metric="em",
    needs_text_context=True,
    multi_table=False,
    default_max_length=2048,
    description="FinQA — text + single-table QA with computation programs",
))
