"""TAT-QA dataset loader."""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

from transformers import PreTrainedTokenizer

from UniGSA.core.graph_utils import table_to_graph_hierarchical
from UniGSA.core.tokenize_utils import (
    tokenize_sample, split_table_for_max_length,
    estimate_token_length, load_reasoning_jsonl,
    build_reasoning_answer, build_reasoning_answer_text_extraction,
)
from UniGSA.prompts.text_table import build_qa_prompt_ttqa
from UniGSA.datasets.base import DatasetSpec, register_dataset

logger = logging.getLogger(__name__)

SCALE_MAP = {
    "percent": "%", "thousand": " thousand",
    "million": " million", "billion": " billion",
}


def format_tatqa_answer(question_item: Dict) -> str:
    answer_type = question_item.get("answer_type", "span")
    answer = question_item.get("answer", "")
    derivation = question_item.get("derivation", "")
    scale = question_item.get("scale", "")
    suffix = SCALE_MAP.get(scale, "")

    if answer_type == "arithmetic":
        if derivation:
            expr = derivation.strip()
            return f"{expr}{suffix}" if suffix else expr
        ans_str = (", ".join(str(a) for a in answer)
                   if isinstance(answer, list) else str(answer))
        return f"{ans_str}{suffix}" if suffix else ans_str

    if answer_type in ("span", "multi-span"):
        ans_str = (", ".join(str(a) for a in answer)
                   if isinstance(answer, list) else str(answer))
        return f"{ans_str}{suffix}" if suffix else ans_str

    if answer_type == "count":
        return str(answer)

    return (", ".join(str(a) for a in answer)
            if isinstance(answer, list) else str(answer))


def _select_paragraphs_for_text_answer(
    answer: str, all_paragraphs: List[str], max_context: int = 5,
) -> List[str]:
    """Pick paragraphs containing the answer + neighbors."""
    if not all_paragraphs:
        return all_paragraphs
    al = answer.lower().strip()
    parts = [a.strip() for a in al.split(", ")]
    containing = set()
    for i, p in enumerate(all_paragraphs):
        pl = p.lower()
        for part in parts:
            if part and part in pl:
                containing.add(i); break
    if not containing:
        return all_paragraphs
    expanded = set()
    for idx in containing:
        for off in range(-1, 2):
            j = idx + off
            if 0 <= j < len(all_paragraphs):
                expanded.add(j)
    sel = sorted(expanded)
    if len(sel) > max_context:
        prio = sorted(containing)
        extra = [i for i in sel if i not in containing]
        sel = sorted(prio + extra[: max_context - len(prio)])
    return [all_paragraphs[i] for i in sel]


def _build_reasoning_answer_ttqa(reasoning, original_answer, answer_from, answer_type):
    if answer_from == "text" and answer_type in ("span", "multi-span"):
        return build_reasoning_answer_text_extraction(reasoning, original_answer)
    return build_reasoning_answer(reasoning, original_answer)


def load_tatqa_dataset(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    max_samples: int = -1,
    split_oversized: bool = False,
    max_no_answer_ratio: float = 1.0,
    no_answer_text: str = "no answer",
    reasoning_path: Optional[str] = None,
    use_gold_text: bool = False,
    **_,
) -> List[Dict]:
    use_reasoning = reasoning_path is not None
    reasoning_map = load_reasoning_jsonl(reasoning_path) if use_reasoning else {}
    if use_reasoning:
        logger.info(f"TAT-QA: loaded {len(reasoning_map)} matched reasoning entries")

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples: List[Dict] = []
    skipped, split_count, reasoning_skipped = 0, 0, 0

    for item in data:
        if 0 < max_samples <= len(samples):
            break

        table_data = item.get("table", {})
        grid = table_data.get("table", [])
        if not grid or not grid[0]:
            skipped += 1; continue

        paragraphs_raw = item.get("paragraphs", [])
        all_paragraphs = [p.get("text", "") for p in
                          sorted(paragraphs_raw, key=lambda x: x.get("order", 0))]

        for q_item in item.get("questions", []):
            if 0 < max_samples <= len(samples):
                break

            question = q_item.get("question", "")
            answer = format_tatqa_answer(q_item)
            sid = q_item.get("uid", "")
            answer_from = q_item.get("answer_from", "")
            answer_type = q_item.get("answer_type", "")

            if not answer or not question:
                skipped += 1; continue

            if use_reasoning:
                if sid not in reasoning_map:
                    reasoning_skipped += 1; continue
                answer = _build_reasoning_answer_ttqa(
                    reasoning_map[sid]["reasoning"], answer, answer_from, answer_type
                )

            if use_gold_text:
                rel_para_ids = q_item.get("rel_paragraphs", [])
                if rel_para_ids:
                    paragraphs = []
                    for pid in rel_para_ids:
                        try:
                            pid_int = int(pid)
                        except (ValueError, TypeError):
                            pid_int = pid
                        for p in paragraphs_raw:
                            if p.get("order") == pid_int or p.get("order") == pid:
                                paragraphs.append(p.get("text", ""))
                                break
                else:
                    if answer_from == "text" and answer_type in ("span", "multi-span"):
                        paragraphs = _select_paragraphs_for_text_answer(answer, all_paragraphs)
                    else:
                        paragraphs = all_paragraphs
            else:
                paragraphs = all_paragraphs

            caption = ""

            def _builder(graph, q, caption=""):
                return build_qa_prompt_ttqa(graph, q, paragraphs=paragraphs, caption=caption,
                                            use_reasoning=use_reasoning)

            if split_oversized:
                # Estimate using full table first
                from UniGSA.core.graph_utils import table_to_graph
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
        logger.warning(f"TAT-QA: skipped {skipped} samples (empty fields)")
    if reasoning_skipped:
        logger.info(f"TAT-QA: skipped {reasoning_skipped} samples (no matched reasoning)")
    if split_count:
        logger.info(f"TAT-QA: split {split_count} oversized samples")
    logger.info(f"TAT-QA: loaded {len(samples)} samples (use_reasoning={use_reasoning})")
    return samples


def _tatqa_evaluator(*args, **kwargs):
    from UniGSA.eval.tatqa import run_tatqa_eval
    return run_tatqa_eval(*args, **kwargs)


register_dataset(DatasetSpec(
    name="tatqa",
    loader=load_tatqa_dataset,
    evaluator=_tatqa_evaluator,
    primary_metric="em",
    needs_text_context=True,
    multi_table=False,
    default_max_length=2048,
    description="TAT-QA — text + single-table QA",
))
