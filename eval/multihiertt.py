"""MultiHiertt evaluator (multi-table + program QA).

Metrics:
  - span_em: exact match for span_selection questions
  - exe_acc: numerical execution accuracy for arithmetic questions
  - overall: weighted average across both
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict

from UniGSA.core.graph_utils import parse_html_table, table_to_graph_multi
from UniGSA.prompts.multi_table import build_qa_prompt_mtqa
from UniGSA.eval.base import finalize_answer, final_answer_exact_match, exact_match, try_eval_expr
from UniGSA.inference.predictor import predict_single
from UniGSA.datasets.multihiertt import (
    format_multihiertt_answer,
    _select_tables_by_evidence,
    _select_paragraphs_by_evidence,
    parse_multihiertt_tables,
)

logger = logging.getLogger(__name__)


def run_multihiertt_eval(
    model, tokenizer,
    data_path: str,
    output_file: str = "multihiertt_predictions.json",
    max_new_tokens: int = 256,
    max_samples: int = -1,
    device: str = "cpu",
    use_gold_evidence: bool = False,    # at test time, default OFF
    add_cross_table_edges: bool = True,
    max_length: int = 4096,
    use_reasoning: bool = False,
    **_,
) -> Dict:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if 0 < max_samples:
        data = data[:max_samples]
    logger.info(f"MultiHiertt: {len(data)} samples from {data_path}")

    results, skipped = [], 0

    for i, item in enumerate(data):
        uid = item.get("uid", "")
        tables_html = item.get("tables", [])
        paragraphs = item.get("paragraphs", [])
        qa = item.get("qa", {})
        question = qa.get("question", "")
        qtype = qa.get("question_type", "")
        gt_answer_raw = qa.get("answer", "")
        gt_expr = format_multihiertt_answer(qa)

        if not tables_html or not question:
            skipped += 1; continue

        grids, table_depths = parse_multihiertt_tables(
            tables_html, qa.get("table_evidence", []) if use_gold_evidence else None,
            item.get("table_depths"),
        )
        if not grids:
            skipped += 1; continue

        if use_gold_evidence:
            paragraphs = _select_paragraphs_by_evidence(
                paragraphs, qa.get("text_evidence", []), keep_all_when_empty=True
            )

        graph = table_to_graph_multi(grids, table_depths=table_depths,
                                     add_cross_table_edges=add_cross_table_edges)
        prompt, _ = build_qa_prompt_mtqa(graph, question, paragraphs=paragraphs,
                                         use_reasoning=use_reasoning)
        res = predict_single(model, tokenizer, graph, prompt,
                             max_new_tokens=max_new_tokens, device=device,
                             max_length=max_length, postprocess=False,
                             use_reasoning=use_reasoning)
        if use_reasoning:
            pred, raw = res
        else:
            pred, raw = res, None

        pred_val = try_eval_expr(pred)
        matched = final_answer_exact_match(pred, gt_answer_raw)
        span_em = matched if qtype == "span_selection" else False
        exe_acc = matched if qtype == "arithmetic" else False

        entry = {
            "uid": uid, "question": question, "question_type": qtype,
            "ground_truth": gt_answer_raw, "ground_truth_expr": gt_expr,
            "prediction": pred, "final_answer": finalize_answer(pred), "pred_val": pred_val,
            "span_em": span_em, "exe_acc": exe_acc,
        }
        if raw is not None:
            entry["raw_output"] = raw
        results.append(entry)

        if (i + 1) % 25 == 0:
            sp = [r for r in results if r["question_type"] == "span_selection"]
            ar = [r for r in results if r["question_type"] == "arithmetic"]
            sp_acc = sum(r["span_em"] for r in sp) / max(len(sp), 1)
            ar_acc = sum(r["exe_acc"] for r in ar) / max(len(ar), 1)
            logger.info(f"  {i+1}/{len(data)} | span EM={sp_acc*100:.1f}% (n={len(sp)}) | "
                        f"arith Exe={ar_acc*100:.1f}% (n={len(ar)})")

    total = len(results)
    sp = [r for r in results if r["question_type"] == "span_selection"]
    ar = [r for r in results if r["question_type"] == "arithmetic"]
    sp_em = sum(r["span_em"] for r in sp)
    ar_exe = sum(r["exe_acc"] for r in ar)
    overall_correct = sp_em + ar_exe

    metrics = {
        "em": round(overall_correct / total * 100, 2) if total else 0,
        "overall": round(overall_correct / total * 100, 2) if total else 0,
        "span_em": round(sp_em / len(sp) * 100, 2) if sp else 0,
        "exe_acc": round(ar_exe / len(ar) * 100, 2) if ar else 0,
        "span_count": len(sp), "arith_count": len(ar), "total": total,
    }

    logger.info("=" * 60)
    logger.info(f"MultiHiertt: Overall={metrics['overall']:.2f}% "
                f"| span EM={metrics['span_em']:.2f}% (n={metrics['span_count']}) "
                f"| arith Exe={metrics['exe_acc']:.2f}% (n={metrics['arith_count']})")
    logger.info("=" * 60)

    out = {"dataset": "MultiHiertt", "metrics": metrics,
           "skipped": skipped, "predictions": results}
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved predictions to {output_file}")
    return metrics
