"""DocLibRAG evaluator.

Metrics:
  - em        : exact-match (string normalized; numeric tolerance) — primary
  - num_acc   : numeric tolerance match for items with numeric gt
  - f1        : token-level F1 (mainly meaningful for short-string answers)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict

from UniGSA.core.graph_utils import parse_html_table, table_to_graph_multi
from UniGSA.prompts.multi_table import build_qa_prompt_mtqa
from UniGSA.eval.base import exact_match, try_eval_expr, compute_f1
from UniGSA.inference.predictor import predict_single
from UniGSA.datasets.doclibrag import (
    format_doclibrag_answer,
    load_doclibrag_documents,
    _default_documents_path,
    _shrink_to_fit,
)

logger = logging.getLogger(__name__)


def _to_float(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(str(x).replace(",", "").replace("%", "").strip())
    except (ValueError, TypeError):
        return None


def _numeric_match(pred_str: str, gt_val) -> bool:
    """Return True if pred (string or expr) numerically matches gt_val."""
    gt_num = _to_float(gt_val)
    if gt_num is None:
        return False
    pred_num = _to_float(pred_str)
    if pred_num is None:
        pred_num = try_eval_expr(pred_str)
    if pred_num is None:
        return False
    if gt_num == 0:
        return abs(pred_num) < 1e-6
    if abs(pred_num - gt_num) / max(abs(gt_num), 1e-9) < 1e-3:
        return True
    # percent vs fraction tolerance (10.3 ↔ 0.103)
    if abs(pred_num * 100 - gt_num) < 0.01 or abs(pred_num - gt_num * 100) < 0.01:
        return True
    return False


def run_doclibrag_eval(
    model, tokenizer,
    data_path: str,
    documents_path: str = None,
    output_file: str = "doclibrag_predictions.json",
    max_new_tokens: int = 128,
    max_samples: int = -1,
    device: str = "cpu",
    add_cross_table_edges: bool = True,
    max_length: int = 4096,
    use_reasoning: bool = False,
    **_,
) -> Dict:
    documents_path = documents_path or _default_documents_path(data_path)
    documents = load_doclibrag_documents(documents_path)
    logger.info(f"DocLibRAG: loaded {len(documents)} documents from {documents_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    if 0 < max_samples:
        qa_data = qa_data[:max_samples]
    logger.info(f"DocLibRAG: {len(qa_data)} eval samples from {data_path}")

    results, skipped = [], 0

    for i, item in enumerate(qa_data):
        quid = item.get("question_uid", "")
        question = (item.get("question") or "").strip()
        doc_uid = item.get("document_uid", "")
        gt_raw = item.get("answer", "")
        gt_str = format_doclibrag_answer(gt_raw)

        doc = documents.get(doc_uid)
        if doc is None or not question:
            skipped += 1; continue

        tables_html = doc["tables"]
        paragraphs = list(doc["paragraphs"])
        if not tables_html:
            skipped += 1; continue

        grids = [parse_html_table(html) for html in tables_html]
        grids = [g for g in grids if g and g[0]]
        if not grids:
            skipped += 1; continue

        grids, paragraphs = _shrink_to_fit(
            grids, paragraphs, question, gt_str, tokenizer, max_length
        )

        graph = table_to_graph_multi(grids, add_cross_table_edges=add_cross_table_edges)
        prompt, _ = build_qa_prompt_mtqa(graph, question, paragraphs=paragraphs,
                                         use_reasoning=use_reasoning)
        res = predict_single(model, tokenizer, graph, prompt,
                             max_new_tokens=max_new_tokens, device=device,
                             use_reasoning=use_reasoning)
        if use_reasoning:
            pred, raw = res
        else:
            pred, raw = res, None

        # Metric components
        is_numeric_gt = _to_float(gt_raw) is not None
        num_acc = _numeric_match(pred, gt_raw) if is_numeric_gt else False
        em = num_acc or exact_match(pred, gt_str)
        f1 = compute_f1(pred, gt_str)

        entry = {
            "question_uid": quid, "document_uid": doc_uid,
            "question": question,
            "ground_truth": gt_raw, "ground_truth_str": gt_str,
            "prediction": pred,
            "em": em, "num_acc": num_acc, "f1": round(f1, 4),
            "is_numeric_gt": is_numeric_gt,
        }
        if raw is not None:
            entry["raw_output"] = raw
        results.append(entry)

        if (i + 1) % 50 == 0:
            em_cnt = sum(r["em"] for r in results)
            logger.info(f"  {i+1}/{len(qa_data)} | EM={em_cnt}/{len(results)} "
                        f"({em_cnt/len(results)*100:.1f}%)")

    total = len(results)
    em_cnt = sum(r["em"] for r in results)
    num_total = sum(r["is_numeric_gt"] for r in results)
    num_cnt = sum(r["num_acc"] for r in results)
    f1_avg = sum(r["f1"] for r in results) / total if total else 0.0

    metrics = {
        "em": round(em_cnt / total * 100, 2) if total else 0,
        "num_acc": round(num_cnt / num_total * 100, 2) if num_total else 0,
        "f1": round(f1_avg * 100, 2),
        "em_count": em_cnt, "num_count": num_cnt,
        "num_total": num_total, "total": total,
    }

    logger.info("=" * 60)
    logger.info(f"DocLibRAG: EM={metrics['em']:.2f}% | "
                f"NumAcc={metrics['num_acc']:.2f}% (n={num_total}) | "
                f"F1={metrics['f1']:.2f}% ({total} samples)")
    logger.info("=" * 60)

    out = {"dataset": "DocLibRAG", "metrics": metrics,
           "skipped": skipped, "predictions": results}
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Saved predictions to {output_file}")
    return metrics
