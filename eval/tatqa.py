"""TAT-QA evaluator (EM + F1, per-type / per-source breakdown)."""

from __future__ import annotations

import json
import logging
import os
from typing import Dict

from UniGSA.core.graph_utils import table_to_graph_hierarchical
from UniGSA.prompts.text_table import build_qa_prompt_ttqa
from UniGSA.eval.base import finalize_answer, final_answer_exact_match, exact_match, compute_f1
from UniGSA.inference.predictor import predict_single
from UniGSA.datasets.tatqa import format_tatqa_answer, _select_paragraphs_for_text_answer

logger = logging.getLogger(__name__)


def run_tatqa_eval(
    model, tokenizer,
    data_path: str,
    output_file: str = "tatqa_predictions.json",
    max_new_tokens: int = 256,
    max_samples: int = -1,
    max_length: int = 2048,
    device: str = "cpu",
    use_reasoning: bool = False,
    use_gold_text: bool = False,
    **_,
) -> Dict:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"TAT-QA: {len(data)} table-paragraph groups from {data_path}")

    results, skipped, total_processed = [], 0, 0

    for item in data:
        tdata = item.get("table", {})
        grid = tdata.get("table", [])
        if not grid or not grid[0]:
            skipped += 1; continue

        para_raw = item.get("paragraphs", [])
        all_paras = [p.get("text", "") for p in
                     sorted(para_raw, key=lambda x: x.get("order", 0))]

        for q_item in item.get("questions", []):
            if 0 < max_samples <= total_processed:
                break

            question = q_item.get("question", "")
            gt = format_tatqa_answer(q_item)
            answer_type = q_item.get("answer_type", "")
            answer_from = q_item.get("answer_from", "")
            if not question:
                skipped += 1; continue

            if use_gold_text:
                rel_para_ids = q_item.get("rel_paragraphs", [])
                if rel_para_ids:
                    paras = []
                    for pid in rel_para_ids:
                        try:
                            pid_int = int(pid)
                        except (ValueError, TypeError):
                            pid_int = pid
                        for p in para_raw:
                            if p.get("order") == pid_int or p.get("order") == pid:
                                paras.append(p.get("text", ""))
                                break
                else:
                    if answer_from == "text" and answer_type in ("span", "multi-span"):
                        paras = _select_paragraphs_for_text_answer(gt, all_paras)
                    else:
                        paras = all_paras
            else:
                paras = all_paras

            graph = table_to_graph_hierarchical(grid, 1, 0)
            prompt, _ = build_qa_prompt_ttqa(graph, question, paragraphs=paras,
                                             use_reasoning=use_reasoning)
            res = predict_single(model, tokenizer, graph, prompt,
                                 max_new_tokens=max_new_tokens, device=device,
                             max_length=max_length, postprocess=False,
                                 use_reasoning=use_reasoning)
            if use_reasoning:
                pred, raw = res
            else:
                pred, raw = res, None

            gold_final = format_tatqa_answer(dict(q_item, derivation=""))
            em = final_answer_exact_match(pred, gold_final)
            f1 = compute_f1(finalize_answer(pred), finalize_answer(gold_final))
            entry = {
                "uid": q_item.get("uid", ""), "question": question,
                "ground_truth": gt, "prediction": pred, "final_answer": finalize_answer(pred),
                "answer_type": answer_type, "answer_from": answer_from,
                "exact_match": em, "f1": round(f1, 4),
            }
            if raw is not None:
                entry["raw_output"] = raw
            results.append(entry)
            total_processed += 1

            if total_processed % 50 == 0:
                em_cnt = sum(r["exact_match"] for r in results)
                avg_f1 = sum(r["f1"] for r in results) / len(results)
                logger.info(f"  Processed {total_processed} | EM={em_cnt}/{len(results)} "
                            f"({em_cnt/len(results)*100:.1f}%) | F1={avg_f1:.4f}")
        if 0 < max_samples <= total_processed:
            break

    total = len(results)
    em_count = sum(r["exact_match"] for r in results)
    avg_f1 = sum(r["f1"] for r in results) / total if total else 0

    type_metrics = {}
    for at in ["span", "multi-span", "arithmetic", "count"]:
        sub = [r for r in results if r["answer_type"] == at]
        if sub:
            type_metrics[at] = {
                "em": round(sum(r["exact_match"] for r in sub) / len(sub) * 100, 2),
                "f1": round(sum(r["f1"] for r in sub) / len(sub), 4),
                "count": len(sub),
            }

    source_metrics = {}
    for af in ["table", "text", "table-text"]:
        sub = [r for r in results if r["answer_from"] == af]
        if sub:
            source_metrics[af] = {
                "em": round(sum(r["exact_match"] for r in sub) / len(sub) * 100, 2),
                "f1": round(sum(r["f1"] for r in sub) / len(sub), 4),
                "count": len(sub),
            }

    metrics = {
        "em": round(em_count / total * 100, 2) if total else 0,
        "f1": round(avg_f1, 4),
        "em_count": em_count, "total": total,
        "per_type": type_metrics, "per_source": source_metrics,
    }

    logger.info("=" * 60)
    logger.info(f"TAT-QA Results: EM={metrics['em']:.2f}% | F1={metrics['f1']:.4f} "
                f"({em_count}/{total})")
    for at, tm in type_metrics.items():
        logger.info(f"  {at}: EM={tm['em']:.2f}%, F1={tm['f1']:.4f} (n={tm['count']})")
    for af, sm in source_metrics.items():
        logger.info(f"  {af}: EM={sm['em']:.2f}%, F1={sm['f1']:.4f} (n={sm['count']})")
    logger.info("=" * 60)

    out = {"dataset": "TAT-QA", "metrics": metrics,
           "skipped": skipped, "predictions": results}
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved predictions to {output_file}")
    return metrics
