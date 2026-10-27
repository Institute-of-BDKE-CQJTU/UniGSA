"""FinQA evaluator (Expr EM + Execution Accuracy)."""

from __future__ import annotations

import json
import logging
import os
from typing import Dict

from UniGSA.core.graph_utils import table_to_graph_hierarchical
from UniGSA.prompts.text_table import build_qa_prompt_ttqa
from UniGSA.eval.base import finalize_answer, final_answer_exact_match, exact_match, try_eval_expr
from UniGSA.inference.predictor import predict_single
from UniGSA.datasets.finqa import format_finqa_answer, _extract_finqa_gold_text

logger = logging.getLogger(__name__)


def run_finqa_eval(
    model, tokenizer,
    data_path: str,
    output_file: str = "finqa_predictions.json",
    max_new_tokens: int = 256,
    max_samples: int = -1,
    max_length: int = 2048,
    device: str = "cpu",
    use_gold_text: bool = False,
    use_reasoning: bool = False,
    **_,
) -> Dict:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if 0 < max_samples:
        data = data[:max_samples]
    logger.info(f"FinQA: {len(data)} samples from {data_path}")

    results, skipped = [], 0

    for i, item in enumerate(data):
        grid = item.get("table", [])
        if not grid or not grid[0]:
            skipped += 1; continue

        pre_text = item.get("pre_text", [])
        post_text = item.get("post_text", [])
        qa = item.get("qa", {})
        question = qa.get("question", "")
        gt_expr = format_finqa_answer(qa)
        exe_ans = qa.get("exe_ans", "")
        program_re = qa.get("program_re", "") or qa.get("program", "")
        if not question:
            skipped += 1; continue

        if use_gold_text:
            paras = _extract_finqa_gold_text(pre_text, post_text, qa.get("gold_inds", {}))
        else:
            paras = pre_text + post_text

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

        expr_em = exact_match(pred, gt_expr)
        pred_exe = try_eval_expr(pred)

        gt_exe = None
        if exe_ans != "":
            try:
                gt_exe = float(exe_ans)
            except (ValueError, TypeError):
                gt_exe = None

        exe_acc = final_answer_exact_match(pred, exe_ans if exe_ans != "" else gt_expr)

        entry = {
            "id": item.get("id", ""), "question": question,
            "ground_truth_expr": gt_expr, "program_re": program_re,
            "prediction": pred, "final_answer": finalize_answer(pred), "exe_ans": exe_ans,
            "pred_exe_ans": pred_exe,
            "expr_em": expr_em, "exe_acc": exe_acc,
        }
        if raw is not None:
            entry["raw_output"] = raw
        results.append(entry)

        if (i + 1) % 50 == 0:
            em_cnt = sum(r["expr_em"] for r in results)
            exe_cnt = sum(r["exe_acc"] for r in results)
            logger.info(f"  {i+1}/{len(data)} | Expr EM={em_cnt}/{len(results)} "
                        f"({em_cnt/len(results)*100:.1f}%) | "
                        f"Exe Acc={exe_cnt}/{len(results)} ({exe_cnt/len(results)*100:.1f}%)")

    total = len(results)
    em_cnt = sum(r["expr_em"] for r in results)
    exe_cnt = sum(r["exe_acc"] for r in results)
    metrics = {
        "em": round(exe_cnt / total * 100, 2) if total else 0,
        "expr_em": round(em_cnt / total * 100, 2) if total else 0,
        "exe_acc": round(exe_cnt / total * 100, 2) if total else 0,
        "expr_em_count": em_cnt, "exe_acc_count": exe_cnt, "total": total,
    }

    logger.info("=" * 60)
    logger.info(f"FinQA Results: Expr EM={metrics['expr_em']:.2f}% | "
                f"Exe Acc={metrics['exe_acc']:.2f}% ({total} samples)")
    logger.info("=" * 60)

    out = {"dataset": "FinQA", "metrics": metrics,
           "skipped": skipped, "predictions": results}
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved predictions to {output_file}")
    return metrics
