"""HiTab: prediction-only execution followed by strict normalized final-answer EM."""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Dict, List, Optional

from UniGSA.core.graph_utils import (
    load_hitab_tables_jsonl, reconstruct_hitab_from_kg,
    table_to_graph_hierarchical,
)
from UniGSA.prompts.table_only import build_qa_prompt
from UniGSA.eval.base import finalize_answer, final_answer_exact_match, exact_match, try_eval_expr
from UniGSA.inference.predictor import predict_single

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Main eval
# ---------------------------------------------------------------------------

def run_hitab_eval(
    model, tokenizer,
    hitab_dir: str,
    split: str = "test",
    output_file: str = "hitab_predictions.json",
    max_new_tokens: int = 256,
    max_samples: int = -1,
    max_length: int = 2048,
    device: str = "cpu",
    use_reasoning: bool = False,
    enhanced_prompt: bool = False,
    **_,
) -> Dict:
    tables = load_hitab_tables_jsonl(os.path.join(hitab_dir, "tables.jsonl"))
    logger.info(f"HiTab: loaded {len(tables)} tables (use_reasoning={use_reasoning})")

    sp = os.path.join(hitab_dir, f"{split}_samples_processed.jsonl")
    if not os.path.exists(sp):
        sp = os.path.join(hitab_dir, f"{split}_samples.jsonl")
    samples = []
    with open(sp) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if 0 < max_samples:
        samples = samples[:max_samples]
    logger.info(f"HiTab: evaluating {len(samples)} samples ({split})")

    results, skipped = [], 0
    for i, item in enumerate(samples):
        tname = str(item.get("context", ""))
        question = item.get("question", "")
        ans = item.get("answer", "")
        # Normalize answer elements consistently with training targets:
        # 51.0 → "51", 52.1 → "52.1"
        def _norm_elem(v):
            if isinstance(v, float) and v == int(v):
                return str(int(v))
            return str(v)
        gt_answer_str = (", ".join(_norm_elem(a) for a in ans)
                         if isinstance(ans, list) else str(ans))
        gold_expr = item.get("expression") or ""
        answer_type = item.get("answer_type", "")

        if tname not in tables:
            skipped += 1; continue
        kg = tables[tname].get("kg", {})
        grid, title, top_depth, left_depth = reconstruct_hitab_from_kg(kg)
        if not grid or not grid[0]:
            skipped += 1; continue

        graph = table_to_graph_hierarchical(grid, top_depth, left_depth)
        prompt, _ = build_qa_prompt(graph, question, caption=title,
                                    use_reasoning=use_reasoning,
                                    enhanced=enhanced_prompt)
        result = predict_single(model, tokenizer, graph, prompt,
                                max_new_tokens=max_new_tokens, device=device,
                             max_length=max_length, postprocess=False,
                                use_reasoning=use_reasoning)
        if use_reasoning:
            pred, raw = result
        else:
            pred, raw = result, None

        # Prediction postprocessing is independent of gold answer_type.
        pred_expr = pred
        pred_val = try_eval_expr(pred)
        is_arith = answer_type == "arithmetic"  # reporting only
        expr_em = exact_match(pred, gold_expr) if gold_expr else False
        em = final_answer_exact_match(pred, gt_answer_str)
        rescored = False

        entry = {
            "id": item.get("id", ""),
            "question": question,
            "answer_type": answer_type,
            "ground_truth": gt_answer_str,
            "gold_expression": gold_expr,
            "prediction": pred, "final_answer": finalize_answer(pred),
            "pred_expression": pred_expr if is_arith else None,
            "pred_value": pred_val,
            "expr_em": expr_em,
            "exact_match": em,
            "rescored": rescored,
        }
        if raw is not None:
            entry["raw_output"] = raw
        results.append(entry)

        if (i + 1) % 20 == 0:
            cnt = sum(r["exact_match"] for r in results)
            logger.info(f"  {i+1}/{len(samples)} EM={cnt}/{len(results)} "
                        f"({cnt/len(results)*100:.1f}%)")

    total = len(results)
    em_cnt = sum(r["exact_match"] for r in results)
    expr_em_cnt = sum(r["expr_em"] for r in results)
    arith_total = sum(1 for r in results if r["answer_type"] == "arithmetic")
    arith_em = sum(1 for r in results
                   if r["answer_type"] == "arithmetic" and r["exact_match"])
    nonarith_total = total - arith_total
    nonarith_em = em_cnt - arith_em
    rescored_cnt = sum(1 for r in results if r.get("rescored"))

    metrics = {
        "em": round(em_cnt / total * 100, 2) if total else 0,
        "expr_em": round(expr_em_cnt / arith_total * 100, 2) if arith_total else 0,
        "arith_exe_acc": round(arith_em / arith_total * 100, 2) if arith_total else 0,
        "nonarith_em": round(nonarith_em / nonarith_total * 100, 2) if nonarith_total else 0,
        "em_count": em_cnt, "total": total,
        "arith_total": arith_total, "nonarith_total": nonarith_total,
        "rescored_fixed": rescored_cnt,
    }
    logger.info("=" * 60)
    logger.info(f"HiTab {split}: overall EM = {em_cnt}/{total} = "
                f"{metrics['em']:.2f}%  (rescored fixes = {rescored_cnt})")
    logger.info(f"  arith exe-acc = {arith_em}/{arith_total} = "
                f"{metrics['arith_exe_acc']:.2f}% | "
                f"expr EM = {expr_em_cnt}/{arith_total} = "
                f"{metrics['expr_em']:.2f}%")
    logger.info(f"  non-arith EM  = {nonarith_em}/{nonarith_total} = "
                f"{metrics['nonarith_em']:.2f}%")
    logger.info("=" * 60)

    out = {"dataset": "HiTab", "metrics": metrics, "split": split,
           "skipped": skipped, "use_reasoning": use_reasoning,
           "predictions": results,
           "rescore_note": (
               "Non-arithmetic samples are matched with tolerant strategies "
               "(expression execution, order-insensitive list match, "
               "percent-rounding match). Samples flagged via these strategies "
               "have `rescored=true`."
           )}
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved predictions to {output_file}")
    return metrics
