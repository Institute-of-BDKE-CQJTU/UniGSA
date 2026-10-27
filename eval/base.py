"""Common metric helpers used by all evaluators."""

from __future__ import annotations

import re
from collections import Counter
from typing import List, Optional


# ---- normalization ----

def normalize(s: str) -> str:
    """Robust string normalization for span-style answers."""
    s = s.strip().lower()
    s = re.sub(r"[\$€£¥]", "", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s*%", "%", s)
    s = " ".join(s.split())
    s = s.rstrip(".,;:")
    try:
        v = float(s)
        s = str(int(v)) if v == int(v) else str(v)
    except (ValueError, OverflowError):
        pass
    return s.strip()


def strip_parens(s: str) -> str:
    return s.replace("(", "").replace(")", "")


def normalize_expr(s: str) -> str:
    return re.sub(r"\s+", "", normalize(s))


# ---- expression evaluation ----

def try_eval_expr(expr: str) -> Optional[float]:
    """Parse the ENTIRE string as arithmetic; never strip arbitrary words.

    Supports numeric units/financial formatting and the arithmetic operators
    emitted by the dataset program converter. No calls, names or code eval.
    A standalone percentage keeps the benchmark's displayed numeric value.
    """
    import ast
    import math
    import operator
    if not expr or len(str(expr)) > 4096:
        return None
    text = str(expr).strip()
    factor = 1
    for unit, value in ((" thousand", 1000), (" million", 1000000), (" billion", 1000000000)):
        if text.lower().endswith(unit):
            text, factor = text[:-len(unit)].strip(), value
            break
    text = text.replace("×", "*").replace("÷", "/").replace("\\times", "*").replace("^", "**")
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    text = re.sub(r"[$€£¥](?=\s*\d)", "", text).strip()
    text = re.sub(r"(?<=\d)%", "", text)
    operations = {ast.Add: operator.add, ast.Sub: operator.sub,
                  ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow}
    def visit(node, depth=0):
        if depth > 64:
            raise ValueError("Expression too deep")
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            result = float(node.value)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            result = visit(node.operand, depth + 1) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and type(node.op) in operations:
            left, right = visit(node.left, depth + 1), visit(node.right, depth + 1)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Exponent out of range")
            result = operations[type(node.op)](left, right)
        else:
            raise ValueError("Not a pure arithmetic expression")
        if not isinstance(result, (int, float)) or not math.isfinite(result):
            raise ValueError("Nonfinite result")
        return result
    try:
        result = visit(ast.parse(text, mode="eval").body) * factor
        return result if math.isfinite(result) else None
    except (SyntaxError, ValueError, TypeError, OverflowError, ZeroDivisionError, RecursionError):
        return None


def finalize_answer(prediction: str) -> str:
    """Paper task formulation: execute when parseable, otherwise normalize."""
    value = try_eval_expr(prediction)
    if value is None:
        return normalize(str(prediction))
    return "0" if value == 0 else format(value, ".15g")


def final_answer_exact_match(prediction: str, gold) -> bool:
    if isinstance(gold, list):
        gold = ", ".join(str(value) for value in gold)
    return finalize_answer(str(prediction)) == finalize_answer(str(gold))


# ---- matching ----

def _normalize_elements(s: str) -> str:
    """Normalize each comma-separated element individually.
    Handles multi-element answers like '51.0, 8189.0' → '51, 8189'."""
    if ", " not in s:
        return normalize(s)
    parts = [normalize(p.strip()) for p in s.split(", ")]
    return ", ".join(parts)


def exact_match(pred: str, gt: str, answer_type: str = "") -> bool:
    if normalize(pred) == normalize(gt):
        return True
    if normalize(strip_parens(pred)) == normalize(strip_parens(gt)):
        return True
    # Element-wise normalization for multi-element answers (e.g., "51.0, 8189.0")
    if ", " in gt or ", " in pred:
        if _normalize_elements(pred) == _normalize_elements(gt):
            return True
        if _normalize_elements(strip_parens(pred)) == _normalize_elements(strip_parens(gt)):
            return True
    if answer_type == "arithmetic":
        if normalize_expr(pred) == normalize_expr(gt):
            return True
        if normalize_expr(strip_parens(pred)) == normalize_expr(strip_parens(gt)):
            return True
        pv, gv = try_eval_expr(pred), try_eval_expr(gt)
        if pv is not None and gv is not None:
            if gv == 0:
                if abs(pv) < 1e-6:
                    return True
            elif abs(pv - gv) / max(abs(gv), 1e-9) < 1e-3:
                return True
            if abs(pv * 100 - gv) < 0.01 or abs(pv - gv * 100) < 0.01:
                return True
    return False


# ---- F1 ----

def _tokenize_for_f1(s: str, answer_type: str = "") -> List[str]:
    s = normalize(s)
    if answer_type == "arithmetic":
        s = re.sub(r"([+\-*/()])", r" \1 ", s)
    s = strip_parens(s)
    return " ".join(s.split()).split()


def compute_f1(pred: str, gt: str, answer_type: str = "") -> float:
    if answer_type == "arithmetic":
        pv, gv = try_eval_expr(pred), try_eval_expr(gt)
        if pv is not None and gv is not None:
            matched = False
            if gv == 0:
                matched = abs(pv) < 1e-6
            elif abs(pv - gv) / max(abs(gv), 1e-9) < 1e-3:
                matched = True
            if not matched and (abs(pv * 100 - gv) < 0.01 or abs(pv - gv * 100) < 0.01):
                matched = True
            return 1.0 if matched else 0.0

    pt = _tokenize_for_f1(pred, answer_type)
    gtok = _tokenize_for_f1(gt, answer_type)
    if not pt and not gtok:
        return 1.0
    if not pt or not gtok:
        return 0.0
    common = Counter(pt) & Counter(gtok)
    same = sum(common.values())
    if same == 0:
        return 0.0
    p = same / len(pt)
    r = same / len(gtok)
    return 2 * p * r / (p + r)
