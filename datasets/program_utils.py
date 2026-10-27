"""
Program → math expression converter shared by FinQA and MultiHiertt.

FinQA programs are flat or nested calls without step references:
    "subtract(5829, 5735)"
    "divide(subtract(153.7, 139.9), 139.9)"

MultiHiertt programs are sequences with step references (#0, #1, ...):
    "subtract(1733,1912), divide(#0,1912), add(const_1,#1), multiply(#2,1733)"

Both styles are reduced to a single standard math expression string.
"""

from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)


# FinQA / MultiHiertt constants
PROG_CONSTANTS = {
    "const_1": "1", "const_2": "2", "const_3": "3", "const_4": "4",
    "const_5": "5", "const_6": "6", "const_10": "10", "const_12": "12",
    "const_100": "100", "const_1000": "1000",
    "const_1000000": "1000000", "const_m1": "-1",
}

OP_MAP = {
    "add": "+", "subtract": "-", "multiply": "*", "divide": "/",
    "exp": "^", "greater": ">",
}

TABLE_OPS = {"table_average", "table_sum", "table_max", "table_min"}


def _split_top_level_args(s: str) -> List[str]:
    args, depth, current = [], 0, []
    for ch in s:
        if ch == "(":
            depth += 1; current.append(ch)
        elif ch == ")":
            depth -= 1; current.append(ch)
        elif ch == "," and depth == 0:
            args.append("".join(current).strip()); current = []
        else:
            current.append(ch)
    if current:
        args.append("".join(current).strip())
    return args


def _parse_call(call_str: str):
    """Parse a single call like 'subtract(1, 2)' → ('subtract', '1', '2').
    Nested calls become nested tuples."""
    s = call_str.strip()
    if not s:
        return s
    paren_start = s.find("(")
    if paren_start == -1:
        return s
    op = s[:paren_start].strip()
    inner = s[paren_start + 1:]
    if inner.endswith(")"):
        inner = inner[:-1]
    else:
        inner = inner.rstrip(")")
    args = _split_top_level_args(inner)
    parsed_args = [_parse_call(a) for a in args]
    return (op, *parsed_args)


def _resolve_arg(val):
    if isinstance(val, tuple):
        return val
    s = str(val).strip()
    return PROG_CONSTANTS.get(s, s)


def _ast_to_expr(ast) -> str:
    if isinstance(ast, str):
        return _resolve_arg(ast)
    if not isinstance(ast, tuple) or len(ast) < 2:
        return str(ast)

    op = ast[0]

    if op in TABLE_OPS:
        friendly = op.replace("table_", "")
        arg_str = _ast_to_expr(ast[1]) if len(ast) > 1 else ""
        return f"{friendly}({arg_str})"

    if op in OP_MAP and len(ast) == 3:
        sym = OP_MAP[op]
        left = _ast_to_expr(ast[1])
        right = _ast_to_expr(ast[2])
        if isinstance(ast[1], tuple) and len(ast[1]) >= 3:
            left = f"({left})"
        if isinstance(ast[2], tuple) and len(ast[2]) >= 3:
            right = f"({right})"
        return f"{left} {sym} {right}"

    return f"{op}({', '.join(_ast_to_expr(a) for a in ast[1:])})"


def _resolve_step_refs(ast, step_results: List):
    """Replace #N tokens (anywhere in the AST) with step_results[N]."""
    if isinstance(ast, str):
        s = ast.strip()
        if s.startswith("#"):
            try:
                idx = int(s[1:])
                if 0 <= idx < len(step_results):
                    return step_results[idx]
            except ValueError:
                pass
        return ast
    if isinstance(ast, tuple):
        return tuple([ast[0]] + [_resolve_step_refs(a, step_results) for a in ast[1:]])
    return ast


def convert_program_to_expr(program: str) -> str:
    """Convert a FinQA / MultiHiertt program string to a math expression.

    Supports both single-call and multi-step (#N) programs.
    Returns the expression of the LAST step.
    """
    if not program or not program.strip():
        return ""

    try:
        steps = _split_top_level_args(program)
        step_asts: List = []
        for step in steps:
            step = step.strip()
            if not step:
                continue
            ast = _parse_call(step)
            ast = _resolve_step_refs(ast, step_asts)
            step_asts.append(ast)
        if not step_asts:
            return program
        return _ast_to_expr(step_asts[-1])
    except Exception as e:
        logger.warning(f"Failed to parse program '{program}': {e}")
        return program
