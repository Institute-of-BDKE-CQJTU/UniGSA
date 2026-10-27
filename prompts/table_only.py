"""Single-table QA prompt (HiTab style)."""

from typing import List, Optional, Tuple

from UniGSA.core.graph_utils import TableGraph
from UniGSA.core.tokenize_utils import serialize_graph_compact


def build_qa_prompt(
    graph: TableGraph,
    question: str,
    caption: str = "",
    use_reasoning: bool = False,
    enhanced: bool = False,
    **kwargs,  # tolerate extra kwargs from generic callers
) -> Tuple[str, str]:
    graph_text = serialize_graph_compact(graph)
    cap = f"Table Caption: {caption}\n" if caption else ""

    if enhanced and graph.nodes:
        # Enhanced prompt: add header structure summary for better cell localization
        header_info = _build_header_summary(graph)
        header_section = f"Table Structure:\n{header_info}\n\n" if header_info else ""
        tail = "Let's think step by step:" if use_reasoning else "The answer is:"
        # Direction-aware hints to mitigate "opposite-of-ground-truth" errors
        # observed on HiTab (see data_check/OPPOSITE_PRED_REPORT.md):
        #  - operand-order flips in subtraction/division
        #  - extremum (highest/lowest) and two-option selection reversed
        #  - percentage complement (answering the leftover category)
        direction_hints = (
            "Important direction rules:\n"
            "1. Read the direction words in the question carefully "
            "(e.g. more/less, higher/lower, increase/decrease, "
            "from A to B, change from X to Y, faster/slower, "
            "earlier/later, largest/smallest, highest/lowest, top/bottom).\n"
            "2. For subtraction/division, the operand order must follow "
            "the question's direction. For 'change/difference/increase "
            "from A to B' compute B - A; for 'decrease from A to B' the "
            "magnitude is A - B; for 'ratio/proportion of A to B' or "
            "'percentage of A among B' compute A / B (not B / A).\n"
            "3. For 'which ... has the highest/lowest/largest/smallest' "
            "or 'A or B' questions, first list the candidate values from "
            "the table, then pick the one matching the requested extreme; "
            "do NOT default to the entity mentioned first.\n"
            "4. For 'what is the percentage of X' questions, answer the "
            "percentage of X itself, not the complement (100 - X).\n"
            "5. If the answer requires multiple cells, list them in the "
            "order they appear in the table, separated by commas.\n\n"
        )
        prompt = (
            f"This is a hierarchical table question answering task. "
            f"The table has multi-level row and column headers. "
            f"Find the correct cell(s) and provide the answer. "
            f"If calculation is needed, compute the result.\n\n"
            f"{direction_hints}"
            f"{cap}{header_section}"
            f"Graph:\n/*\n{graph_text}\n*/\n\n"
            f"Question: {question}\n{tail}"
        )
    else:
        tail = "Let's think step by step:" if use_reasoning else "The answer is:"
        prompt = (
            f"This is a table question answering task. Answer the given question "
            f"based on the table represented as a graph.\n\n"
            f"{cap}Graph:\n/*\n{graph_text}\n*/\n\n"
            f"Question: {question}\n{tail}"
        )
    return prompt, graph_text


def _build_header_summary(graph: TableGraph) -> str:
    """Extract top-level and left-level header values for the enhanced prompt."""
    if not graph.nodes:
        return ""

    top_depth = graph.top_depth
    left_depth = graph.left_depth
    if top_depth == 0 and left_depth == 0:
        return ""

    parts = []

    # Collect unique top header values (level by level)
    if top_depth > 0:
        for d in range(top_depth):
            vals = set()
            for n in graph.nodes:
                if n.table_id == 0 and n.row == d and n.col >= left_depth:
                    v = n.value.strip()
                    if v:
                        vals.add(v)
            if vals:
                # Limit to avoid overly long headers
                display_vals = sorted(vals)[:15]
                suffix = f" (+{len(vals)-15} more)" if len(vals) > 15 else ""
                parts.append(f"  Column headers (level {d+1}): {', '.join(display_vals)}{suffix}")

    # Collect unique left header values (level by level)
    if left_depth > 0:
        for d in range(left_depth):
            vals = set()
            for n in graph.nodes:
                if n.table_id == 0 and n.col == d and n.row >= top_depth:
                    v = n.value.strip()
                    if v:
                        vals.add(v)
            if vals:
                display_vals = sorted(vals)[:15]
                suffix = f" (+{len(vals)-15} more)" if len(vals) > 15 else ""
                parts.append(f"  Row headers (level {d+1}): {', '.join(display_vals)}{suffix}")

    return "\n".join(parts)
