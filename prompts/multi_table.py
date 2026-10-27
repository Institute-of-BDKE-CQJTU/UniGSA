"""Multi-table + text QA prompt (MultiHiertt style).

The graph contains nodes from N tables, separated in serialization by
``[TABLE]`` markers. The prompt also surfaces text paragraphs as in TTQA.
"""

from typing import List, Optional, Tuple

from UniGSA.core.graph_utils import TableGraph
from UniGSA.core.tokenize_utils import serialize_graph_compact


def build_qa_prompt_mtqa(
    graph: TableGraph,
    question: str,
    paragraphs: Optional[List[str]] = None,
    caption: str = "",
    use_reasoning: bool = False,
    **kwargs,
) -> Tuple[str, str]:
    graph_text = serialize_graph_compact(graph)
    cap = f"Caption: {caption}\n\n" if caption else ""

    text_section = ""
    if paragraphs:
        lines = []
        for i, p in enumerate(paragraphs, 1):
            p = (p or "").strip()
            if p:
                lines.append(f"[{i}] {p}")
        if lines:
            text_section = "Text:\n" + "\n".join(lines) + "\n\n"

    n_tables = max(graph.num_tables, 1)
    tail = "Let's think step by step:" if use_reasoning else "Answer:"
    prompt = (
        f"This is a multi-table text-table question answering task. "
        f"You are given {n_tables} table(s) (separated by [TABLE]) and related text. "
        f"Answer the question based on them. If the answer can be directly found in "
        f"the text or table, extract it; if calculation is needed, provide the "
        f"computation expression.\n\n"
        f"{cap}"
        f"Tables:\n/*\n{graph_text}\n*/\n\n"
        f"{text_section}Question: {question}\n{tail}"
    )
    return prompt, graph_text
