"""Text-Table QA prompt (TAT-QA / FinQA style)."""

from typing import List, Optional, Tuple

from UniGSA.core.graph_utils import TableGraph
from UniGSA.core.tokenize_utils import serialize_graph_compact


def build_qa_prompt_ttqa(
    graph: TableGraph,
    question: str,
    paragraphs: Optional[List[str]] = None,
    caption: str = "",
    use_reasoning: bool = False,
    **kwargs,
) -> Tuple[str, str]:
    """Build a prompt that contains text paragraphs + table graph + question.

    Instructs the model to extract spans for text-sourced answers and to emit
    a computation expression for arithmetic answers.
    """
    graph_text = serialize_graph_compact(graph)
    cap = f"Table Caption: {caption}\n\n" if caption else ""

    text_section = ""
    if paragraphs:
        lines = []
        for i, p in enumerate(paragraphs, 1):
            p = (p or "").strip()
            if p:
                lines.append(f"[{i}] {p}")
        if lines:
            text_section = "Text:\n" + "\n".join(lines) + "\n\n"

    tail = "Let's think step by step:" if use_reasoning else "Answer:"
    prompt = (
        f"This is a text-table question answering task. Based on the table and "
        f"related text, answer the following question. If the answer can be directly "
        f"found in the text, extract the relevant text span. If calculation is needed, "
        f"provide the computation expression.\n\n"
        f"{cap}"
        f"Graph:\n/*\n{graph_text}\n*/\n\n"
        f"{text_section}Question: {question}\n{tail}"
    )
    return prompt, graph_text
