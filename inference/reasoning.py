"""Extract final answer from reasoning-augmented model output."""

from __future__ import annotations

import re


def extract_answer_from_reasoning(text: str) -> str:
    """Try several markers used by the training answer builders.

    Order:
      1. "The extracted answer is: ..."  (TTQA span / multi-span)
      2. "So the final answer is ..."    (default / arithmetic / hitab)
      3. fallback: full text
    """
    m = re.search(r"[Tt]he extracted answer is:\s*(.+?)$", text, re.DOTALL)
    if m:
        return m.group(1).strip().rstrip(".,;")
    m = re.search(r"[Ss]o the final answer is\s*(.+?)$", text, re.DOTALL)
    if m:
        return m.group(1).strip().rstrip(".,;")
    return text.strip()
