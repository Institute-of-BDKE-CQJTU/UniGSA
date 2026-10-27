"""Prompt builders. Each returns ``(prompt: str, graph_text: str)``.

All builders share the same first parameter ``graph`` (a TableGraph) so that
generic helpers (e.g. ``split_table_for_max_length``) can pass through one of
them without knowing which dataset is being processed.
"""

from UniGSA.prompts.table_only import build_qa_prompt
from UniGSA.prompts.text_table import build_qa_prompt_ttqa
from UniGSA.prompts.multi_table import build_qa_prompt_mtqa

__all__ = ["build_qa_prompt", "build_qa_prompt_ttqa", "build_qa_prompt_mtqa"]
