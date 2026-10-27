"""
DocLibRAG dataset loader (multi-table + text QA over a document library).

Directory layout (TTQA_dataset/DocLibRAG/):
    Documents.json     # list[ {uid, paragraphs: [str, ...], tables: [HTML_str, ...]} ]
    train_qa.json      # list[ {question_uid, question, answer, document_uid} ]
    dev_qa.json
    test_qa.json

QA item:
    {
      "question_uid": str,
      "question": str,
      "answer": float | int | str,     # mostly numeric, sometimes "yes"/"no"/"<short str>"
      "document_uid": str              # links to Documents[*].uid
    }

Training answer:
    The raw answer is converted to a clean string target (floats like 53369.0 → "53369").

Pipeline (mirrors MultiHiertt):  parse HTML → grids → multi-table graph →
text+table prompt → length-aware shrinking → tokenize_sample.

Optimizations:
  - Documents are parsed into grids ONCE per uid and cached (many QA items
    share the same document, so this saves a lot of BeautifulSoup work).
  - Per-QA work (length shrinking + graph build + tokenization) runs through
    a thread pool — HuggingFace fast tokenizers release the GIL so threads
    give near-linear speedup without the pickling overhead of processes.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from threading import Lock
from typing import Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizer

from UniGSA.core.graph_utils import parse_html_table, table_to_graph_multi
from UniGSA.core.tokenize_utils import (
    tokenize_sample, estimate_token_length,
    load_reasoning_jsonl, build_reasoning_answer,
)
from UniGSA.prompts.multi_table import build_qa_prompt_mtqa
from UniGSA.datasets.base import DatasetSpec, register_dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Answer formatting
# ---------------------------------------------------------------------------

def format_doclibrag_answer(answer) -> str:
    """Convert raw DocLibRAG answer to a clean string target."""
    if isinstance(answer, bool):
        return "yes" if answer else "no"
    if isinstance(answer, int):
        return str(answer)
    if isinstance(answer, float):
        return str(int(answer)) if answer == int(answer) else str(answer)
    if isinstance(answer, list):
        return ", ".join(format_doclibrag_answer(a) for a in answer)
    return str(answer).strip()


# ---------------------------------------------------------------------------
# Document library
# ---------------------------------------------------------------------------

def load_doclibrag_documents(documents_path: str) -> Dict[str, Dict]:
    """Load Documents.json once and return a {uid: doc} mapping."""
    with open(documents_path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    out: Dict[str, Dict] = {}
    for d in docs:
        uid = d.get("uid")
        if not uid:
            continue
        out[uid] = {
            "paragraphs": d.get("paragraphs", []) or [],
            "tables": d.get("tables", []) or [],
        }
    return out


def _default_documents_path(qa_path: str) -> str:
    """Derive Documents.json path from the QA file path (same directory)."""
    return os.path.join(os.path.dirname(qa_path), "Documents.json")


def _parse_doc_grids(doc: Dict) -> List[List[List[str]]]:
    """Parse the HTML tables of a single document into non-empty grids."""
    grids = [parse_html_table(html) for html in doc.get("tables", [])]
    return [g for g in grids if g and g[0]]


def _build_doc_grid_cache(
    documents: Dict[str, Dict],
    needed_uids: set,
    num_workers: int,
) -> Dict[str, List[List[List[str]]]]:
    """Parse HTML tables for every needed document, in parallel."""
    cache: Dict[str, List[List[List[str]]]] = {}
    if not needed_uids:
        return cache
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = {
            ex.submit(_parse_doc_grids, documents[uid]): uid
            for uid in needed_uids
            if uid in documents
        }
        for fut in as_completed(futures):
            uid = futures[fut]
            try:
                cache[uid] = fut.result()
            except Exception as e:  # pragma: no cover
                logger.warning(f"DocLibRAG: failed to parse tables for {uid}: {e}")
                cache[uid] = []
    return cache


# ---------------------------------------------------------------------------
# Length-aware shrinking (identical strategy as MultiHiertt)
# ---------------------------------------------------------------------------

def _shrink_to_fit(
    grids: List[List[List[str]]],
    paragraphs: List[str],
    question: str,
    answer: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> Tuple[List[List[List[str]]], List[str]]:
    """Greedily drop paragraphs / largest tables until the prompt fits."""
    cur_grids = list(grids)
    cur_paras = list(paragraphs)

    def _est():
        graph = table_to_graph_multi(cur_grids, add_cross_table_edges=False)
        prompt, _ = build_qa_prompt_mtqa(graph, question, paragraphs=cur_paras)
        return estimate_token_length(prompt, answer, tokenizer)

    if _est() <= max_length:
        return cur_grids, cur_paras

    while cur_paras and _est() > max_length:
        cur_paras.pop()

    while len(cur_grids) > 1 and _est() > max_length:
        sizes = [(sum(len(r) for r in g), i) for i, g in enumerate(cur_grids)]
        _, drop_i = max(sizes)
        cur_grids.pop(drop_i)

    if _est() > max_length:
        # Truncate long cell strings (last resort)
        for g in cur_grids:
            for r in g:
                for ci in range(len(r)):
                    if len(r[ci]) > 80:
                        r[ci] = r[ci][:80]

    return cur_grids, cur_paras


# ---------------------------------------------------------------------------
# Per-sample worker
# ---------------------------------------------------------------------------

def _process_one_qa(
    item: Dict,
    documents: Dict[str, Dict],
    grid_cache: Dict[str, List[List[List[str]]]],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    add_cross_table_edges: bool,
    reasoning_map: Optional[Dict[str, Dict]],
    use_reasoning: bool,
) -> Tuple[Optional[Dict], str]:
    """Process a single QA item.

    Returns:
        (sample_dict_or_None, status_tag)  where status_tag is one of
        {'ok', 'empty', 'no_doc', 'no_reasoning'}.
    """
    quid = item.get("question_uid", "")
    question = (item.get("question") or "").strip()
    doc_uid = item.get("document_uid", "")
    ans_raw = item.get("answer", "")
    answer = format_doclibrag_answer(ans_raw)

    if not question or answer == "":
        return None, "empty"

    doc = documents.get(doc_uid)
    if doc is None:
        return None, "no_doc"

    # Grids come from the shared cache. Deep-copy because _shrink_to_fit may
    # mutate the cells in its last-resort branch.
    base_grids = grid_cache.get(doc_uid)
    if not base_grids:
        return None, "empty"
    grids = deepcopy(base_grids)
    paragraphs = list(doc["paragraphs"])

    if use_reasoning:
        if not reasoning_map or quid not in reasoning_map:
            return None, "no_reasoning"
        answer = build_reasoning_answer(reasoning_map[quid]["reasoning"], answer)

    grids, paragraphs = _shrink_to_fit(
        grids, paragraphs, question, answer, tokenizer, max_length
    )

    graph = table_to_graph_multi(grids, add_cross_table_edges=add_cross_table_edges)
    prompt, _ = build_qa_prompt_mtqa(
        graph, question, paragraphs=paragraphs, use_reasoning=use_reasoning,
    )
    sample = tokenize_sample(prompt, answer, graph, tokenizer, max_length)
    if sample is None:
        return None, "empty"
    return sample, "ok"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_doclibrag_dataset(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    documents_path: Optional[str] = None,
    max_length: int = 4096,
    max_samples: int = -1,
    add_cross_table_edges: bool = True,
    reasoning_path: Optional[str] = None,
    num_workers: Optional[int] = None,
    **_,
) -> List[Dict]:
    """Load DocLibRAG QA samples (multi-table + text, single answer).

    Args:
        num_workers: thread-pool size for per-sample work. Defaults to
            min(8, os.cpu_count()) or can be overridden via the
            DOCLIBRAG_NUM_WORKERS env var.
    """
    documents_path = documents_path or _default_documents_path(data_path)
    documents = load_doclibrag_documents(documents_path)
    logger.info(f"DocLibRAG: loaded {len(documents)} documents from {documents_path}")

    use_reasoning = reasoning_path is not None
    reasoning_map = load_reasoning_jsonl(reasoning_path) if use_reasoning else {}
    if use_reasoning:
        logger.info(f"DocLibRAG: loaded {len(reasoning_map)} matched reasoning entries")

    with open(data_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    # Decide worker count
    if num_workers is None:
        num_workers = int(os.environ.get(
            "DOCLIBRAG_NUM_WORKERS",
            min(8, max(os.cpu_count() or 4, 2)),
        ))
    num_workers = max(1, int(num_workers))

    # Optionally cap samples up-front to avoid wasted work
    if 0 < max_samples < len(qa_data):
        # Note: workers may produce skipped samples that won't count toward the
        # cap, so keep some headroom by oversampling 2× then trimming.
        qa_data = qa_data[: min(len(qa_data), max_samples * 2)]

    # Pre-parse all needed documents in parallel (HTML → grids), cached by uid
    needed_uids = {it.get("document_uid", "") for it in qa_data}
    needed_uids.discard("")
    logger.info(f"DocLibRAG: parsing tables for {len(needed_uids)} unique documents "
                f"using {num_workers} workers …")
    grid_cache = _build_doc_grid_cache(documents, needed_uids, num_workers)

    # Process per-QA work concurrently
    samples: List[Dict] = []
    counters = {"ok": 0, "empty": 0, "no_doc": 0, "no_reasoning": 0}
    lock = Lock()
    cap_reached = False

    logger.info(f"DocLibRAG: tokenizing {len(qa_data)} QA items with {num_workers} workers …")

    def _on_done(sample_status):
        nonlocal cap_reached
        sample, status = sample_status
        with lock:
            counters[status] = counters.get(status, 0) + 1
            if sample is not None and not cap_reached:
                samples.append(sample)
                if 0 < max_samples <= len(samples):
                    cap_reached = True

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = []
        for item in qa_data:
            if cap_reached:
                break
            fut = ex.submit(
                _process_one_qa,
                item, documents, grid_cache, tokenizer,
                max_length, add_cross_table_edges,
                reasoning_map, use_reasoning,
            )
            futures.append(fut)

        done = 0
        for fut in as_completed(futures):
            try:
                _on_done(fut.result())
            except Exception as e:  # pragma: no cover
                logger.warning(f"DocLibRAG: worker error: {e}")
                with lock:
                    counters["empty"] += 1
            done += 1
            if done % 200 == 0:
                logger.info(f"  DocLibRAG: processed {done}/{len(futures)} "
                            f"(kept {len(samples)})")
            if cap_reached:
                # No way to cancel pending futures cleanly, but we stop counting.
                pass

    if counters["no_doc"]:
        logger.warning(f"DocLibRAG: skipped {counters['no_doc']} samples (document_uid not found)")
    if counters["empty"]:
        logger.warning(f"DocLibRAG: skipped {counters['empty']} samples (empty fields / invalid tables)")
    if counters["no_reasoning"]:
        logger.info(f"DocLibRAG: skipped {counters['no_reasoning']} samples (no matched reasoning)")
    logger.info(f"DocLibRAG: loaded {len(samples)} training samples "
                f"(use_reasoning={use_reasoning}, num_workers={num_workers})")
    return samples


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def _doclibrag_evaluator(*args, **kwargs):
    from UniGSA.eval.doclibrag import run_doclibrag_eval
    return run_doclibrag_eval(*args, **kwargs)


register_dataset(DatasetSpec(
    name="doclibrag",
    loader=load_doclibrag_dataset,
    evaluator=_doclibrag_evaluator,
    primary_metric="em",
    needs_text_context=True,
    multi_table=True,
    default_max_length=4096,
    description="DocLibRAG — multi-table + text QA over a shared document library",
))
