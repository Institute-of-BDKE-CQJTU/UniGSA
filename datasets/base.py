"""Dataset specification & registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional


@dataclass
class DatasetSpec:
    name: str
    loader: Callable                    # (data_path, tokenizer, ...) → list[sample dict]
    evaluator: Callable                 # (model, tokenizer, data_path, ...) → metrics dict
    primary_metric: str = "em"          # the metric used for "best checkpoint" selection
    needs_text_context: bool = False    # text + table prompt
    multi_table: bool = False           # multiple tables per sample
    default_max_length: int = 2048
    description: str = ""
    extra: Dict = field(default_factory=dict)


DATASET_REGISTRY: Dict[str, DatasetSpec] = {}


def register_dataset(spec: DatasetSpec) -> None:
    DATASET_REGISTRY[spec.name] = spec


def get_spec(name: str) -> DatasetSpec:
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Registered: {list(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[name]
