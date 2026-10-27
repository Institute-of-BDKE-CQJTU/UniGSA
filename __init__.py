"""UniGSA: paper-aligned table/text-table structural adaptation.

HAG adds one masked residual at the input embedding layer. Independent
HierGraph2D-LoRA modules add a six-identifier bias at adapted K/V projections.
The new serialization requires freshly trained UniGSA checkpoints.
"""

from UniGSA.core.model import (
    GraphLoRAModel,
    SpecialTokenEncoder,
    HierGATLayer,
    HierGraph2DLoRA,
    detect_target_modules,
    detect_embed_layer,
    detect_decoder_layer_class,
    save_graphlora_fsdp,
)
from UniGSA.core.collator import GraphLoRACollator
from UniGSA.core.tokenize_utils import (
    SPECIAL_TOKENS,
    IGNORE_INDEX,
    get_all_special_tokens,
    GraphQADataset,
)
from UniGSA.datasets import DATASET_REGISTRY, DatasetSpec

__all__ = [
    "GraphLoRAModel",
    "SpecialTokenEncoder",
    "HierGATLayer",
    "HierGraph2DLoRA",
    "detect_target_modules",
    "detect_embed_layer",
    "detect_decoder_layer_class",
    "save_graphlora_fsdp",
    "GraphLoRACollator",
    "SPECIAL_TOKENS",
    "IGNORE_INDEX",
    "get_all_special_tokens",
    "GraphQADataset",
    "DATASET_REGISTRY",
    "DatasetSpec",
]

UniGSAModel = GraphLoRAModel
