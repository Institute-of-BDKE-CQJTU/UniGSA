"""Dataset loaders + registry.

Importing this module registers the built-in datasets:
    hitab, tatqa, finqa, multihiertt, generic
"""

from UniGSA.datasets.base import DatasetSpec, DATASET_REGISTRY, register_dataset, get_spec

# Side-effect imports to populate DATASET_REGISTRY
from UniGSA.datasets import hitab as _hitab            # noqa: F401
from UniGSA.datasets import tatqa as _tatqa            # noqa: F401
from UniGSA.datasets import finqa as _finqa            # noqa: F401
from UniGSA.datasets import multihiertt as _mhtt       # noqa: F401
from UniGSA.datasets import doclibrag as _doclibrag    # noqa: F401
from UniGSA.datasets import generic as _generic        # noqa: F401

__all__ = ["DatasetSpec", "DATASET_REGISTRY", "register_dataset", "get_spec"]
