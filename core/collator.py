"""
Data Collator for UniGSA.

Handles batching of:
  - Token-level fields: input_ids, attention_mask, labels, row_ids, col_ids,
    node_ids, depth_ids, parent_ids, table_ids (padded to max_seq_len)
  - Edge-level fields: edge_src, edge_dst, edge_types, edge_mask
    (padded to max_edges in batch)
"""

from typing import Any, Dict, List
import torch


class GraphLoRACollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)

        token_fields = {
            "input_ids": self.pad_token_id,
            "attention_mask": 0,
            "labels": -100,
            "row_ids": 0,
            "col_ids": 0,
            "node_ids": 0,
            "depth_ids": 0,
            "parent_ids": 0,
            "table_ids": 0,
        }

        batch = {k: [] for k in token_fields}
        for f in features:
            pad = max_len - len(f["input_ids"])
            for k, pad_val in token_fields.items():
                # `table_ids` may be missing in old samples → fall back to zeros
                seq = f.get(k, [pad_val] * len(f["input_ids"]))
                batch[k].append(seq + [pad_val] * pad)

        node_fields = ("graph_rows", "graph_cols", "graph_depths", "graph_tables")
        if all("graph_rows" in f for f in features):
            max_nodes = max(len(f["graph_rows"]) for f in features)
            for key in node_fields:
                batch[key] = [f[key] + [0] * (max_nodes - len(f[key])) for f in features]

        max_edges = max(len(f.get("edge_src", [])) for f in features)
        max_edges = max(max_edges, 1)

        edge_src_batch, edge_dst_batch, edge_types_batch, edge_mask_batch = [], [], [], []
        for f in features:
            es = f.get("edge_src", [])
            ed = f.get("edge_dst", [])
            et = f.get("edge_types", [])
            n_edges = len(es)
            pad_n = max_edges - n_edges
            edge_src_batch.append(es + [0] * pad_n)
            edge_dst_batch.append(ed + [0] * pad_n)
            edge_types_batch.append(et + [0] * pad_n)
            edge_mask_batch.append([True] * n_edges + [False] * pad_n)

        batch["edge_src"] = edge_src_batch
        batch["edge_dst"] = edge_dst_batch
        batch["edge_types"] = edge_types_batch
        batch["edge_mask"] = edge_mask_batch

        result = {}
        for k, v in batch.items():
            if k == "edge_mask":
                result[k] = torch.tensor(v, dtype=torch.bool)
            else:
                result[k] = torch.tensor(v, dtype=torch.long)
        return result
