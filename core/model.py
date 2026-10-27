"""UniGSA: paper-aligned table/text-table structural adaptation.

HAG adds one masked residual at the input embedding layer. Independent
HierGraph2D-LoRA modules add a six-identifier bias at adapted K/V projections.
The new serialization requires freshly trained UniGSA checkpoints.
"""

from __future__ import annotations

import math
import os
import json
import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from peft.tuners.lora.layer import Linear as LoraLinear

from UniGSA.core.tokenize_utils import SPECIAL_TOKENS, SERIALIZATION_VERSION
from UniGSA.core.graph_utils import NUM_EDGE_TYPES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Auto-detection helpers
# ---------------------------------------------------------------------------

def detect_target_modules(model: PreTrainedModel) -> List[str]:
    linear_names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            short = name.split(".")[-1]
            linear_names.add(short)
    candidates = ["k_proj", "v_proj"]
    found = [c for c in candidates if c in linear_names]
    if found:
        return found
    candidates2 = ["q_proj", "k_proj", "v_proj", "o_proj"]
    found2 = [c for c in candidates2 if c in linear_names]
    if found2:
        return found2[:2]
    for name in sorted(linear_names):
        if "attn" in name or "proj" in name:
            found.append(name)
    return found[:2] if found else ["k_proj", "v_proj"]


def detect_embed_layer(model):
    for attr_chain in [
        ["model", "embed_tokens"],
        ["model", "model", "embed_tokens"],
        ["transformer", "wte"],
        ["embed_tokens"],
    ]:
        obj = model
        try:
            for attr in attr_chain:
                obj = getattr(obj, attr)
            if isinstance(obj, nn.Embedding):
                return obj
        except AttributeError:
            continue
    raise AttributeError(f"Cannot find embedding layer in {type(model).__name__}")


def detect_decoder_layer_class(model: PreTrainedModel):
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if "DecoderLayer" in cls_name:
            return type(module)
    model_type = getattr(model.config, "model_type", "")
    try:
        if model_type in ("qwen2", "qwen3", "qwen2_moe"):
            from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
            return Qwen2DecoderLayer
        elif model_type == "llama":
            from transformers.models.llama.modeling_llama import LlamaDecoderLayer
            return LlamaDecoderLayer
        elif model_type == "mistral":
            from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
            return MistralDecoderLayer
    except ImportError:
        pass
    raise ValueError(f"Cannot auto-detect decoder layer class for model_type='{model_type}'.")


# ---------------------------------------------------------------------------
# FSDP checkpoint save utility
# ---------------------------------------------------------------------------

def _normalize_fsdp_key(key: str) -> str:
    prefixes = ["_fsdp_wrapped_module.", "_orig_mod."]
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if key.startswith(p):
                key = key[len(p):]
                changed = True
    return key


def save_graphlora_fsdp(fsdp_model, save_path, rank, graphlora_config):
    """Save GraphLoRA from FSDP. Must be called by ALL ranks; only rank 0 writes."""
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        FullStateDictConfig,
        StateDictType,
    )

    full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, full_cfg):
        state = fsdp_model.state_dict()

    if rank == 0:
        os.makedirs(save_path, exist_ok=True)
        normalized_state = {_normalize_fsdp_key(k): v for k, v in state.items()}

        se_state, gl_state, gat_state, lora_weights = {}, {}, {}, {}
        for k, v in normalized_state.items():
            if "special_encoder." in k:
                idx = k.index("special_encoder.")
                se_state[k[idx + len("special_encoder."):]] = v
            elif "hier_gat." in k:
                idx = k.index("hier_gat.")
                gat_state[k[idx + len("hier_gat."):]] = v
            elif "graph_layers." in k:
                idx = k.index("graph_layers.")
                gl_state[k[idx + len("graph_layers."):]] = v
            elif "lora_A" in k or "lora_B" in k or "lora_embedding" in k:
                peft_key = k[len("model."):] if k.startswith("model.") else k
                lora_weights[peft_key] = v

        logger.info(f"Extracted: {len(se_state)} special_encoder, {len(gl_state)} graph_layers, "
                    f"{len(gat_state)} hier_gat, {len(lora_weights)} LoRA weights")

        torch.save(se_state, os.path.join(save_path, "special_encoder.pt"))
        torch.save(gl_state, os.path.join(save_path, "graph_layers.pt"))
        torch.save(gat_state, os.path.join(save_path, "hier_gat.pt"))
        torch.save(lora_weights, os.path.join(save_path, "lora_weights.pt"))

        token_ids = list(graphlora_config["special_token_ids"].values())
        input_key = next((k for k in normalized_state if k.endswith("embed_tokens.weight")
                          or k.endswith("wte.weight")), None)
        output_key = next((k for k in normalized_state if k.endswith("lm_head.weight")), input_key)
        if input_key is None or output_key is None:
            raise ValueError("Cannot locate frozen token rows in FSDP checkpoint")
        torch.save({"ids": token_ids, "input": normalized_state[input_key][token_ids].cpu(),
                    "output": normalized_state[output_key][token_ids].cpu()},
                   os.path.join(save_path, "special_token_rows.pt"))
        graphlora_config = dict(graphlora_config, serialization_version=SERIALIZATION_VERSION)

        adapter_dir = os.path.join(save_path, "lora_adapter")
        os.makedirs(adapter_dir, exist_ok=True)
        torch.save(lora_weights, os.path.join(adapter_dir, "adapter_model.bin"))
        adapter_config = {
            "auto_mapping": None, "base_model_name_or_path": "", "bias": "none",
            "fan_in_fan_out": False, "inference_mode": False, "init_lora_weights": True,
            "lora_alpha": graphlora_config["lora_alpha"],
            "lora_dropout": graphlora_config.get("lora_dropout", 0.1),
            "modules_to_save": None, "peft_type": "LORA",
            "r": graphlora_config["lora_rank"], "revision": None,
            "target_modules": graphlora_config["target_modules"], "task_type": "CAUSAL_LM",
        }
        with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
            json.dump(adapter_config, f, indent=2)

        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(graphlora_config, f, indent=2)
        logger.info(f"FSDP checkpoint saved to {save_path}")


# ---------------------------------------------------------------------------
# Special Token Encoder
# ---------------------------------------------------------------------------

class SpecialTokenEncoder(nn.Module):
    def __init__(self, num_tokens: int, dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, dim)
        self.linear = nn.Linear(dim, dim)
        self.act = nn.Tanh()

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(self.embedding(indices)))

    @torch.no_grad()
    def initialize_from_embeddings(self, values: torch.Tensor):
        """Preserve expanded-vocabulary input embeddings at step zero.

        tanh(atanh(e)) = e, with an identity linear projection. The encoder
        remains fully trainable. Values outside (-1, 1) cannot be represented
        by the paper's tanh output and are rejected instead of clipped.
        """
        values = values.detach().to(self.embedding.weight)
        if not torch.isfinite(values).all() or (values.abs() >= 1).any():
            raise ValueError("Special-token embeddings must lie in (-1, 1) for step-zero equivalence")
        self.embedding.weight.copy_(torch.atanh(values))
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)


# ---------------------------------------------------------------------------
# Hierarchy-Aware Graph Attention Layer (with optional table feature)
# ---------------------------------------------------------------------------

class HierGATLayer(nn.Module):
    """Lightweight GAT for hierarchy-aware message passing.

    Operates at the cell (node) level for efficiency. Edge types include
    PARENT_CHILD, SIBLING, and CROSS_TABLE (for multi-table data).
    """

    def __init__(
        self,
        hidden_dim: int,
        gat_rank: int = 32,
        max_rows: int = 600,
        max_cols: int = 50,
        max_depths: int = 20,
        max_tables: int = 16,
        num_edge_types: int = NUM_EDGE_TYPES,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gat_rank = gat_rank
        self.num_edge_types = num_edge_types

        self.row_emb = nn.Embedding(max_rows, gat_rank)
        self.col_emb = nn.Embedding(max_cols, gat_rank)
        self.depth_emb = nn.Embedding(max_depths, gat_rank)
        self.table_emb = nn.Embedding(max_tables, gat_rank)
        self.edge_type_emb = nn.Embedding(num_edge_types, gat_rank)

        self.W_q = nn.Linear(gat_rank, gat_rank, bias=False)
        self.W_k = nn.Linear(gat_rank, gat_rank, bias=False)
        self.attn_vec = nn.Linear(3 * gat_rank, 1, bias=False)
        self.act = nn.LeakyReLU(0.2)
        self.msg_linear = nn.Linear(gat_rank, gat_rank, bias=False)
        self.up_proj = nn.Linear(gat_rank, hidden_dim, bias=False)
        self.layer_norm = nn.LayerNorm(gat_rank)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.zeros(1))

        nn.init.zeros_(self.up_proj.weight)
        nn.init.kaiming_uniform_(self.row_emb.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.col_emb.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.depth_emb.weight, a=math.sqrt(5))
        nn.init.zeros_(self.table_emb.weight)  # zero table-membership contribution at initialization

    def forward(
        self,
        node_ids: torch.Tensor,
        row_ids: torch.Tensor,
        col_ids: torch.Tensor,
        depth_ids: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_types: torch.Tensor,
        edge_mask: torch.Tensor,
        table_ids: Optional[torch.Tensor] = None,
        graph_rows=None, graph_cols=None, graph_depths=None, graph_tables=None,
    ) -> torch.Tensor:
        B, S = node_ids.shape
        R = self.gat_rank
        device = node_ids.device
        if graph_rows is None:
            # Compatibility for direct module callers; normal data paths supply
            # explicit metadata so empty cells retain their structural features.
            size = int(node_ids.max().item()) + 1
            graph_rows = torch.zeros(B, size, dtype=torch.long, device=device)
            graph_cols = torch.zeros_like(graph_rows)
            graph_depths = torch.zeros_like(graph_rows)
            graph_tables = torch.zeros_like(graph_rows)
            for target, source in ((graph_rows, row_ids), (graph_cols, col_ids),
                                   (graph_depths, depth_ids)):
                target.scatter_(1, node_ids, source)
            if table_ids is not None:
                graph_tables.scatter_(1, node_ids, table_ids)
        size = graph_rows.shape[1]
        node_feat = (self.row_emb(graph_rows) + self.col_emb(graph_cols)
                     + self.depth_emb(graph_depths) + self.table_emb(graph_tables))
        valid_node = graph_rows > 0
        valid_node[:, 0] = False

        # Project once per node, not per incident edge.
        q_nodes = self.W_q(node_feat)
        k_nodes = self.W_k(node_feat)
        msg_nodes = self.msg_linear(node_feat)
        agg = torch.zeros_like(node_feat)
        valid_edge = edge_mask.bool()
        invalid = ((edge_src <= 0) | (edge_src >= size)
                   | (edge_dst <= 0) | (edge_dst >= size)) & valid_edge
        if invalid.any():
            raise ValueError("Valid graph edge references a missing node")
        src = edge_src.masked_fill(~valid_edge, 0)
        dst = edge_dst.masked_fill(~valid_edge, 0)
        valid_edge = valid_edge & valid_node.gather(1, src) & valid_node.gather(1, dst)
        src_idx = src.unsqueeze(-1).expand(-1, -1, R)
        dst_idx = dst.unsqueeze(-1).expand(-1, -1, R)
        q = q_nodes.gather(1, dst_idx)
        k = k_nodes.gather(1, src_idx)
        e_feat = self.edge_type_emb(edge_types.masked_fill(~valid_edge, 0))
        scores = self.act(self.attn_vec(torch.cat([q, k, e_feat], dim=-1))).squeeze(-1)
        weights = self.dropout(torch.sigmoid(scores) * valid_edge.to(scores.dtype))
        messages = weights.unsqueeze(-1) * msg_nodes.gather(1, src_idx)
        agg.scatter_add_(1, dst_idx, messages)
        residual = node_feat + agg
        normalized = F.layer_norm(
            residual.float(), self.layer_norm.normalized_shape,
            self.layer_norm.weight.float(), self.layer_norm.bias.float(), self.layer_norm.eps,
        ).to(self.up_proj.weight.dtype)
        # The paper's residual remains meaningful for isolated real nodes.
        # Project at node granularity before broadcasting to subword tokens.
        node_out = torch.sigmoid(self.gate) * self.up_proj(normalized)
        node_out = node_out * valid_node.unsqueeze(-1).to(node_out.dtype)
        return node_out.gather(1, node_ids.unsqueeze(-1).expand(-1, -1, node_out.shape[-1]))


# ---------------------------------------------------------------------------
# Hierarchy-Aware Graph 2D LoRA (with table_emb)
# ---------------------------------------------------------------------------

class HierGraph2DLoRA(nn.Module):
    """Per-layer structural encoding.

    output = B_graph · (Emb_row + Emb_col + Emb_node + Emb_depth + Emb_parent
                         + Emb_table) * scaling
    """

    def __init__(
        self,
        out_features: int,
        rank: int = 8,
        max_rows: int = 600,
        max_cols: int = 50,
        max_nodes: int = 2000,
        max_depths: int = 20,
        max_parents: int = 2000,
        max_tables: int = 16,
        lora_alpha: int = 16,
    ):
        super().__init__()
        self.row_emb = nn.Embedding(max_rows, rank)
        self.col_emb = nn.Embedding(max_cols, rank)
        self.node_emb = nn.Embedding(max_nodes, rank)
        self.depth_emb = nn.Embedding(max_depths, rank)
        self.parent_emb = nn.Embedding(max_parents, rank)
        self.table_emb = nn.Embedding(max_tables, rank)
        self.B_graph = nn.Linear(rank, out_features, bias=False)
        self.scaling = lora_alpha / rank

        nn.init.kaiming_uniform_(self.row_emb.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.col_emb.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.node_emb.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.depth_emb.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.parent_emb.weight, a=math.sqrt(5))
        nn.init.zeros_(self.table_emb.weight)  # zero table-membership contribution at initialization
        nn.init.zeros_(self.B_graph.weight)

    def forward(self, row_ids, col_ids, node_ids, depth_ids, parent_ids, table_ids=None):
        r = self.row_emb(row_ids)
        c = self.col_emb(col_ids)
        n = self.node_emb(node_ids)
        d = self.depth_emb(depth_ids)
        p = self.parent_emb(parent_ids)
        if table_ids is not None:
            t = self.table_emb(table_ids)
            return self.B_graph(r + c + n + d + p + t) * self.scaling
        return self.B_graph(r + c + n + d + p) * self.scaling


# ---------------------------------------------------------------------------
# GraphLoRA Model
# ---------------------------------------------------------------------------

class GraphLoRAModel(nn.Module):
    """Wraps a pretrained LLM with LoRA + SpecialTokenEncoder + HierGAT
    + per-layer HierGraph2DLoRA bias. Supports single- and multi-table data."""

    def __init__(
        self,
        base_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        max_rows: int = 600,
        max_cols: int = 50,
        max_nodes: int = 2000,
        max_depths: int = 20,
        max_parents: int = 2000,
        max_tables: int = 16,
        gat_rank: int = 32,
        gat_dropout: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.gat_dropout = gat_dropout
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.max_nodes = max_nodes
        self.max_depths = max_depths
        self.max_parents = max_parents
        self.max_tables = max_tables
        self.gat_rank = gat_rank

        if target_modules is None:
            target_modules = detect_target_modules(base_model)
        self._target_modules = target_modules

        self.special_token_ids = {
            name: tokenizer.convert_tokens_to_ids(tok)
            for name, tok in SPECIAL_TOKENS.items()
        }
        self._stid_to_idx = {}
        for i, (name, tok) in enumerate(SPECIAL_TOKENS.items()):
            self._stid_to_idx[tokenizer.convert_tokens_to_ids(tok)] = i

        # 1) LoRA
        lora_cfg = LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=target_modules, task_type="CAUSAL_LM", bias="none",
        )
        self.model = get_peft_model(base_model, lora_cfg)

        # 2) Special Token Encoder
        hidden = base_model.config.hidden_size
        self.special_encoder = SpecialTokenEncoder(len(SPECIAL_TOKENS), hidden)
        self.special_encoder.initialize_from_embeddings(
            base_model.get_input_embeddings().weight[list(self._stid_to_idx)]
        )

        # 3) HierGAT
        self.hier_gat = HierGATLayer(
            hidden_dim=hidden, gat_rank=gat_rank,
            max_rows=max_rows, max_cols=max_cols, max_depths=max_depths,
            max_tables=max_tables, num_edge_types=NUM_EDGE_TYPES,
            dropout=gat_dropout,
        )

        # 4) Per-layer HierGraph2DLoRA
        self.graph_layers = nn.ModuleDict()
        self._lora_map = {}
        for name, mod in self.model.named_modules():
            if isinstance(mod, LoraLinear):
                out_f = mod.base_layer.out_features
                key = name.replace(".", "_")
                self.graph_layers[key] = HierGraph2DLoRA(
                    out_f, lora_rank, max_rows, max_cols, max_nodes,
                    max_depths, max_parents, max_tables, lora_alpha,
                )
                self._lora_map[id(mod)] = key

        # 5) Hooks
        self._row_ids = None
        self._col_ids = None
        self._node_ids = None
        self._depth_ids = None
        self._parent_ids = None
        self._table_ids = None
        self._hooks = []
        for name, mod in self.model.named_modules():
            if isinstance(mod, LoraLinear):
                self._hooks.append(mod.register_forward_hook(self._hook))

    def _hook(self, module, inp, out):
        if self._row_ids is None:
            return out
        key = self._lora_map.get(id(module))
        if key is None:
            return out
        gl = self.graph_layers[key]
        # Make sure structural-id tensors live on the same device as this
        # LoRA layer (needed for naive model parallel / device_map="auto").
        gl_dev = next(gl.parameters()).device
        g = gl(
            self._row_ids.to(gl_dev),
            self._col_ids.to(gl_dev),
            self._node_ids.to(gl_dev),
            self._depth_ids.to(gl_dev),
            self._parent_ids.to(gl_dev),
            self._table_ids.to(gl_dev) if self._table_ids is not None else None,
        )
        return out + g.to(device=out.device, dtype=out.dtype)

    def _get_embed_layer(self):
        base = self.model.base_model
        if hasattr(base, "model"):
            base = base.model
        return detect_embed_layer(base)

    def _replace_special_embeds(self, input_ids):
        embed = self._get_embed_layer()
        embeds = embed(input_ids).clone()
        # Under naive model parallel (`device_map="auto"`), embed may live on a
        # different device than `input_ids`. Align everything to embed device.
        emb_dev = embeds.device
        if input_ids.device != emb_dev:
            input_ids = input_ids.to(emb_dev)
        ids_list = list(self._stid_to_idx.keys())
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for sid in ids_list:
            mask |= (input_ids == sid)
        if mask.any():
            idx_map = torch.zeros_like(input_ids)
            for tid, idx in self._stid_to_idx.items():
                idx_map[input_ids == tid] = idx
            se_dev = next(self.special_encoder.parameters()).device
            enc_out = self.special_encoder(idx_map[mask].to(se_dev))
            embeds[mask] = enc_out.to(device=emb_dev, dtype=embeds.dtype)
        return embeds

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        row_ids=None,
        col_ids=None,
        node_ids=None,
        depth_ids=None,
        parent_ids=None,
        table_ids=None,
        edge_src=None,
        edge_dst=None,
        edge_types=None,
        edge_mask=None,
        graph_rows=None, graph_cols=None, graph_depths=None, graph_tables=None,
        **kw,
    ):
        self._row_ids = row_ids
        self._col_ids = col_ids
        self._node_ids = node_ids
        self._depth_ids = depth_ids
        self._parent_ids = parent_ids
        self._table_ids = table_ids

        embeds = self._replace_special_embeds(input_ids)

        if (edge_src is not None and edge_mask is not None
                and node_ids is not None and depth_ids is not None):
            gat_out = self.hier_gat(
                node_ids=node_ids, row_ids=row_ids, col_ids=col_ids,
                depth_ids=depth_ids, edge_src=edge_src, edge_dst=edge_dst,
                edge_types=edge_types, edge_mask=edge_mask,
                table_ids=table_ids,
                graph_rows=graph_rows, graph_cols=graph_cols,
                graph_depths=graph_depths, graph_tables=graph_tables,
            )
            embeds = embeds + gat_out.to(device=embeds.device, dtype=embeds.dtype)

        return self.model(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kw,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        attention_mask=None,
        row_ids=None,
        col_ids=None,
        node_ids=None,
        depth_ids=None,
        parent_ids=None,
        table_ids=None,
        edge_src=None,
        edge_dst=None,
        edge_types=None,
        edge_mask=None,
        graph_rows=None, graph_cols=None, graph_depths=None, graph_tables=None,
        max_new_tokens=256,
        temperature=1.0,
        do_sample=False,
        **kw,
    ):
        self._row_ids = row_ids
        self._col_ids = col_ids
        self._node_ids = node_ids
        self._depth_ids = depth_ids
        self._parent_ids = parent_ids
        self._table_ids = table_ids

        embeds = self._replace_special_embeds(input_ids)

        if (edge_src is not None and edge_mask is not None
                and node_ids is not None and depth_ids is not None):
            gat_out = self.hier_gat(
                node_ids=node_ids, row_ids=row_ids, col_ids=col_ids,
                depth_ids=depth_ids, edge_src=edge_src, edge_dst=edge_dst,
                edge_types=edge_types, edge_mask=edge_mask,
                table_ids=table_ids,
                graph_rows=graph_rows, graph_cols=graph_cols,
                graph_depths=graph_depths, graph_tables=graph_tables,
            )
            embeds = embeds + gat_out.to(device=embeds.device, dtype=embeds.dtype)

        device = embeds.device
        B = embeds.shape[0]
        emb_layer = self._get_embed_layer()
        generated = []
        past = None
        cur_embeds = embeds
        cur_mask = attention_mask

        for _ in range(max_new_tokens):
            if past is not None:
                self._row_ids = torch.zeros(B, 1, device=device, dtype=torch.long)
                self._col_ids = torch.zeros(B, 1, device=device, dtype=torch.long)
                self._node_ids = torch.zeros(B, 1, device=device, dtype=torch.long)
                self._depth_ids = torch.zeros(B, 1, device=device, dtype=torch.long)
                self._parent_ids = torch.zeros(B, 1, device=device, dtype=torch.long)
                self._table_ids = torch.zeros(B, 1, device=device, dtype=torch.long)

            out = self.model(
                inputs_embeds=cur_embeds,
                attention_mask=cur_mask,
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            logits = out.logits[:, -1, :]

            if do_sample and temperature > 0:
                nxt = torch.multinomial(
                    torch.softmax(logits / temperature, -1), 1
                ).squeeze(-1)
            else:
                nxt = logits.argmax(-1)

            generated.append(nxt)
            if (self.tokenizer.eos_token_id is not None
                    and (nxt == self.tokenizer.eos_token_id).all()):
                break

            cur_embeds = emb_layer(nxt.unsqueeze(1))
            cur_mask = torch.cat(
                [cur_mask, torch.ones(B, 1, device=device, dtype=cur_mask.dtype)], 1
            )

        self._row_ids = self._col_ids = self._node_ids = None
        self._depth_ids = self._parent_ids = self._table_ids = None

        if generated:
            return torch.stack(generated, 1)
        return torch.tensor([[]], dtype=torch.long, device=device)

    def get_trainable_parameters(self):
        names, trainable, total = [], 0, 0
        for n, p in self.named_parameters():
            total += p.numel()
            if p.requires_grad:
                names.append(n)
                trainable += p.numel()
        return names, trainable, total

    def save_graphlora(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(os.path.join(path, "lora_adapter"), save_embedding_layers=False)
        torch.save(self.special_encoder.state_dict(),
                   os.path.join(path, "special_encoder.pt"))
        torch.save(self.graph_layers.state_dict(),
                   os.path.join(path, "graph_layers.pt"))
        torch.save(self.hier_gat.state_dict(),
                   os.path.join(path, "hier_gat.pt"))
        cfg = {
            "serialization_version": SERIALIZATION_VERSION,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout, "gat_dropout": self.gat_dropout,
            "max_rows": self.max_rows,
            "max_cols": self.max_cols,
            "max_nodes": self.max_nodes,
            "max_depths": self.max_depths,
            "max_parents": self.max_parents,
            "max_tables": self.max_tables,
            "gat_rank": self.gat_rank,
            "target_modules": self._target_modules,
            "special_token_ids": self.special_token_ids,
        }
        self.tokenizer.save_pretrained(path)
        self._save_added_token_rows(path)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    def _save_added_token_rows(self, path):
        ids = list(self._stid_to_idx)
        inputs = self._get_embed_layer().weight
        outputs = self.model.get_output_embeddings().weight
        torch.save({"ids": ids, "input": inputs[ids].detach().cpu(),
                    "output": outputs[ids].detach().cpu()},
                   os.path.join(path, "special_token_rows.pt"))

    def _load_added_token_rows(self, path):
        state = torch.load(os.path.join(path, "special_token_rows.pt"), map_location="cpu", weights_only=True)
        if state["ids"] != list(self._stid_to_idx):
            raise ValueError("Checkpoint and tokenizer have different special token IDs")
        with torch.no_grad():
            for name, layer in (("input", self._get_embed_layer()),
                                ("output", self.model.get_output_embeddings())):
                layer.weight[state["ids"]] = state[name].to(layer.weight)

    @classmethod
    def load_graphlora(cls, base_model, tokenizer, path, **kw):
        with open(os.path.join(path, "config.json")) as f:
            cfg = json.load(f)
        if cfg.get("serialization_version") != SERIALIZATION_VERSION:
            raise ValueError("Checkpoint uses a different graph serialization. Train a new UniGSA checkpoint.")

        lora_weights_path = os.path.join(path, "lora_weights.pt")
        lora_path = os.path.join(path, "lora_adapter")

        if os.path.exists(lora_weights_path):
            logger.info(f"Loading LoRA weights from FSDP format: {lora_weights_path}")
            lora_cfg = LoraConfig(
                r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"],
                lora_dropout=cfg.get("lora_dropout", 0.1),
                target_modules=cfg["target_modules"], task_type="CAUSAL_LM", bias="none",
            )
            peft_model = get_peft_model(base_model, lora_cfg)
            saved_lora = torch.load(lora_weights_path, map_location="cpu")
            model_state = peft_model.state_dict()
            loaded_count = 0
            for key in saved_lora:
                if key in model_state:
                    model_state[key] = saved_lora[key]
                    loaded_count += 1
            if loaded_count == 0 and len(saved_lora) > 0:
                suffix_to_model_key = {}
                for mk in model_state:
                    parts = mk.split(".")
                    for i in range(1, len(parts)):
                        suffix = ".".join(parts[i:])
                        suffix_to_model_key[suffix] = mk
                for sk, sv in saved_lora.items():
                    parts = sk.split(".")
                    for i in range(len(parts)):
                        suffix = ".".join(parts[i:])
                        if suffix in suffix_to_model_key:
                            model_state[suffix_to_model_key[suffix]] = sv
                            loaded_count += 1
                            break
            peft_model.load_state_dict(model_state, strict=False)
            logger.info(f"Loaded {loaded_count}/{len(saved_lora)} LoRA weights")
        elif os.path.exists(lora_path):
            peft_model = PeftModel.from_pretrained(base_model, lora_path)
        else:
            raise FileNotFoundError(f"No LoRA weights found at {path}.")

        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.tokenizer = tokenizer
        obj.lora_rank = cfg["lora_rank"]
        obj.lora_alpha = cfg["lora_alpha"]
        obj.lora_dropout = cfg.get("lora_dropout", 0.1)
        obj.gat_dropout = cfg.get("gat_dropout", 0.1)
        obj.max_rows = cfg["max_rows"]
        obj.max_cols = cfg["max_cols"]
        obj.max_nodes = cfg["max_nodes"]
        obj.max_depths = cfg.get("max_depths", 20)
        obj.max_parents = cfg.get("max_parents", 2000)
        obj.max_tables = cfg.get("max_tables", 16)
        obj.gat_rank = cfg.get("gat_rank", 32)
        obj._target_modules = cfg["target_modules"]
        obj.special_token_ids = cfg["special_token_ids"]
        obj.model = peft_model

        obj._stid_to_idx = {}
        for i, (name, tok) in enumerate(SPECIAL_TOKENS.items()):
            obj._stid_to_idx[tokenizer.convert_tokens_to_ids(tok)] = i

        hidden = base_model.config.hidden_size
        obj.special_encoder = SpecialTokenEncoder(len(SPECIAL_TOKENS), hidden)
        obj.special_encoder.load_state_dict(
            torch.load(os.path.join(path, "special_encoder.pt"), map_location="cpu"))

        obj.hier_gat = HierGATLayer(
            hidden_dim=hidden, gat_rank=obj.gat_rank,
            max_rows=obj.max_rows, max_cols=obj.max_cols,
            max_depths=obj.max_depths, max_tables=obj.max_tables,
            num_edge_types=NUM_EDGE_TYPES, dropout=obj.gat_dropout,
        )
        # New serialization checkpoints must contain all structural weights.
        gat_state = torch.load(os.path.join(path, "hier_gat.pt"), map_location="cpu")
        obj.hier_gat.load_state_dict(gat_state, strict=True)

        obj.graph_layers = nn.ModuleDict()
        obj._lora_map = {}
        for name, mod in obj.model.named_modules():
            if isinstance(mod, LoraLinear):
                out_f = mod.base_layer.out_features
                key = name.replace(".", "_")
                obj.graph_layers[key] = HierGraph2DLoRA(
                    out_f, cfg["lora_rank"], cfg["max_rows"], cfg["max_cols"],
                    cfg["max_nodes"], obj.max_depths, obj.max_parents, obj.max_tables,
                    cfg["lora_alpha"],
                )
                obj._lora_map[id(mod)] = key
        gl_state = torch.load(os.path.join(path, "graph_layers.pt"), map_location="cpu")
        obj.graph_layers.load_state_dict(gl_state, strict=True)

        obj._row_ids = obj._col_ids = obj._node_ids = None
        obj._depth_ids = obj._parent_ids = obj._table_ids = None
        obj._hooks = []
        for name, mod in obj.model.named_modules():
            if isinstance(mod, LoraLinear):
                obj._hooks.append(mod.register_forward_hook(obj._hook))

        obj._load_added_token_rows(path)
        return obj
