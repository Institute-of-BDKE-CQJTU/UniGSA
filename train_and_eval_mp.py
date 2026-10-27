"""
GraphLoRA Training + Evaluation with **Naive Model Parallelism**.

Unlike `train_and_eval.py` (which uses FSDP and replicates a copy of the
trainable wrappers per rank), this script holds **one** copy of the model
parameters split across multiple GPUs via HuggingFace `accelerate`'s
`device_map`. It is single-process (no torchrun / no DDP) and is designed for
long-context workloads such as MultiHiertt at `max_length=4096` where each
rank can no longer fit a full activation graph.

Layout:
  * Base LLM transformer blocks are pipelined across all visible GPUs.
  * `embed_tokens`, `special_encoder`, `hier_gat`            → first GPU
  * Each per-layer `HierGraph2DLoRA`                        → same GPU as
    its corresponding LoRA module (auto-detected after dispatch).

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m UniGSA.train_and_eval_mp \
      --model_name_or_path /path/to/model \
      --dataset_type multihiertt \
      --data_path TTQA_dataset/MultiHiertt/train.json \
      --output_dir ./output/graphlora_mhtt_mp \
      --max_length 4096 \
      --batch_size 1 --grad_accum 8 --bf16 \
      --eval_every_epoch
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import math
from transformers import set_seed, get_cosine_schedule_with_warmup

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, AutoModelForCausalLM

from peft.tuners.lora.layer import Linear as LoraLinear

from UniGSA.core.tokenize_utils import get_all_special_tokens, GraphQADataset
from UniGSA.core.model import GraphLoRAModel, detect_decoder_layer_class
from UniGSA.core.collator import GraphLoRACollator
from UniGSA.datasets import DATASET_REGISTRY, get_spec

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Results JSON helpers (same as train_and_eval.py)
# ============================================================================

def _init_results_json(args, path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"args": vars(args), "evaluations": []}


def _save_results_json(results, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)


def _append_eval_result(results, path, trigger, global_step, epoch, train_loss, metrics):
    entry = {
        "trigger": trigger, "global_step": global_step, "epoch": epoch,
        "train_loss": round(train_loss, 6), "metrics": metrics,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    results["evaluations"].append(entry)
    _save_results_json(results, path)
    return entry


# ============================================================================
# Naive model parallel dispatch
# ============================================================================

def _build_balanced_device_map(inner_model,
                               decoder_layer_cls,
                               n_gpus: int) -> dict:
    """Manually build an *evenly-balanced* device_map by spreading the
    transformer decoder layers round-robin across all visible GPUs.

    `infer_auto_device_map` packs everything onto GPU 0 when the model is
    much smaller than per-GPU memory (e.g. a 4B model on 48 GB cards). We
    instead force a true pipeline split: every decoder layer is one shard,
    layer i lands on GPU (i * n_gpus // n_layers).
    """
    # Find all decoder layers and their dotted module paths.
    layer_paths = []
    for name, module in inner_model.named_modules():
        if isinstance(module, decoder_layer_cls):
            layer_paths.append(name)
    if not layer_paths:
        raise RuntimeError(
            f"No modules of type {decoder_layer_cls.__name__} found - "
            "cannot build balanced device map.")

    # Sanity: layers should share a common parent (e.g. ".layers.0", ".layers.1", ...).
    n_layers = len(layer_paths)
    assignments: dict = {}
    for i, p in enumerate(layer_paths):
        gpu = min(i * n_gpus // n_layers, n_gpus - 1)
        assignments[p] = gpu

    # Pin embed/norm/lm_head etc. to GPU 0 (input) and last GPU (output).
    # Anything not yet assigned goes on GPU 0 by default; lm_head goes to last GPU
    # so the final logits don't have to round-trip back.
    top_level = set()
    for n, _ in inner_model.named_children():
        top_level.add(n)

    # Walk the named_modules and assign top-level modules that aren't decoder layers.
    layer_prefixes = tuple(p + "." for p in layer_paths) + tuple(layer_paths)

    def _is_under_layer(name: str) -> bool:
        return any(name == p or name.startswith(p + ".") for p in layer_paths)

    last_gpu = n_gpus - 1
    for name, module in inner_model.named_modules():
        if name == "" or _is_under_layer(name):
            continue
        # Top-level only (depth 1) — coarser placements; sub-modules inherit.
        depth = name.count(".")
        if depth > 3:
            continue
        # Heuristic placements
        lname = name.lower()
        if any(t in lname for t in ["embed_tokens", "wte"]):
            assignments[name] = 0
        elif "lm_head" in lname:
            assignments[name] = last_gpu
        elif lname.endswith("norm") or lname.endswith(".final_layernorm"):
            assignments[name] = last_gpu

    return assignments


def _dispatch_graphlora_mp(model: GraphLoRAModel,
                           decoder_layer_cls,
                           compute_dtype: torch.dtype,
                           reserve_first_gpu_gib: float = 4.0):
    """Place the LLM (with LoRA) evenly across GPUs, then co-locate graph
    submodules with their respective LoRA layers."""
    from accelerate import dispatch_model

    inner = model.model  # PeftModel wrapping the LLM
    n_gpus = torch.cuda.device_count()

    device_map = _build_balanced_device_map(inner, decoder_layer_cls, n_gpus)
    # Log a compact summary instead of the (possibly huge) full map.
    layer_per_gpu = {}
    for k, v in device_map.items():
        layer_per_gpu.setdefault(v, []).append(k)
    logger.info(f"Balanced device_map: " + ", ".join(
        f"GPU{g}={len(ks)} modules" for g, ks in sorted(layer_per_gpu.items())))
    for g, ks in sorted(layer_per_gpu.items()):
        sample = ks[:3] + (["..."] if len(ks) > 3 else [])
        logger.info(f"  GPU{g} ← {sample}")

    # `dispatch_model` adds device-transition hooks; this is the heart of MP.
    dispatch_model(inner, device_map=device_map)

    # Identify embed device and place graph-side modules there.
    embed = model._get_embed_layer()
    try:
        embed_dev = next(embed.parameters()).device
    except StopIteration:
        embed_dev = torch.device("cuda:0")
    if embed_dev.type == "meta":
        embed_dev = torch.device("cuda:0")
    logger.info(f"Embed layer device: {embed_dev}")

    # IMPORTANT: keep trainable graph/LoRA submodules in fp32 (autocast will
    # handle bf16 compute). Casting them to bf16 makes AdamW updates explode
    # to NaN within a few steps for LoRA-style finetuning.
    model.special_encoder.to(embed_dev)
    model.hier_gat.to(embed_dev)

    # Each per-layer HierGraph2DLoRA → device of its corresponding LoRA module.
    for name, mod in model.model.named_modules():
        if isinstance(mod, LoraLinear):
            key = model._lora_map.get(id(mod))
            if key is None:
                continue
            try:
                lora_dev = next(mod.parameters()).device
            except StopIteration:
                lora_dev = embed_dev
            if lora_dev.type == "meta":
                lora_dev = embed_dev
            model.graph_layers[key].to(lora_dev)

    # Force every *trainable* parameter to fp32 master weights for stable
    # optimization. Frozen base-LLM weights stay in bf16 to save memory.
    for n, p in model.named_parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()

    return embed_dev


# ============================================================================
# Evaluator dispatch (single-process)
# ============================================================================

def _build_evaluator_kwargs(args, eval_data_path):
    return {
        "data_path": eval_data_path,
        "max_new_tokens": args.eval_max_new_tokens,
        "max_samples": args.eval_max_samples,
        "use_reasoning": args.reasoning_path is not None,
        "use_gold_text": False,
        "use_gold_evidence": False,
        "max_length": args.max_length,
        "add_cross_table_edges": args.add_cross_table_edges,
    }


def _resolve_eval_data_path(args):
    if args.eval_data_path:
        return args.eval_data_path
    if args.dataset_type == "hitab":
        return None
    if args.data_path:
        candidate = args.data_path.replace("train.json", "dev.json")
        if os.path.exists(candidate):
            return candidate
    raise ValueError("No dev split found; set --eval_data_path explicitly")


def _run_eval(model, tokenizer, args, device, eval_output_file):
    spec = get_spec(args.dataset_type)
    eval_kwargs = _build_evaluator_kwargs(args, _resolve_eval_data_path(args))
    eval_kwargs["device"] = str(device)
    eval_kwargs["output_file"] = eval_output_file

    if args.dataset_type == "hitab":
        eval_kwargs.pop("data_path", None)
        eval_kwargs["enhanced_prompt"] = getattr(args, "enhanced_prompt", False)
        return spec.evaluator(
            model, tokenizer,
            args.hitab_dir or "GraphOTTER-main/dataset/hitab",
            split=args.eval_split,
            **eval_kwargs,
        )
    return spec.evaluator(model, tokenizer, **eval_kwargs)


def _run_eval_inplace(model, tokenizer, args, embed_dev,
                      trigger, global_step, epoch, train_loss,
                      results, results_json_path):
    model.eval()
    eval_output_file = os.path.join(args.output_dir, f"eval_predictions_{trigger}.json")
    metrics = _run_eval(model, tokenizer, args, embed_dev, eval_output_file)
    _append_eval_result(results, results_json_path, trigger, global_step,
                        epoch, train_loss, metrics)
    model.train()


# ============================================================================
# Main
# ============================================================================

def train_and_eval(args):
    set_seed(args.seed)
    if args.learning_rate is None:
        args.learning_rate = 3e-5 if args.dataset_type == "hitab" else 5e-5
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires CUDA + multiple GPUs.")
    n_gpus = torch.cuda.device_count()
    logger.info(f"GraphLoRA Train+Eval (Naive MP) | visible_gpus={n_gpus}")
    logger.info(f"Dataset: {args.dataset_type}")

    spec = get_spec(args.dataset_type)
    if args.max_length <= 0:
        args.max_length = spec.default_max_length

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": get_all_special_tokens()})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ---- Compute dtype ----
    if args.bf16:
        compute_dtype = torch.bfloat16
    elif args.fp16:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    # ---- Base LLM (load on CPU first to allow embedding resize before dispatch) ----
    logger.info("Loading base LLM on CPU (will be dispatched across GPUs after wrapping)...")
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=compute_dtype,
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    base.resize_token_embeddings(len(tokenizer))
    logger.info(f"Model type: {base.config.model_type}")

    decoder_layer_cls = detect_decoder_layer_class(base)
    logger.info(f"Decoder layer class: {decoder_layer_cls.__name__}")

    if args.gradient_checkpointing:
        base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()
        logger.info("Gradient checkpointing enabled")

    # ---- Dataset ----
    loader_kwargs = dict(
        max_length=args.max_length,
        max_samples=args.max_samples,
        split_oversized=args.split_oversized,
        max_no_answer_ratio=args.max_no_answer_ratio,
        reasoning_path=args.reasoning_path,
        use_gold_text=args.use_gold_text,
        use_gold_evidence=args.use_gold_evidence,
        add_cross_table_edges=args.add_cross_table_edges,
        enhanced_prompt=args.enhanced_prompt,
    )
    if args.dataset_type == "hitab":
        hdir = args.hitab_dir or "GraphOTTER-main/dataset/hitab"
        sp = args.data_path or os.path.join(hdir, "train_samples_processed.jsonl")
        tp = args.tables_path or os.path.join(hdir, "tables.jsonl")
        samples = spec.loader(sp, tokenizer, tables_path=tp, **loader_kwargs)
    else:
        if not args.data_path:
            raise ValueError(f"--data_path is required for dataset '{args.dataset_type}'")
        samples = spec.loader(args.data_path, tokenizer, **loader_kwargs)
    logger.info(f"Loaded {len(samples)} training samples")

    if not samples:
        raise ValueError("No training samples were loaded")
    ds = GraphQADataset(samples)
    collator = GraphLoRACollator(tokenizer.pad_token_id)
    dl = DataLoader(ds, args.batch_size, shuffle=True, collate_fn=collator,
                    num_workers=args.num_workers, pin_memory=True)

    # ---- Build GraphLoRA model on CPU ----
    target_mods = args.target_modules.split(",") if args.target_modules else None
    model = GraphLoRAModel(
        base, tokenizer, args.lora_rank, args.lora_alpha, args.lora_dropout,
        target_mods, args.max_rows, args.max_cols, args.max_nodes,
        args.max_depths, args.max_parents, args.max_tables,
        args.gat_rank, args.gat_dropout,
    )
    _, trainable, total = model.get_trainable_parameters()
    logger.info(f"Params: {total:,} total, {trainable:,} trainable "
                f"({trainable/total*100:.3f}%)")

    # ---- Naive Model Parallel dispatch ----
    embed_dev = _dispatch_graphlora_mp(
        model, decoder_layer_cls, compute_dtype,
        reserve_first_gpu_gib=args.reserve_gpu0_gib,
    )

    # Optimizer / scheduler (parameters live on multiple devices; AdamW handles that)
    opt = AdamW([p for p in model.parameters() if p.requires_grad],
                lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_ep = max(math.ceil(len(dl) / args.grad_accum), 1)
    total_steps = max(steps_per_ep * args.num_epochs, 1)
    sched = get_cosine_schedule_with_warmup(
        opt, int(round(args.warmup_ratio * total_steps)), total_steps)

    results_data = _init_results_json(args, args.results_json)
    _save_results_json(results_data, args.results_json)
    eff_bs = args.batch_size * args.grad_accum
    logger.info(f"Training: {args.num_epochs} ep, bs={args.batch_size}, "
                f"eff_bs={eff_bs}, parallel=NaiveMP({n_gpus} GPUs)")

    # autocast: forward in bf16/fp16 while master weights stay fp32 for the
    # trainable submodules (frozen base is already bf16 → effectively pure bf16
    # compute everywhere, with stable fp32 master weights for the optimizer).
    autocast_enabled = compute_dtype != torch.float32
    autocast_dtype = compute_dtype if autocast_enabled else torch.float32

    # ---- Training loop ----
    model.train()
    global_step = 0
    for ep in range(args.num_epochs):
        ep_loss, nb = 0.0, 0
        for bi, batch in enumerate(dl):
            accumulation_size = min(args.grad_accum, len(dl) - (bi // args.grad_accum) * args.grad_accum)
            # All inputs go to the embed device; downstream device transitions
            # are handled by accelerate hooks + our forward-hook _hook().
            batch = {k: v.to(embed_dev, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=autocast_dtype,
                                enabled=autocast_enabled):
                loss = model(**batch).loss / accumulation_size
            loss.backward()
            ep_loss += loss.item() * accumulation_size
            nb += 1

            if (bi + 1) % args.grad_accum == 0 or bi + 1 == len(dl):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.max_grad_norm,
                )
                opt.step()
                sched.step()
                opt.zero_grad()
                global_step += 1

                if args.eval_every_steps > 0 and global_step % args.eval_every_steps == 0:
                    avg_loss = ep_loss / max(nb, 1)
                    _run_eval_inplace(model, tokenizer, args, embed_dev,
                                      f"step_{global_step}", global_step, ep + 1, avg_loss,
                                      results_data, args.results_json)

            if (bi + 1) % args.log_interval == 0:
                logger.info(f"Ep {ep+1}/{args.num_epochs} step {bi+1}/{len(dl)} "
                            f"(g:{global_step}) loss={loss.item()*accumulation_size:.4f} "
                            f"avg={ep_loss/nb:.4f}")

        avg_ep_loss = ep_loss / max(nb, 1)
        logger.info(f"Epoch {ep+1} done. avg_loss={avg_ep_loss:.4f}")

        if args.save_every_epoch or ep == args.num_epochs - 1:
            sp = os.path.join(args.output_dir, f"checkpoint-ep{ep+1}")
            model.save_graphlora(sp)
            logger.info(f"Saved checkpoint: {sp}")

        if args.eval_every_epoch:
            _run_eval_inplace(model, tokenizer, args, embed_dev,
                              f"epoch_{ep+1}", global_step, ep + 1, avg_ep_loss,
                              results_data, args.results_json)

    fp = os.path.join(args.output_dir, "final")
    model.save_graphlora(fp)
    tokenizer.save_pretrained(os.path.join(args.output_dir, "tokenizer"))
    logger.info(f"Done! Saved to {fp}")
    if results_data and results_data["evaluations"]:
        primary = spec.primary_metric
        best = max(results_data["evaluations"],
                   key=lambda e: e["metrics"].get(primary, -1))
        logger.info(f"Best {primary}: {best['metrics'].get(primary, 0):.2f} "
                    f"at {best['trigger']}")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--target_modules", default="k_proj,v_proj")

    # GraphLoRA hyper-params
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--max_rows", type=int, default=600)
    p.add_argument("--max_cols", type=int, default=50)
    p.add_argument("--max_nodes", type=int, default=2000)
    p.add_argument("--max_depths", type=int, default=20)
    p.add_argument("--max_parents", type=int, default=2000)
    p.add_argument("--max_tables", type=int, default=16)
    p.add_argument("--gat_rank", type=int, default=32)
    p.add_argument("--gat_dropout", type=float, default=0.1)

    # Dataset
    p.add_argument("--dataset_type", default="multihiertt",
                   choices=list(DATASET_REGISTRY.keys()) or ["multihiertt"])
    p.add_argument("--data_path", default=None)
    p.add_argument("--hitab_dir", default=None)
    p.add_argument("--tables_path", default=None)
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--max_length", type=int, default=-1)
    p.add_argument("--split_oversized", action="store_true")
    p.add_argument("--max_no_answer_ratio", type=float, default=1.0)
    p.add_argument("--reasoning_path", default=None)
    p.add_argument("--use_gold_text", action="store_true", default=False)
    p.add_argument("--no_gold_text", action="store_false", dest="use_gold_text")
    p.add_argument("--use_gold_evidence", action="store_true", default=False)
    p.add_argument("--no_gold_evidence", action="store_false", dest="use_gold_evidence")
    p.add_argument("--add_cross_table_edges", action="store_true", default=True)
    p.add_argument("--no_cross_table_edges", action="store_false", dest="add_cross_table_edges")
    p.add_argument("--enhanced_prompt", action="store_true", default=False,
                   help="(HiTab) Use enhanced prompt with header summary for better cell localization")

    # Training
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing", action="store_false", dest="gradient_checkpointing")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--output_dir", default="./output/graphlora_mp")
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_every_epoch", action="store_true")

    # Naive MP knobs
    p.add_argument("--reserve_gpu0_gib", type=float, default=4.0,
                   help="Headroom (GiB) reserved on GPU 0 for embeds/graph/activations.")

    # Eval
    p.add_argument("--eval_every_epoch", action="store_true")
    p.add_argument("--eval_every_steps", type=int, default=0)
    p.add_argument("--eval_split", default="dev", choices=["train", "dev", "test"])
    p.add_argument("--eval_data_path", default=None)
    p.add_argument("--eval_max_samples", type=int, default=-1)
    p.add_argument("--eval_max_new_tokens", type=int, default=256)

    # Misc
    p.add_argument("--results_json", default=None)

    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.grad_accum < 1 or args.batch_size < 1:
        p.error("batch_size and grad_accum must be positive")
    if args.results_json is None:
        args.results_json = os.path.join(args.output_dir, "train_eval_results.json")
    return args


if __name__ == "__main__":
    train_and_eval(parse_args())
