"""UniGSA paper-aligned single-GPU/FSDP training entry.

Default: K/V adapters, cosine schedule with 3% warmup, seeds selectable by
--seed. MultiHiertt uses train_and_eval_mp.py for single-process model parallel.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import math
from transformers import set_seed, get_cosine_schedule_with_warmup
from functools import partial

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from UniGSA.core.tokenize_utils import get_all_special_tokens, GraphQADataset
from UniGSA.core.model import (
    GraphLoRAModel, detect_decoder_layer_class, save_graphlora_fsdp,
)
from UniGSA.core.collator import GraphLoRACollator
from UniGSA.datasets import DATASET_REGISTRY, get_spec

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Distributed setup
# ============================================================================

def setup_distributed():
    rank = int(os.environ.get("LOCAL_RANK", -1))
    if rank == -1:
        return 0, 1, False
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(rank)
    return rank, dist.get_world_size(), True


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
# Dataset-aware evaluator dispatch
# ============================================================================

def _build_evaluator_kwargs(args, eval_data_path):
    """Common kwargs shared by all evaluators."""
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
    """Pick the eval data path (dev / test split or custom)."""
    if args.eval_data_path:
        return args.eval_data_path
    if args.dataset_type == "hitab":
        return None  # hitab evaluator uses hitab_dir
    if args.data_path:
        # Try common train→dev substitutions
        for src, dst in [("train.json", "dev.json"),
                         ("train_qa.json", "dev_qa.json")]:
            candidate = args.data_path.replace(src, dst)
            if candidate != args.data_path and os.path.exists(candidate):
                return candidate
    raise ValueError("No dev split found; set --eval_data_path explicitly")


def _run_eval(model, tokenizer, args, device, eval_output_file):
    """Dispatch to the correct evaluator based on dataset spec."""
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
    if args.dataset_type == "doclibrag":
        eval_kwargs["documents_path"] = args.documents_path
    return spec.evaluator(model, tokenizer, **eval_kwargs)


# ============================================================================
# Training-time evaluation hooks
# ============================================================================

def run_evaluation_fsdp(fsdp_model, tokenizer, args, device, rank, graphlora_config,
                       trigger, global_step, epoch, train_loss,
                       results, results_json_path):
    """FSDP evaluation: save temp checkpoint, load full model on rank 0."""
    if rank == 0:
        logger.info(f"{'='*60}\nRunning evaluation (trigger: {trigger})\n{'='*60}")

    tmp_path = os.path.join(args.output_dir, f"_tmp_eval_{trigger}")
    save_graphlora_fsdp(fsdp_model, tmp_path, rank, graphlora_config)
    if rank == 0:
        tokenizer.save_pretrained(tmp_path)

    metrics = {}
    if rank == 0:
        if args.bf16:
            eval_dtype = torch.bfloat16
        elif args.fp16:
            eval_dtype = torch.float16
        else:
            eval_dtype = torch.float32

        eval_base = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, torch_dtype=eval_dtype, trust_remote_code=True)
        eval_base.resize_token_embeddings(len(tokenizer))
        eval_model = GraphLoRAModel.load_graphlora(eval_base, tokenizer, tmp_path)
        eval_model = eval_model.to(device).eval()

        eval_output_file = os.path.join(args.output_dir, f"eval_predictions_{trigger}.json")
        metrics = _run_eval(eval_model, tokenizer, args, device, eval_output_file)
        _append_eval_result(results, results_json_path, trigger, global_step,
                            epoch, train_loss, metrics)

        del eval_model, eval_base
        torch.cuda.empty_cache()

        import shutil
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)

    if dist.is_initialized():
        dist.barrier()
    return metrics


def _run_eval_single_gpu(model, tokenizer, args, device,
                         trigger, global_step, epoch, train_loss,
                         results, results_json_path):
    model.eval()
    eval_output_file = os.path.join(args.output_dir, f"eval_predictions_{trigger}.json")
    metrics = _run_eval(model, tokenizer, args, device, eval_output_file)
    _append_eval_result(results, results_json_path, trigger, global_step,
                        epoch, train_loss, metrics)
    model.train()


# ============================================================================
# Main training loop
# ============================================================================

def train_and_eval(args):
    set_seed(args.seed)
    if args.learning_rate is None:
        args.learning_rate = 3e-5 if args.dataset_type == "hitab" else 5e-5
    rank, world, distributed = setup_distributed()
    is_main = (not distributed) or rank == 0
    use_reasoning = args.reasoning_path is not None

    if is_main:
        logger.info(f"GraphLoRA Train+Eval  FSDP={distributed}, world={world}")
        logger.info(f"Dataset: {args.dataset_type}, use_reasoning={use_reasoning}")

    device = (torch.device(f"cuda:{rank}") if distributed
              else torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"))

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
    # Prefer BF16 when supported unless precision is explicitly selected.
    bf16_supported = (torch.cuda.is_available()
                      and torch.cuda.is_bf16_supported())
    if args.bf16 and args.fp16:
        if is_main:
            logger.warning("Both --bf16 and --fp16 set; preferring --bf16.")
        args.fp16 = False
    if (not args.bf16) and (not args.fp16):
        if bf16_supported:
            args.bf16 = True
            if is_main:
                logger.info("Auto-enabled bf16 (hardware supported, no flag given).")
        else:
            if is_main:
                logger.info("No mixed precision: bf16 unsupported, fp16 not requested.")
    elif args.fp16 and bf16_supported and is_main:
        logger.warning(
            "Using --fp16 on bf16-capable hardware. For numeric-reasoning "
            "tasks (FinQA / TAT-QA) --bf16 is usually more stable and avoids "
            "GradScaler-induced loss spikes.")

    if args.bf16:
        compute_dtype = torch.bfloat16
    elif args.fp16:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32
    load_dtype = torch.float32 if distributed else compute_dtype

    # ---- Base LLM ----
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=load_dtype, trust_remote_code=True)
    base.resize_token_embeddings(len(tokenizer))
    if is_main:
        logger.info(f"Model type: {base.config.model_type}")

    decoder_layer_cls = detect_decoder_layer_class(base)
    if is_main:
        logger.info(f"Decoder layer class: {decoder_layer_cls.__name__}")

    if args.gradient_checkpointing:
        base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()
        if is_main:
            logger.info("Gradient checkpointing enabled")

    # ---- Load dataset ----
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
        documents_path=args.documents_path,
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

    if is_main:
        logger.info(f"Loaded {len(samples)} training samples")

    if not samples:
        raise ValueError("No training samples were loaded")
    ds = GraphQADataset(samples)
    collator = GraphLoRACollator(tokenizer.pad_token_id)
    if distributed:
        sampler = DistributedSampler(ds, world, rank, shuffle=True, seed=args.seed)
        dl = DataLoader(ds, args.batch_size, sampler=sampler, collate_fn=collator,
                        num_workers=args.num_workers, pin_memory=True)
    else:
        # Honour the configured single-GPU data-loader worker count.
        dl = DataLoader(ds, args.batch_size, shuffle=True, collate_fn=collator,
                        num_workers=args.num_workers, pin_memory=True)

    # ---- Build GraphLoRA model ----
    target_mods = args.target_modules.split(",") if args.target_modules else ["k_proj", "v_proj"]
    model = GraphLoRAModel(
        base, tokenizer, args.lora_rank, args.lora_alpha, args.lora_dropout,
        target_mods, args.max_rows, args.max_cols, args.max_nodes,
        args.max_depths, args.max_parents, args.max_tables,
        args.gat_rank, args.gat_dropout,
    )
    if is_main:
        _, trainable, total = model.get_trainable_parameters()
        logger.info(f"Params: {total:,} total, {trainable:,} trainable "
                    f"({trainable/total*100:.3f}%)")

    graphlora_config = {
        "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "max_rows": args.max_rows, "max_cols": args.max_cols, "max_nodes": args.max_nodes,
        "max_depths": args.max_depths, "max_parents": args.max_parents,
        "max_tables": args.max_tables,
        "gat_rank": args.gat_rank, "gat_dropout": args.gat_dropout,
        "target_modules": model._target_modules,
        "special_token_ids": model.special_token_ids,
    }

    # ---- FSDP ----
    if distributed:
        auto_wrap_pol = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={decoder_layer_cls},
        )
        mp_policy = (MixedPrecision(
            param_dtype=compute_dtype, reduce_dtype=compute_dtype, buffer_dtype=compute_dtype,
        ) if compute_dtype != torch.float32 else None)
        model = FSDP(
            model, auto_wrap_policy=auto_wrap_pol, mixed_precision=mp_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=rank, use_orig_params=True,
        )
        if is_main:
            logger.info(f"FSDP wrapped: FULL_SHARD, {decoder_layer_cls.__name__}")
    else:
        model = model.to(device)

    opt = AdamW([p for p in model.parameters() if p.requires_grad],
                lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_ep = max(math.ceil(len(dl) / args.grad_accum), 1)
    total_steps = max(steps_per_ep * args.num_epochs, 1)

    # Configurable schedule; the paper default is cosine with 3% warmup.
    if args.num_warmup_steps is not None and args.num_warmup_steps >= 0:
        num_warmup = int(args.num_warmup_steps)
    else:
        num_warmup = int(round(args.warmup_ratio * total_steps))
    num_warmup = max(0, min(num_warmup, total_steps - 1))

    sched_name = (args.lr_scheduler or "cosine").lower()
    if sched_name == "cosine":
        sched = get_cosine_schedule_with_warmup(opt, num_warmup, total_steps)
    elif sched_name == "linear":
        sched = get_linear_schedule_with_warmup(opt, num_warmup, total_steps)
    elif sched_name in ("constant", "constant_with_warmup"):
        sched = get_constant_schedule_with_warmup(opt, num_warmup)
    elif sched_name == "cosine_legacy":
        # Fallback to old behaviour (no warmup) for strict reproducibility.
        sched = CosineAnnealingLR(opt, total_steps)
    else:
        raise ValueError(f"Unknown --lr_scheduler: {args.lr_scheduler}")

    if is_main:
        logger.info(f"LR scheduler: {sched_name}, total_steps={total_steps}, "
                    f"warmup_steps={num_warmup} "
                    f"({num_warmup/max(total_steps,1)*100:.1f}%)")

    use_manual_amp = (compute_dtype != torch.float32) and (not distributed) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_manual_amp and args.fp16 else None

    results_data = _init_results_json(args, args.results_json) if is_main else None
    if is_main:
        _save_results_json(results_data, args.results_json)
        eff_bs = args.batch_size * world * args.grad_accum
        logger.info(f"Training: {args.num_epochs} ep, bs/gpu={args.batch_size}, "
                    f"eff_bs={eff_bs}, parallel={'FSDP' if distributed else 'single'}")

    # ---- Training loop ----
    model.train()
    global_step = 0
    for ep in range(args.num_epochs):
        if distributed:
            sampler.set_epoch(ep)
        ep_loss, nb = 0.0, 0
        for bi, batch in enumerate(dl):
            accumulation_size = min(args.grad_accum, len(dl) - (bi // args.grad_accum) * args.grad_accum)
            batch = {k: v.to(device) for k, v in batch.items()}
            if use_manual_amp:
                with torch.amp.autocast("cuda", dtype=compute_dtype):
                    loss = model(**batch).loss / accumulation_size
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                loss = model(**batch).loss / accumulation_size
                loss.backward()
            ep_loss += loss.item() * accumulation_size; nb += 1

            last_grad_norm = None  # for logging
            if (bi + 1) % args.grad_accum == 0 or bi + 1 == len(dl):
                step_taken = True
                if scaler:
                    scaler.unscale_(opt)
                    last_grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)
                    # Detect whether GradScaler will actually take a step.
                    prev_scale = scaler.get_scale()
                    scaler.step(opt); scaler.update()
                    # If scaler skipped (inf/NaN), scale is reduced → don't
                    # advance the LR scheduler, keeping lr aligned with the
                    # actual optimiser progress.
                    if scaler.get_scale() < prev_scale:
                        step_taken = False
                else:
                    if distributed:
                        last_grad_norm = model.clip_grad_norm_(args.max_grad_norm)
                    else:
                        last_grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm)
                    opt.step()
                if step_taken:
                    sched.step()
                opt.zero_grad()
                if step_taken:
                    global_step += 1

                if args.eval_every_steps > 0 and global_step % args.eval_every_steps == 0:
                    avg_loss = ep_loss / max(nb, 1)
                    if distributed:
                        run_evaluation_fsdp(model, tokenizer, args, device, rank, graphlora_config,
                                            f"step_{global_step}", global_step, ep + 1, avg_loss,
                                            results_data, args.results_json)
                    elif is_main:
                        _run_eval_single_gpu(model, tokenizer, args, device,
                                             f"step_{global_step}", global_step, ep + 1, avg_loss,
                                             results_data, args.results_json)
                    model.train()

            if is_main and (bi + 1) % args.log_interval == 0:
                try:
                    cur_lr = sched.get_last_lr()[0]
                except Exception:
                    cur_lr = opt.param_groups[0]["lr"]
                gn_str = (f"{float(last_grad_norm):.3f}"
                          if last_grad_norm is not None else "n/a")
                logger.info(f"Ep {ep+1}/{args.num_epochs} step {bi+1}/{len(dl)} "
                            f"(g:{global_step}) loss={loss.item()*accumulation_size:.4f} "
                            f"avg={ep_loss/nb:.4f} lr={cur_lr:.2e} "
                            f"grad_norm={gn_str}")

        avg_ep_loss = ep_loss / max(nb, 1)
        if is_main:
            logger.info(f"Epoch {ep+1} done. avg_loss={avg_ep_loss:.4f}")

        if args.save_every_epoch or ep == args.num_epochs - 1:
            sp = os.path.join(args.output_dir, f"checkpoint-ep{ep+1}")
            if distributed:
                save_graphlora_fsdp(model, sp, rank, graphlora_config)
                if rank == 0:
                    tokenizer.save_pretrained(sp)
            else:
                model.save_graphlora(sp)
            if is_main:
                logger.info(f"Saved checkpoint: {sp}")

        if args.eval_every_epoch:
            if distributed:
                run_evaluation_fsdp(model, tokenizer, args, device, rank, graphlora_config,
                                    f"epoch_{ep+1}", global_step, ep + 1, avg_ep_loss,
                                    results_data, args.results_json)
            elif is_main:
                _run_eval_single_gpu(model, tokenizer, args, device,
                                     f"epoch_{ep+1}", global_step, ep + 1, avg_ep_loss,
                                     results_data, args.results_json)
            model.train()

        if distributed:
            dist.barrier()

    fp = os.path.join(args.output_dir, "final")
    if distributed:
        save_graphlora_fsdp(model, fp, rank, graphlora_config)
        if rank == 0:
            tokenizer.save_pretrained(fp)
    else:
        model.save_graphlora(fp)
    if is_main:
        tokenizer.save_pretrained(os.path.join(args.output_dir, "tokenizer"))
        logger.info(f"Done! Saved to {fp}")
        if results_data and results_data["evaluations"]:
            primary = spec.primary_metric
            best = max(results_data["evaluations"],
                       key=lambda e: e["metrics"].get(primary, -1))
            logger.info(f"Best {primary}: {best['metrics'].get(primary, 0):.2f} "
                        f"at {best['trigger']}")
    if distributed:
        dist.destroy_process_group()


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
    p.add_argument("--dataset_type", default="hitab",
                   choices=list(DATASET_REGISTRY.keys()) or ["hitab"])
    p.add_argument("--data_path", default=None)
    p.add_argument("--hitab_dir", default=None)
    p.add_argument("--tables_path", default=None)
    p.add_argument("--documents_path", default=None,
                   help="(DocLibRAG) Path to Documents.json; defaults to "
                        "<data_path dir>/Documents.json")
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--max_length", type=int, default=-1,
                   help="-1 → use dataset spec default")
    p.add_argument("--split_oversized", action="store_true")
    p.add_argument("--max_no_answer_ratio", type=float, default=1.0)
    p.add_argument("--reasoning_path", default=None)
    p.add_argument("--use_gold_text", action="store_true", default=False)
    p.add_argument("--no_gold_text", action="store_false", dest="use_gold_text")
    p.add_argument("--use_gold_evidence", action="store_true", default=False,
                   help="(MultiHiertt) Use table_evidence/text_evidence at training time")
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
    # Paper default: cosine decay with 3% warmup.
    p.add_argument("--lr_scheduler", default="cosine",
                   choices=["cosine", "linear", "constant",
                            "constant_with_warmup", "cosine_legacy"],
                   help="LR scheduler. 'cosine_legacy' = old no-warmup behaviour.")
    p.add_argument("--warmup_ratio", type=float, default=0.03,
                   help="Fraction of total steps used for linear warmup. "
                        "Ignored if --num_warmup_steps is set.")
    p.add_argument("--num_warmup_steps", type=int, default=None,
                   help="Absolute warmup steps. Overrides --warmup_ratio.")

    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing", action="store_false", dest="gradient_checkpointing")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", default="./output/graphlora")
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_every_epoch", action="store_true")

    # Eval
    p.add_argument("--eval_every_epoch", action="store_true")
    p.add_argument("--eval_every_steps", type=int, default=0)
    p.add_argument("--eval_split", default="dev", choices=["train", "dev", "test"])
    p.add_argument("--eval_data_path", default=None,
                   help="Override eval data path (otherwise auto-derived from data_path)")
    p.add_argument("--eval_max_samples", type=int, default=-1)
    p.add_argument("--eval_max_new_tokens", type=int, default=256)

    # Misc
    p.add_argument("--results_json", default=None)
    p.add_argument("--local_rank", type=int, default=-1)

    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.grad_accum < 1 or args.batch_size < 1:
        p.error("batch_size and grad_accum must be positive")
    if args.results_json is None:
        args.results_json = os.path.join(args.output_dir, "train_eval_results.json")
    return args


if __name__ == "__main__":
    train_and_eval(parse_args())
