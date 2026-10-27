#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_PARENT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PACKAGE_PARENT}${PYTHONPATH:+:${PYTHONPATH}}"
: "${MODEL:?Set MODEL to a local pretrained model directory}"
DATA=${DATA:-${PACKAGE_PARENT}/GraphLoRA_v2/TTQA_dataset/TAT-QA/train.json}
OUTPUT=${OUTPUT:-${SCRIPT_DIR}/../output/tatqa}
SEED=${SEED:-42}
GPUS=${GPUS:-0}
CUDA_VISIBLE_DEVICES="${GPUS}" conda run --no-capture-output -n xq_py311 python -m UniGSA.train_and_eval_v2 \
  --model_name_or_path "${MODEL}" --dataset_type tatqa --data_path "${DATA}" \
  --output_dir "${OUTPUT}" --seed "${SEED}" \
  --target_modules k_proj,v_proj --lora_rank 8 --lora_alpha 16 --gat_rank 32 \
  --num_epochs 3 --batch_size 1 --grad_accum 2 --learning_rate 5e-5 \
  --warmup_ratio 0.03 --max_length 2048 --max_tables 16 \
  --no_gold_text --no_gold_evidence --eval_every_epoch \
  --eval_max_new_tokens 256 --save_every_epoch --bf16 "$@"
