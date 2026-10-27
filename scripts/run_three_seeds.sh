#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET=${1:?Usage: run_three_seeds.sh hitab|tatqa|finqa|multihiertt [training options]}
shift
case "$DATASET" in hitab|tatqa|finqa|multihiertt) ;; *) exit 2 ;; esac
OUTPUT_BASE=${OUTPUT:-${SCRIPT_DIR}/../output/${DATASET}}
for SEED in 42 43 44; do
  SEED="$SEED" OUTPUT="${OUTPUT_BASE}/seed_${SEED}" bash "${SCRIPT_DIR}/train_${DATASET}_qwen3.sh" "$@"
done
