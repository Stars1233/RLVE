#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 WANDB_PROJECT"
    exit 1
fi

WANDB_PROJECT=$1

bash scripts/training/Nemotron-Research-Reasoning-Qwen-1.5B-v2/rlve.sh "${WANDB_PROJECT}" \
    "[Nemotron-Research-Reasoning-Qwen-1.5B-v2]_[num-environment=4]" \
    "Division EuclidGame Multiplication Sorting"
