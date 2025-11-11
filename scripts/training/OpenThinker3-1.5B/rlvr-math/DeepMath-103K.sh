#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 WANDB_PROJECT"
    exit 1
fi

WANDB_PROJECT=$1

bash scripts/training/OpenThinker3-1.5B/rlvr-math.sh "${WANDB_PROJECT}" \
    "[OpenThinker3-1.5B]_[DeepMath-103K]" \
    "DeepMath-103K"
