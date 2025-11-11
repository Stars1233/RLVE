#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 WANDB_PROJECT"
    exit 1
fi

WANDB_PROJECT=$1

bash scripts/training/DeepScaleR-1.5B-Preview/rlve.sh "${WANDB_PROJECT}" \
    "[DeepScaleR-1.5B-Preview]_[num-environment=16]" \
    "Division EuclidGame GCDOne_Counting HamiltonianPath LampChanging LargestConvexPolygon Multiplication PCPPermutation Path_NoGoingBack_Counting SAT ShortestPath Sorting SpiralMatrix SubsequenceReversalLNDS UndamagedSubmatrixCounting WYRLevelingGround"
