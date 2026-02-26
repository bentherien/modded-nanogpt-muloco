#!/bin/bash
# ============================================================
# Hyperparameter sweep for MuLoCo outer optimizer on Tamia (8xH200)
# Submit with: bash sweep_tamia.sh
# ============================================================

# Phase 1: Sweep outer_lr and outer_momentum with H=5
for olr in 0.3 0.5 0.7 1.0; do
    for omom in 0.3 0.5 0.7; do
        export OUTER_LR=$olr
        export OUTER_MOMENTUM=$omom
        export SYNC_INTERVAL=5
        export WANDB_RUN_NAME="tamia_olr${olr}_omom${omom}_H5"
        echo "Submitting: olr=$olr omom=$omom H=5"
        sbatch --export=ALL job_tamia.sh
    done
done

# Phase 1b: Also test H=2 and H=10 with promising defaults
for h in 2 10; do
    export OUTER_LR=0.5
    export OUTER_MOMENTUM=0.5
    export SYNC_INTERVAL=$h
    export WANDB_RUN_NAME="tamia_olr0.5_omom0.5_H${h}"
    echo "Submitting: olr=0.5 omom=0.5 H=$h"
    sbatch --export=ALL job_tamia.sh
done

# Baseline (no outer optimizer)
export USE_OUTER_OPTIMIZER=0
export WANDB_RUN_NAME="tamia_baseline"
echo "Submitting: baseline (no outer optimizer)"
sbatch --export=ALL job_tamia.sh
