#!/bin/bash
#SBATCH --account=aip-irina
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --job-name=muloco_1gpu
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# ============================================================
# Single-GPU MuLoCo sweep on Tamia (1 GPU, grad_accum=8)
# ============================================================

module load python/3.11.5 cuda/12.6 gcc arrow/23.0.1 thrift/0.22.0
source ~/scratch/modded-nanogpt-muloco/venv/bin/activate

export http_proxy=http://proxy.tamia.alliancecan.ca:3128
export https_proxy=http://proxy.tamia.alliancecan.ca:3128
export WANDB_API_KEY=9fa3792d90a05640029b7725e310b9904ac00119
export USE_WANDB=1
export WANDB_PROJECT=modded-nanogpt-muloco
export DATA_PATH=~/scratch/modded-nanogpt-muloco
export TIKTOKEN_CACHE_DIR=~/scratch/modded-nanogpt-muloco/tiktoken_cache
export HF_DATASETS_OFFLINE=1

export USE_OUTER_OPTIMIZER=${USE_OUTER_OPTIMIZER:-1}
export OUTER_LR=${OUTER_LR:-0.5}
export OUTER_MOMENTUM=${OUTER_MOMENTUM:-0.5}
export SYNC_INTERVAL=${SYNC_INTERVAL:-5}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-"tamia1gpu_olr${OUTER_LR}_omom${OUTER_MOMENTUM}_H${SYNC_INTERVAL}"}

cd ~/scratch/modded-nanogpt-muloco
mkdir -p logs

echo "Starting 1-GPU sweep: olr=$OUTER_LR omom=$OUTER_MOMENTUM H=$SYNC_INTERVAL"
torchrun --standalone --nproc_per_node=1 train_gpt.py
