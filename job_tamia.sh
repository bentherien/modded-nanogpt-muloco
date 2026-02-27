#!/bin/bash
#SBATCH --account=aip-irina
#SBATCH --gres=gpu:h200:8
#SBATCH --mem=950G
#SBATCH --cpus-per-task=64
#SBATCH --time=3:00:00
#SBATCH --job-name=muloco_speedrun
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# ============================================================
# MuLoCo outer optimizer + modded-nanogpt speedrun on Tamia (8xH200)
# ============================================================

# --- Environment setup ---
module load python/3.11.5 cuda/12.6 gcc arrow/23.0.1 thrift/0.22.0
source ~/scratch/modded-nanogpt-muloco/venv/bin/activate

# --- Proxy for internet access (needed on tamia) ---
export http_proxy=http://proxy.tamia.alliancecan.ca:3128
export https_proxy=http://proxy.tamia.alliancecan.ca:3128

# --- Env vars ---
export WANDB_API_KEY=9fa3792d90a05640029b7725e310b9904ac00119
export USE_WANDB=1
export WANDB_PROJECT=modded-nanogpt-muloco
export DATA_PATH=~/scratch/modded-nanogpt-muloco
export TIKTOKEN_CACHE_DIR=~/scratch/modded-nanogpt-muloco/tiktoken_cache
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# --- MuLoCo outer optimizer hyperparams (override via sbatch env) ---
export USE_OUTER_OPTIMIZER=${USE_OUTER_OPTIMIZER:-1}
export OUTER_LR=${OUTER_LR:-0.5}
export OUTER_MOMENTUM=${OUTER_MOMENTUM:-0.5}
export SYNC_INTERVAL=${SYNC_INTERVAL:-5}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-"tamia_olr${OUTER_LR}_omom${OUTER_MOMENTUM}_H${SYNC_INTERVAL}"}

cd ~/scratch/modded-nanogpt-muloco

mkdir -p logs

echo "Starting MuLoCo speedrun on Tamia (8xH200)"
echo "USE_OUTER_OPTIMIZER=$USE_OUTER_OPTIMIZER OUTER_LR=$OUTER_LR OUTER_MOMENTUM=$OUTER_MOMENTUM SYNC_INTERVAL=$SYNC_INTERVAL"

torchrun --standalone --nproc_per_node=8 train_gpt.py
