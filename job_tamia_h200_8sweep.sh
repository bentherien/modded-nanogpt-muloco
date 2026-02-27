#!/bin/bash
#SBATCH --account=aip-irina
#SBATCH --gpus-per-node=h200:8
#SBATCH --mem=950G
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --job-name=muloco_8sweep
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# ============================================================
# Run 8 independent 1-GPU experiments in parallel on Tamia H200 node
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
export HF_HUB_OFFLINE=1

cd ~/scratch/modded-nanogpt-muloco
mkdir -p logs

run_config() {
    GPU=$1
    CONFIG=$2
    IFS=',' read -r OLR OMOM H USE_OUTER <<< "$CONFIG"
    USE_OUTER=${USE_OUTER:-1}
    export CUDA_VISIBLE_DEVICES=$GPU
    export OUTER_LR=$OLR
    export OUTER_MOMENTUM=$OMOM
    export SYNC_INTERVAL=$H
    export USE_OUTER_OPTIMIZER=$USE_OUTER
    if [ "$USE_OUTER" = "0" ]; then
        export WANDB_RUN_NAME="h200_baseline_gpu${GPU}"
    else
        export WANDB_RUN_NAME="h200_olr${OLR}_omom${OMOM}_H${H}"
    fi
    echo "GPU $GPU: olr=$OLR omom=$OMOM H=$H use_outer=$USE_OUTER"
    torchrun --standalone --nproc_per_node=1 --master_port=$((29500 + GPU)) train_gpt.py
}

CONFIG1=${CONFIG1:-"0.5,0.5,5,1"}
CONFIG2=${CONFIG2:-"0.3,0.5,5,1"}
CONFIG3=${CONFIG3:-"0.7,0.5,5,1"}
CONFIG4=${CONFIG4:-"0.5,0.3,5,1"}
CONFIG5=${CONFIG5:-"0.5,0.7,5,1"}
CONFIG6=${CONFIG6:-"0.5,0.5,2,1"}
CONFIG7=${CONFIG7:-"0.5,0.5,10,1"}
CONFIG8=${CONFIG8:-"0,0,5,0"}

echo "Running 8 configs in parallel on H200:"
run_config 0 "$CONFIG1" &
run_config 1 "$CONFIG2" &
run_config 2 "$CONFIG3" &
run_config 3 "$CONFIG4" &
run_config 4 "$CONFIG5" &
run_config 5 "$CONFIG6" &
run_config 6 "$CONFIG7" &
run_config 7 "$CONFIG8" &
wait
echo "All 8 configs complete"
