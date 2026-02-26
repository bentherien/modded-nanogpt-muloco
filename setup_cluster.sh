#!/bin/bash
# ============================================================
# Setup script for modded-nanogpt-muloco on Alliance Canada clusters
# Run this once on each cluster to set up the environment
# Usage: bash setup_cluster.sh
# ============================================================

set -e

SCRATCH_DIR=~/scratch/modded-nanogpt-muloco

echo "=== Setting up modded-nanogpt-muloco on $(hostname) ==="

# Clone or update the repo
if [ -d "$SCRATCH_DIR" ]; then
    echo "Directory exists, pulling latest..."
    cd $SCRATCH_DIR
    git pull
else
    echo "Cloning repo..."
    git clone https://github.com/bentherien/modded-nanogpt-muloco.git $SCRATCH_DIR
    cd $SCRATCH_DIR
fi

# Load modules
module load python/3.11.5 cuda/12.6 gcc arrow/23.0.1 thrift/0.22.0

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi
source venv/bin/activate

# Install dependencies (use --no-index for compute nodes)
echo "Installing dependencies..."
pip install --no-index torch numpy triton wandb 2>/dev/null || pip install torch numpy triton wandb
pip install tqdm huggingface-hub kernels setuptools typing-extensions 2>/dev/null || true

# Create tiktoken cache dir
mkdir -p tiktoken_cache
export TIKTOKEN_CACHE_DIR=$SCRATCH_DIR/tiktoken_cache

# Download data (may need proxy on tamia)
if [ ! -d "data/fineweb10B" ]; then
    echo "Downloading FineWeb data (first 9 shards = 900M tokens)..."
    # On tamia, set proxy first:
    # export http_proxy=http://proxy.tamia.alliancecan.ca:3128
    # export https_proxy=http://proxy.tamia.alliancecan.ca:3128
    python data/cached_fineweb10B.py 9
else
    echo "Data already exists"
fi

mkdir -p logs

echo "=== Setup complete! ==="
echo "To run a single job: sbatch job_fir.sh (or job_tamia.sh)"
echo "To run sweep: bash sweep_fir.sh (or sweep_tamia.sh)"
