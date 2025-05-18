#!/usr/bin/env bash
# Updated install.sh – works with host CUDA 11.8 (no sudo needed)
set -euo pipefail

echo "1. System update & prerequisites"
apt-get update -qq
apt-get install -y --no-install-recommends \
        bzip2 ca-certificates curl wget git && \
    rm -rf /var/lib/apt/lists/*

echo "2. Install Miniconda to /opt/miniconda3 (skip if already present)"
if [[ ! -d /opt/miniconda3 ]]; then
  curl -fsSL -o /tmp/miniconda.sh \
       https://repo.anaconda.com/miniconda/Miniconda3-py39_24.3.0-0-Linux-x86_64.sh
  bash /tmp/miniconda.sh -b -p /opt/miniconda3
  rm /tmp/miniconda.sh
fi
# shell hook for conda
eval "$(/opt/miniconda3/bin/conda shell.bash hook)"

echo "3. Create & activate the 'zero' environment"
if ! conda info --envs | grep -q '^zero'; then
  conda create -y -n zero python=3.9
fi
conda activate zero

echo "4. Python packages (pin CUDA-compatible builds)"

# Remove any mismatching wheels first (ignore errors if not installed)
pip uninstall -y torch torchvision torchaudio numpy || true

# ---- PyTorch & friends compiled for CUDA 11.8 ----
pip install --extra-index-url https://download.pytorch.org/whl/cu118 \
  torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0

# ---- Keep NumPy below 2 to avoid ABI conflicts ----
pip install "numpy<2" --force-reinstall

# ---- General requirements ----
pip install -U pip wheel
pip install -r requirements.txt            # your project requirements

# ---- OpenMMLab stack -------------------------------------------------
pip install -U openmim                     # provides the `mim` CLI

# pre-built mmcv-full that matches Torch 2.2 + cu118
mim install mmcv-full==2.1.0 -f \
  https://download.openmmlab.com/mmcv/dist/cu118/torch2.2.0/index.html

echo
echo "✅  All dependencies installed."
echo "   Activate the environment with:  conda activate zero"
