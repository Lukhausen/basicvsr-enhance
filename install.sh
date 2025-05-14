#!/usr/bin/env bash
# Installation script for BasicVSR++ environment on RunPod (Docker/container friendly)
set -e

# 1. Install system dependencies (already root, no sudo)
apt-get update && \
apt-get install -y git wget python3-pip python3-venv

# 2. Create and activate a Python virtual environment
python3 -m venv vsrenv
source vsrenv/bin/activate

# 3. Upgrade pip, wheel, packaging
pip install --upgrade pip wheel packaging

# 4. Install PyTorch + torchvision (match CUDA in container)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117

# 5. Install BasicSR (provides BasicVSR++ implementation)
pip install basicsr

# 6. Install OpenMIM and mmcv-full (for config utilities)
pip install openmim
mim install mmcv-full

# 7. Download pretrained BasicVSR++ weights
mkdir -p checkpoints
wget -O checkpoints/basicvsr_plusplus_reds4.pth \
    https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth

# Done

echo "Installation complete! Activate environment with:
  source vsrenv/bin/activate"