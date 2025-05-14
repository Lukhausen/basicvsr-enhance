#!/usr/bin/env bash
# Installation script for BasicVSR++ environment (Docker/container friendly)
set -e

# 1. Install system dependencies (without sudo for Docker)
apt-get update && \
apt-get install -y git wget python3-pip python3-venv

# 2. Create and activate a virtual environment
python3 -m venv vsrenv
source vsrenv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install PyTorch (ensure CUDA support matches your container)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117

# 5. Install OpenMIM
echo "Installing OpenMIM..."
pip install openmim

# 6. Install mmcv-full via OpenMIM
echo "Installing mmcv-full..."
mim install mmcv-full

# 7. Clone BasicVSR++ repository and install as editable package
echo "Cloning mmediting..."
git clone https://github.com/open-mmlab/mmediting.git
cd mmediting
pip install -v -e .
cd ..

# 8. Download pretrained BasicVSR++ weights
echo "Downloading pretrained weights..."
mkdir -p checkpoints
wget -O checkpoints/basicvsr_plusplus_reds4.pth \
    https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth

echo "
Installation complete! Activate with: source vsrenv/bin/activate"