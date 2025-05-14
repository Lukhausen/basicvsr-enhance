#!/usr/bin/env bash
# install.sh — BasicVSR++ 4K enhancement setup on RunPod
set -e

# 1) System deps (already root)
apt-get update && \
apt-get install -y git wget python3-pip python3-venv

# 2) Python venv
python3 -m venv vsrenv
source vsrenv/bin/activate

# 3) Core tooling
pip install --upgrade pip wheel packaging

# 4) Pin NumPy for BasicSR compatibility
pip install "numpy<2.0"

# 5) PyTorch (match your container’s CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117

# 6) BasicSR (includes BasicVSR++), MMEngine, OpenMIM
pip install basicsr mmengine openmim

# 7) MMCV for config utilities
mim install mmcv-full

# 8) Download pretrained weights
mkdir -p checkpoints
wget -O checkpoints/basicvsr_plusplus_reds4.pth \
  https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/\
basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth

echo
echo "✔️  install.sh complete!"
echo "Activate with:  source vsrenv/bin/activate"
