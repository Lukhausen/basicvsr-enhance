#!/usr/bin/env bash
set -e

# 1) System deps
apt-get update
apt-get install -y git wget python3-pip python3-venv

# 2) Create & activate Python venv
python3 -m venv vsrenv
source vsrenv/bin/activate

# 3) Upgrade core packaging tools
pip install --upgrade pip wheel packaging

# 4) Pin NumPy to <2.0 for BasicSR compatibility
pip install "numpy<2.0"

# 5) Install PyTorch matching your CUDA (here cu117 as on RunPod)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117

# 6) Install BasicSR, MMEngine & OpenMIM
pip install basicsr mmengine openmim

# 7) Install mmcv-full (needed by BasicSR)
mim install mmcv-full

# 8) Download pretrained BasicVSR++ REDS4 weights
mkdir -p checkpoints
wget -qO checkpoints/basicvsr_plusplus_reds4.pth \
  https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/\
basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth

echo
echo "✔️  install.sh complete!"
echo "   Activate environment with:  source vsrenv/bin/activate"
