#!/usr/bin/env bash
set -euo pipefail

# 1) system deps
apt-get update
apt-get install -y git wget python3-pip python3-venv

# 2) python venv
python3 -m venv venv
source venv/bin/activate

# 3) core python tooling
pip install --upgrade pip wheel packaging

# 4) pin numpy for BasicSR compatibility
pip install "numpy<2.0"

# 5) install PyTorch (CUDA 11.7)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117

# 6) install BasicSR, MMEngine & OpenMIM
pip install basicsr mmengine openmim

# 7) install mmcv-full via OpenMIM
mim install mmcv-full

# 8) vendor only the base‐config file we need
mkdir -p configs/_base_/models
wget -qO configs/_base_/models/basicvsr_plusplus.py \
  https://raw.githubusercontent.com/open-mmlab/mmediting/master/configs/_base_/models/basicvsr_plusplus.py

# 9) download pretrained weights
mkdir -p checkpoints
wget -qO checkpoints/basicvsr_plusplus_reds4.pth \
  https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/\
basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth

echo
echo "✅  setup complete!"
echo "   source venv/bin/activate"
