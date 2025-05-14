#!/usr/bin/env bash
set -euo pipefail

# 1) System packages
apt-get update
apt-get install -y git wget python3-pip python3-venv python3-wheel

# 2) Create & activate venv
python3 -m venv vsrenv
source vsrenv/bin/activate

# 3) Core Python tooling
pip install --upgrade pip
pip install wheel packaging

# 4) Pin NumPy <2 for BasicSR
pip install "numpy<2.0"

# 5) Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117

# 6) Install BasicSR & MMEngine & OpenMIM
pip install basicsr mmengine openmim

# 7) Install mmcv-full via OpenMIM
mim install mmcv-full

# 8) Pull in MMEditing repo (for base‐config files)
git clone --depth 1 https://github.com/open-mmlab/mmediting.git mmediting

# 9) Link MMEditing’s _base_ dir into your configs/
mkdir -p configs
ln -s ../mmediting/configs/_base configs/_base

# 10) Download pretrained weights
mkdir -p checkpoints
wget -qO checkpoints/basicvsr_plusplus_reds4.pth \
  https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/\
basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth

echo
echo "✔️  install.sh complete!"
echo "   Activate with:  source vsrenv/bin/activate"
