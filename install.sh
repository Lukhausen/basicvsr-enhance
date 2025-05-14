#!/usr/bin/env bash
# Installation script for BasicVSR++ environment on Ubuntu
set -e

# 1. Update and install system dependencies
sudo apt-get update
sudo apt-get install -y git wget python3-pip python3-venv

# 2. Create and activate a virtual environment
python3 -m venv vsrenv
source vsrenv/bin/activate

# 3. Upgrade pip and install OpenMIM
pip install --upgrade pip
pip install openmim

# 4. Install mmcv-full via OpenMIM
mim install mmcv-full

# 5. Clone BasicVSR++ and install as editable package
git clone https://github.com/open-mmlab/mmediting.git
cd mmediting
pip install -v -e .  # installs mmediting, mmcv, etc.
cd ..

# 6. Download pretrained BasicVSR++ weights
mkdir -p checkpoints
wget -O checkpoints/basicvsr_plusplus_reds4.pth \
    https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth

echo "\nInstallation complete!\nActivate with: source vsrenv/bin/activate"
