#!/usr/bin/env bash
set -e

# 1) System deps
apt-get update
apt-get install -y git wget python3-pip python3-venv

# 2) Python venv
python3 -m venv vsrenv
source vsrenv/bin/activate

# 3) Core tooling
pip install --upgrade pip

# 4) PyTorch (match your CUDA; here cu117 as on RunPod)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117

# 5) BasicSR (dependency), plus any extras
pip install basicsr

# 6) Clone the *official* BasicVSR++ code
git clone https://github.com/ckkelvinchan/basicvsr_plusplus.git
cd basicvsr_plusplus

# 7) Install its requirements
pip install -r requirements.txt

# 8) Download the matching REDS4 checkpoint
mkdir -p checkpoints
wget -qO checkpoints/basicvsr_plusplus_c64n7_8x1_600k_reds4.pth \
  https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/\
basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth

cd ..

echo
echo "✔️  install.sh complete!"
echo "   Activate environment with: source vsrenv/bin/activate"
