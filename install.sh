#!/usr/bin/env bash
set -e

# 1) System deps
apt-get update
apt-get install -y git wget python3-pip python3-venv

# 2) Python virtual environment
python3 -m venv vsrenv
source vsrenv/bin/activate

# 3) Upgrade pip & wheel
pip install --upgrade pip wheel

# 4) **Pin NumPy to 1.x** for OpenCV / mmcv compat
pip install "numpy<2.0"

# 5) Install PyTorch + torchvision (match your CUDA; here cu117)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117

# 6) Install BasicSR (prerequisite)
pip install basicsr

# 7) Clone the official BasicVSR++ repo
git clone https://github.com/ckkelvinchan/basicvsr_plusplus.git
cd basicvsr_plusplus

# 8) Install runtime requirements (mmcv-full, OpenCV≤4.5.4, etc.)
pip install -r requirements/runtime.txt

# 9) Download the REDS4 checkpoint
mkdir -p checkpoints
wget -qO checkpoints/basicvsr_plusplus_c64n7_8x1_600k_reds4.pth \
  https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/\
basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth

cd ..

echo
echo "✔️  install.sh complete!"
echo "   Activate with: source vsrenv/bin/activate"
