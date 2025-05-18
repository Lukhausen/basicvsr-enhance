#!/usr/bin/env bash
set -e  # exit on any error

echo "1. System update & prerequisites"
apt update && apt upgrade -y
apt install -y wget curl ca-certificates bzip2

echo "2. Install Miniconda (to /opt/miniconda3) if missing"
if [ ! -d "/opt/miniconda3" ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O ~/miniconda.sh
  bash ~/miniconda.sh -b -p /opt/miniconda3
  rm ~/miniconda.sh
  # Initialize in current shell
  eval "$(/opt/miniconda3/bin/conda shell.bash hook)"
  conda init bash
else
  echo "   Miniconda already installed."
  eval "$(/opt/miniconda3/bin/conda shell.bash hook)"
fi

echo "3. Create & activate 'zero' environment"
if ! conda env list | grep -q '^zero'; then
  conda create -n zero python=3.9 -y
fi

echo "   To activate this env in any new shell, run:"
echo "     source ~/.bashrc           # if not already done"
echo "     conda activate zero"

# Activate now for script-run installs
conda activate zero

echo "4. Install demo dependencies in 'zero'"

# -----------------------------------------------------
# 4.0  PyTorch (install BEFORE mmcv-full / mim)
# -----------------------------------------------------
# CUDA-12.x wheel
pip install --quiet --no-cache-dir torch==2.2.2+cu121 torchvision==0.17.2+cu121 \
  torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# If your image is still on CUDA-11.8, comment the line above and use:
# pip install --quiet --no-cache-dir torch==2.2.2+cu118 torchvision==0.17.2+cu118 \
#   torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118

# -----------------------------------------------------
# 4.1  OpenMIM & MMCV
# -----------------------------------------------------
pip install --quiet openmim
mim install mmcv-full  # now succeeds because Torch is present

# -----------------------------------------------------
# 4.2  BasicVSR++ demo
# -----------------------------------------------------
git clone --depth 1 https://github.com/ckkelvinchan/BasicVSR_PlusPlus.git
cd BasicVSR_PlusPlus
pip install -v -e .
mkdir -p chkpts
wget -q https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth \
     -O chkpts/basicvsr_plusplus_reds4.pth


echo "All done!  You’re ready to run BasicVSR++ demos inside the 'zero' env."
