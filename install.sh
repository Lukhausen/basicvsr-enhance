#!/usr/bin/env bash
set -euo pipefail

echo "1. System update & prerequisites"
apt-get update -qq
# build-essential might still be needed if any minor dependency compiles, but major ones should be wheels
apt-get install -y --no-install-recommends \
        bzip2 ca-certificates curl wget git build-essential && \
    rm -rf /var/lib/apt/lists/*

echo "2. Install Miniconda to /opt/miniconda3 (skip if already present)"
CONDA_INSTALL_PATH="/opt/miniconda3"
if [[ ! -d "${CONDA_INSTALL_PATH}" ]]; then
  curl -fsSL -o /tmp/miniconda.sh \
       https://repo.anaconda.com/miniconda/Miniconda3-py39_24.3.0-0-Linux-x86_64.sh
  bash /tmp/miniconda.sh -b -p "${CONDA_INSTALL_PATH}"
  rm /tmp/miniconda.sh
else
  echo "Miniconda already installed at ${CONDA_INSTALL_PATH}."
fi
eval "$(${CONDA_INSTALL_PATH}/bin/conda shell.bash hook)"
# To ensure conda command is available in .bashrc for future interactive shells:
# Check if conda init has already been run for bash
if ! grep -q "# >>> conda initialize >>>" ~/.bashrc; then
    echo "Initializing Conda for bash..."
    conda init bash
    # The following source is to make conda available in the current script session
    # if conda init was just run, as it modifies .bashrc which needs to be reloaded.
    # source ~/.bashrc # This can be problematic in non-interactive scripts.
    # eval is generally safer for the current script.
fi


echo "3. Create & activate the 'zero' Conda environment"
ENV_NAME="zero"
if ! conda info --envs | grep -qw "^${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.9
else
  echo "Conda environment '${ENV_NAME}' already exists. Re-activating."
fi
conda activate "${ENV_NAME}"

echo "4. Installing Python packages for BasicVSR++ (Colab-inspired, pre-built wheels)"

# Uninstall potentially conflicting packages
pip uninstall -y torch torchvision torchaudio numpy mmcv mmcv-full mmedit mmediting || true

# Install PyTorch 1.10.2+cu111 (compatible with CUDA 11.8 runtime and mmcv-full 1.4.8 wheel)
# Wheel URL: https://download.pytorch.org/whl/cu111/torch_stable.html
echo "Installing PyTorch 1.10.2+cu111..."
pip install torch==1.10.2+cu111 torchvision==0.11.3+cu111 torchaudio==0.10.2+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html

# Install NumPy < 2.0
echo "Installing NumPy < 2.0..."
pip install "numpy<2.0" --force-reinstall

# Install mmcv-full==1.4.8 (pre-compiled wheel for PyTorch 1.10.x, CUDA 11.1, Python 3.9)
MMCV_FULL_WHEEL_URL="https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/mmcv_full-1.4.8-cp39-cp39-manylinux1_x86_64.whl"
echo "Installing mmcv-full==1.4.8 from wheel: ${MMCV_FULL_WHEEL_URL}"
pip install "${MMCV_FULL_WHEEL_URL}"
INSTALLED_MMCV_FULL_VERSION="1.4.8" # We know this version

# Install OpenMIM (though not strictly used for mmcv-full installation here, BasicVSR setup might use it for other things or info)
pip install openmim==0.1.5 # Using version from Colab

# mmedit will be installed as a dependency of BasicVSR_PlusPlus via its setup.py

# Verify installations (optional, for debugging)
echo "Verifying installed package versions..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}, CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python -c "import mmcv; print(f'MMCV (mmcv-full) version: {mmcv.__version__}')"


echo "5. Clone BasicVSR++ repository and prepare for install"
TARGET_REPO_DIR="BasicVSR_PlusPlus"
if [ -d "${TARGET_REPO_DIR}" ] && [ -d "${TARGET_REPO_DIR}/.git" ]; then
    echo "${TARGET_REPO_DIR} directory already exists. Using existing."
    (cd "${TARGET_REPO_DIR}" && git pull)
else
    echo "Cloning BasicVSR_PlusPlus repository..."
    rm -rf "${TARGET_REPO_DIR}"
    git clone https://github.com/ckkelvinchan/BasicVSR_PlusPlus.git "${TARGET_REPO_DIR}"
fi
cd "${TARGET_REPO_DIR}"

echo "Adjusting BasicVSR_PlusPlus setup.py for installed mmcv-full version ${INSTALLED_MMCV_FULL_VERSION}..."
# Patch setup.py to require the mmcv-full version we just installed, overriding 'mmcv-full>=1.7.0'
cp setup.py setup.py.bak
sed -i.bak -E "s/'mmcv-full>=[0-9]+\.[0-9]+\.[0-9]+(rc[0-9]+)?'/'mmcv-full==${INSTALLED_MMCV_FULL_VERSION}'/" setup.py
echo "Diff for setup.py after patching:"
diff setup.py.bak setup.py || true


echo "Installing BasicVSR_PlusPlus and its dependencies (like mmedit==0.14.0)..."
pip install -v -e .


echo "6. Download pre-trained weights for BasicVSR++ demo"
mkdir -p chkpts
CHECKPOINT_FILE="chkpts/basicvsr_plusplus_reds4.pth"
CHECKPOINT_URL="https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth"
if [ ! -f "${CHECKPOINT_FILE}" ]; then
    echo "Downloading BasicVSR++ REDS4 checkpoint from ${CHECKPOINT_URL}..."
    wget --progress=bar:force:noscroll "${CHECKPOINT_URL}" -O "${CHECKPOINT_FILE}"
else
    echo "Checkpoint ${CHECKPOINT_FILE} already exists. Skipping download."
fi
cd .. # Go back to the original directory

echo
echo "✅ BasicVSR++ setup script finished using pre-built wheels strategy."
echo
echo "   To activate the environment in a NEW shell:"
echo "     eval \"\$(${CONDA_INSTALL_PATH}/bin/conda shell.bash hook)\"  # Or source ~/.bashrc if 'conda init bash' has been run"
echo "     conda activate ${ENV_NAME}"
echo "   Then navigate to ${TARGET_REPO_DIR} (e.g., cd ${TARGET_REPO_DIR}) to run demos."
echo "   For example, to run the video restoration demo (ensure you have an input video):"
echo "     cp ../my_input_video.mp4 ."
echo "     python demo/restoration_video_demo.py configs/basicvsr_plusplus/basicvsr_plusplus_reds4.py chkpts/basicvsr_plusplus_reds4.pth my_input_video.mp4 results/output_video.mp4"