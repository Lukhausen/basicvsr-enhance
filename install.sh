#!/usr/bin/env bash
set -euo pipefail

echo "1. System update & prerequisites"
apt-get update -qq
# Added build-essential for C++/CUDA compilation needed for mmcv-full or other packages
apt-get install -y --no-install-recommends \
        bzip2 ca-certificates curl wget git build-essential && \
    rm -rf /var/lib/apt/lists/*

echo "2. Install Miniconda to /opt/miniconda3 (skip if already present)"
CONDA_INSTALL_PATH="/opt/miniconda3"
if [[ ! -d "${CONDA_INSTALL_PATH}" ]]; then
  # Using a fixed, known-good version of Miniconda installer for Python 3.9 based envs
  # Pinned version: Miniconda3-py39_24.3.0-0-Linux-x86_64.sh
  curl -fsSL -o /tmp/miniconda.sh \
       https://repo.anaconda.com/miniconda/Miniconda3-py39_24.3.0-0-Linux-x86_64.sh
  bash /tmp/miniconda.sh -b -p "${CONDA_INSTALL_PATH}"
  rm /tmp/miniconda.sh
else
  echo "Miniconda already installed at ${CONDA_INSTALL_PATH}."
fi
# Initialize Conda for the current script execution
eval "$(${CONDA_INSTALL_PATH}/bin/conda shell.bash hook)"
# Initialize Conda for future shell sessions (writes to .bashrc)
# conda init bash # Not strictly needed for the script itself if eval is used, but good for user.
# Re-sourcing .bashrc to make conda available immediately if conda init was run in a previous script.
# However, direct eval is more reliable for the current script.

echo "3. Create & activate the 'zero' Conda environment"
ENV_NAME="zero"
if ! conda info --envs | grep -qw "^${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.9
else
  echo "Conda environment '${ENV_NAME}' already exists. Re-activating."
fi
conda activate "${ENV_NAME}"

echo "4. Installing Python packages for BasicVSR++"

# Uninstall potentially conflicting packages from previous runs to ensure a clean state
pip uninstall -y torch torchvision torchaudio numpy mmcv mmcv-full mmedit mmediting || true

# Install PyTorch 2.2.0 for CUDA 11.8
# This should match the CUDA toolkit version (11.8) found by nvcc during compilation.
echo "Installing PyTorch 2.2.0 (for CUDA 11.8)..."
pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Install NumPy < 2.0 to avoid ABI issues with older vision packages
echo "Installing NumPy < 2.0..."
pip install "numpy<2.0" --force-reinstall

# Install OpenMIM (useful for MMLab ecosystem, though mmedit will handle mmcv-full here)
pip install openmim

# Install mmedit 0.15.1 (required by original BasicVSR++)
# This will pull in a compatible mmcv-full (should be <1.6.0, e.g., 1.5.3).
# mmcv-full will likely be compiled from source if no pre-built wheel matches PT 2.2+cu118.
echo "Installing mmedit==0.15.1. This will install a compatible mmcv-full (likely compiling it)..."
pip install mmedit==0.15.1

# Get the installed mmcv-full version for BasicVSR++ setup.py adjustment
# Use python -m pip show to be more robust
INSTALLED_MMCV_FULL_VERSION=$(python -m pip show mmcv-full | grep Version | awk '{print $2}')
if [ -z "$INSTALLED_MMCV_FULL_VERSION" ]; then
    echo "Error: mmcv-full was not installed correctly (expected as a dependency of mmedit)."
    echo "Please check the output of 'pip install mmedit==0.15.1'."
    # As a fallback, try to install a common compatible version if mmedit didn't pull it.
    echo "Attempting to install mmcv-full==1.5.3, which may require compilation."
    pip install mmcv-full==1.5.3
    INSTALLED_MMCV_FULL_VERSION=$(python -m pip show mmcv-full | grep Version | awk '{print $2}')
    if [ -z "$INSTALLED_MMCV_FULL_VERSION" ]; then
        echo "Critical Error: mmcv-full could not be installed. Aborting."
        exit 1
    fi
fi
echo "Currently installed mmcv-full version: $INSTALLED_MMCV_FULL_VERSION"

# Verify installations (optional, for debugging)
echo "Verifying installed package versions..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}, CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python -c "import mmcv; print(f'MMCV (mmcv-full) version: {mmcv.__version__}')"
python -c "import mmedit; print(f'MMEdit version: {mmedit.__version__}')"

echo "5. Clone BasicVSR++ repository and prepare for install"
TARGET_REPO_DIR="BasicVSR_PlusPlus"
if [ -d "${TARGET_REPO_DIR}" ] && [ -d "${TARGET_REPO_DIR}/.git" ]; then
    echo "${TARGET_REPO_DIR} directory already exists and is a git repository. Using existing."
    (cd "${TARGET_REPO_DIR}" && git pull) # Optional: update if already cloned
else
    echo "Cloning BasicVSR_PlusPlus repository..."
    rm -rf "${TARGET_REPO_DIR}" # Remove if it's not a git repo or incomplete
    git clone https://github.com/ckkelvinchan/BasicVSR_PlusPlus.git "${TARGET_REPO_DIR}"
fi
cd "${TARGET_REPO_DIR}"

echo "Adjusting BasicVSR_PlusPlus setup.py for installed mmcv-full version ${INSTALLED_MMCV_FULL_VERSION}..."
# BasicVSR++ setup.py requests 'mmcv-full>=1.7.0', which conflicts with mmedit 0.15.1's need for mmcv-full < 1.6.0.
# We adjust it to 'mmcv-full==${INSTALLED_MMCV_FULL_VERSION}' to match what mmedit installed.
# This uses a temporary backup of setup.py.
cp setup.py setup.py.bak
sed -i.bak -E "s/'mmcv-full>=[0-9]+\.[0-9]+\.[0-9]+(rc[0-9]+)?'/'mmcv-full==${INSTALLED_MMCV_FULL_VERSION}'/" setup.py
# Verify the change (optional)
# echo "Diff for setup.py:"
# diff setup.py.bak setup.py || true

echo "Installing BasicVSR_PlusPlus..."
pip install -v -e .

# Restore original setup.py (optional, good practice)
# mv setup.py.bak setup.py

echo "6. Download pre-trained weights for BasicVSR++ demo"
mkdir -p chkpts
CHECKPOINT_FILE="chkpts/basicvsr_plusplus_reds4.pth"
CHECKPOINT_URL="https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth"
if [ ! -f "${CHECKPOINT_FILE}" ]; then
    echo "Downloading BasicVSR++ REDS4 checkpoint from ${CHECKPOINT_URL}..."
    wget "${CHECKPOINT_URL}" -O "${CHECKPOINT_FILE}"
else
    echo "Checkpoint ${CHECKPOINT_FILE} already exists. Skipping download."
fi
cd .. # Go back to the original directory (e.g., basicvsr-enhance)

echo
echo "✅ BasicVSR++ setup script finished."
echo "   The script attempted to install BasicVSR++ with its original dependencies,"
echo "   including compiling mmcv-full if necessary and adjusting BasicVSR++'s setup.py."
echo
echo "   To activate the environment in a NEW shell:"
echo "     eval \"\$(${CONDA_INSTALL_PATH}/bin/conda shell.bash hook)\"  # Or source ~/.bashrc after running 'conda init bash' once"
echo "     conda activate ${ENV_NAME}"
echo "   Then navigate to ${TARGET_REPO_DIR} (e.g., cd ${TARGET_REPO_DIR}) to run demos, for example:"
echo "     python demo/restoration_video_demo.py configs/basicvsr_plusplus/basicvsr_plusplus_reds4.py chkpts/basicvsr_plusplus_reds4.pth ../input_video.mp4 ../output_video.mp4"
echo "     (Ensure you have an 'input_video.mp4' in the parent directory or adjust paths)"