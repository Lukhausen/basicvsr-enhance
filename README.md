# BasicVSR++ Enhanced Setup

Quick setup for BasicVSR++ (super-resolution, denoising) using pre-downloaded models.

## Prerequisites

*   Linux, `git`, Conda
*   NVIDIA GPU + CUDA (recommended)

## Quick Start

1.  **Clone & Setup Environment:**
    ```bash
    git clone https://github.com/Lukhausen/basicvsr-enhance
    cd basicvsr-enhance
    bash install.sh # Creates 'zero' conda env & installs deps
    ```

2.  **Navigate & Activate Environment:**
    ```bash
    cd BasicVSR_PlusPlus # Navigate into the main scripts directory
    source ~/.bashrc     # Or equivalent for your shell (e.g., ~/.zshrc)
    conda activate zero
    ```
    Your prompt should now show `(zero)`.

## Run Inference

Use `demo/restoration_by_ordered_file_name.py` for processing image sequences.

**Command Template:**

```bash
python demo/restoration_by_ordered_file_name.py \
    <CONFIG_FILE_PATH> \
    <CHECKPOINT_FILE_PATH> \
    <INPUT_FRAMES_DIR> \
    <OUTPUT_FRAMES_DIR>
```

**Example: 4x Super-Resolution**
(Assumes input frames are in `data/demo_000`, e.g., `00000000.png`, `00000001.png`, ...)

```bash
# Ensure (zero) env is active & you're in BasicVSR_PlusPlus directory
python demo/restoration_by_ordered_file_name.py \
    configs/basicvsr_plusplus_reds4.py \
    chkpts/basicvsr_plusplus_c64n7_8x1_600k_reds4.pth \
    data/demo_000 \
    results/demo_4x_output
```

**Example: Denoising**

```bash
python demo/restoration_by_ordered_file_name.py \
    configs/basicvsr_plusplus_denoise.py \
    chkpts/basicvsr_plusplus_denoise-28f6920c.pth \
    data/your_noisy_input_frames \
    results/denoised_output_frames
```

## Customization

*   To use different models/tasks:
    1.  Choose a `.py` config from `configs/`.
    2.  Choose its corresponding `.pth` checkpoint from `chkpts/`.
    3.  Update these paths in the command template.
*   Input frames in `<INPUT_FRAMES_DIR>` must be sequentially named (default: `00000000.png`, `00000001.png`, ...).
*   For other options (like `--start-idx`, `--filename-tmpl`, `--device`), see `python demo/restoration_by_ordered_file_name.py --help`.

## Notes

*   All model checkpoints are pre-downloaded in the `chkpts/` directory.
*   The `install.sh` script handles dependency setup.
*   Denoising model is trained for Gaussian noise. Performance on other noise types may vary.