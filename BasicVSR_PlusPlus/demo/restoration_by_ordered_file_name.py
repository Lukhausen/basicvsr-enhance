# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import shutil

import cv2
import mmcv # mmcv-full is typically needed for Config
import numpy as np
import torch

from mmedit.apis import init_model, restoration_video_inference
from mmedit.core import tensor2img
from mmedit.utils import modify_args

VIDEO_EXTENSIONS = ('.mp4', '.mov')
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def parse_args():
    modify_args()
    parser = argparse.ArgumentParser(description='Restoration demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('input_dir', help='directory of the input video or frames')
    parser.add_argument('output_dir', help='directory of the output video or frames')
    parser.add_argument(
        '--start-idx',
        type=int,
        default=0,
        help='index corresponds to the first frame of the sequence (for input renaming and processing)')
    parser.add_argument(
        '--filename-tmpl',
        default='{:08d}.png',
        help='template of the file names (used for input renaming and output saving)')
    parser.add_argument(
        '--window-size',
        type=int,
        default=0,
        help='window size if sliding-window framework is used')
    parser.add_argument(
        '--max-seq-len',
        type=int,
        default=None,
        help='maximum sequence length if recurrent framework is used')
    parser.add_argument(
        '--output-scale', # New argument
        type=int,
        default=None, # Default to None, meaning use config file's scale
        help='Target scale factor for the output. Set to 1 for no upscaling. '
             'If set, tries to override model config. '
             'If None, uses the scale from the model config file.'
    )
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args


def preprocess_rename_input_frames(input_dir, start_idx, filename_tmpl):
    if not os.path.isdir(input_dir):
        if os.path.isfile(input_dir) and input_dir.lower().endswith(VIDEO_EXTENSIONS):
            print(f"Info: Input '{input_dir}' is a video file. Skipping frame renaming pre-processing.")
        else:
            print(f"Warning: Input '{input_dir}' is not a directory. Skipping frame renaming pre-processing.")
        return

    print(f"Preprocessing: Attempting to rename files in '{input_dir}' to match format '{filename_tmpl}' starting from index {start_idx}.")
    
    try:
        files_in_dir = os.listdir(input_dir)
    except FileNotFoundError:
        print(f"Error: Input directory '{input_dir}' not found for preprocessing.")
        return
    except NotADirectoryError:
        print(f"Error: '{input_dir}' is not a directory for preprocessing.")
        return

    image_files = sorted([
        f for f in files_in_dir 
        if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(IMAGE_EXTENSIONS)
    ])

    if not image_files:
        print(f"No image files (with extensions {IMAGE_EXTENSIONS}) found in '{input_dir}' to rename.")
        return

    print(f"Found {len(image_files)} image files. IMPORTANT: Ensure these files were sorted correctly for your sequence.")
    
    renamed_count = 0
    skipped_count = 0
    for i, old_filename in enumerate(image_files):
        current_frame_idx = start_idx + i
        new_filename_stem = filename_tmpl.split('.')[0].format(current_frame_idx)
        template_ext = os.path.splitext(filename_tmpl)[1]
        if not template_ext:
            original_ext = os.path.splitext(old_filename)[1]
            new_filename = new_filename_stem + original_ext
        else:
            new_filename = filename_tmpl.format(current_frame_idx)

        old_filepath = os.path.join(input_dir, old_filename)
        new_filepath = os.path.join(input_dir, new_filename)

        if old_filepath == new_filepath:
            skipped_count +=1
            continue

        try:
            os.rename(old_filepath, new_filepath)
            renamed_count += 1
        except OSError as e:
            print(f"Error renaming '{old_filename}' to '{new_filename}': {e}")
    
    if renamed_count > 0:
        print(f"Successfully renamed {renamed_count} files in '{input_dir}'.")
    if skipped_count > 0 and renamed_count == 0 :
        print(f"All {skipped_count} files in '{input_dir}' already matched the target naming scheme. No files were renamed.")
    elif skipped_count > 0 :
         print(f"{skipped_count} files already matched the target naming scheme and were skipped.")
    if not image_files and not (skipped_count > 0 and renamed_count == 0): # Avoid double message if all skipped
        print(f"No image files processed in {input_dir}.")


def main():
    args = parse_args()

    print("--- Starting Pre-processing: Renaming Input Frames (if applicable) ---")
    preprocess_rename_input_frames(args.input_dir, args.start_idx, args.filename_tmpl)
    print("--- Finished Pre-processing ---")

    # Load configuration
    cfg = mmcv.Config.fromfile(args.config)

    # --- Attempt to override scale if --output-scale is provided ---
    if args.output_scale is not None:
        print(f"Attempting to override model scale to: {args.output_scale} based on --output-scale argument.")
        scale_overridden = False
        # Common paths for 'scale' in mmedit model configurations
        if hasattr(cfg, 'model') and 'scale' in cfg.model:
            original_scale = cfg.model.get('scale')
            cfg.model.scale = args.output_scale
            print(f"  Overridden cfg.model.scale from {original_scale} to {cfg.model.scale}")
            scale_overridden = True
        elif hasattr(cfg, 'model') and hasattr(cfg.model, 'generator') and 'scale' in cfg.model.generator:
            original_scale = cfg.model.generator.get('scale')
            cfg.model.generator.scale = args.output_scale
            print(f"  Overridden cfg.model.generator.scale from {original_scale} to {cfg.model.generator.scale}")
            scale_overridden = True
        # Add other potential paths if you know them, e.g., for specific model types
        # elif hasattr(cfg, 'scale'): # Less common for complex models but possible
        #     original_scale = cfg.get('scale')
        #     cfg.scale = args.output_scale
        #     print(f"  Overridden top-level cfg.scale from {original_scale} to {cfg.scale}")
        #     scale_overridden = True
        
        if not scale_overridden:
            print(f"  Warning: Could not automatically find and override a 'scale' attribute in the model config using common paths.")
            print(f"  The script will proceed using the scale defined in '{args.config}'.")
            print(f"  For no upscaling (scale=1), please ensure 'scale=1' is set in your model definition (e.g., model.generator.scale=1) within '{args.config}'.")
        if args.output_scale == 1:
            print("  Aiming for same input/output resolution (no upscaling).")
        else:
            print(f"  Aiming for output scale x{args.output_scale}.")
    else:
        print(f"Using scale factor defined in the configuration file: '{args.config}'.")
        print("If you want no upscaling, ensure 'scale=1' is set in the config or use the '--output-scale 1' argument.")
    # --- End of scale override attempt ---

    if args.device < 0 or not torch.cuda.is_available():
        device = torch.device('cpu')
        print("Warning: CUDA not available or device < 0. Using CPU. This might be very slow.")
    else:
        device = torch.device('cuda', args.device)
        print(f"Using CUDA device: {args.device}")

    # Initialize the model using the (potentially modified) cfg object
    model = init_model(
        cfg,  # Use the cfg object
        args.checkpoint,
        device=device
    )

    print(f"Starting video restoration inference for input: '{args.input_dir}'")
    output = restoration_video_inference(model, args.input_dir,
                                         args.window_size, args.start_idx,
                                         args.filename_tmpl, args.max_seq_len)
    print("Video restoration inference complete.")

    if output is None or output.numel() == 0:
        print("Error: Output from model is empty. Cannot save.")
        return

    if not os.path.splitext(args.output_dir)[1] in VIDEO_EXTENSIONS:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Ensured output directory for frames exists: {args.output_dir}")

    file_extension = os.path.splitext(args.output_dir)[1].lower()
    if file_extension in VIDEO_EXTENSIONS:
        print(f"Saving output as video to: {args.output_dir}")
        
        # Determine H, W from the model's output tensor
        if output.ndim == 5: # (1, T, C, H, W)
            num_frames = output.size(1)
            h, w = output.shape[-2:]
        elif output.ndim == 4: # (T, C, H, W)
            num_frames = output.size(0)
            h, w = output.shape[-2:]
            output = output.unsqueeze(0) # Add batch dim for consistency
        else:
            print(f"Error: Unexpected output tensor shape: {output.shape}. Expected 4D or 5D.")
            return
        
        print(f"Output video resolution will be: {w}x{h} (based on model output)")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if file_extension == '.mov':
            fourcc = cv2.VideoWriter_fourcc(*'avc1')

        video_writer = cv2.VideoWriter(args.output_dir, fourcc, 25, (w, h))
        
        for i in range(0, num_frames):
            img_tensor_frame = output[:, i, :, :, :]
            # Adjust min_max based on your model's output range. Common defaults.
            img = tensor2img(img_tensor_frame, min_max=(-1, 1) if model.cfg.model.get('pixel_mean') or model.cfg.model.get('bgr_mean') else (0, 255))
            video_writer.write(img.astype(np.uint8))
        
        cv2.destroyAllWindows()
        video_writer.release()
        print(f"Successfully saved video to {args.output_dir}")
    else:
        print(f"Saving output as individual frames to directory: {args.output_dir}")
        
        if output.ndim == 5:
            num_frames_output = output.size(1)
        elif output.ndim == 4:
            num_frames_output = output.size(0)
            output = output.unsqueeze(0)
        else:
            print(f"Error: Unexpected output tensor shape: {output.shape}. Expected 4D or 5D.")
            return

        # Output frames are indexed 0 to N-1. Name them starting from args.start_idx.
        for i in range(0, num_frames_output):
            output_frame_tensor = output[:, i, :, :, :]
            img = tensor2img(output_frame_tensor, min_max=(-1, 1) if model.cfg.model.get('pixel_mean') or model.cfg.model.get('bgr_mean') else (0, 255))
            save_frame_idx = args.start_idx + i 
            save_path_i = os.path.join(args.output_dir, args.filename_tmpl.format(save_frame_idx))
            mmcv.imwrite(img.astype(np.uint8), save_path_i)
        print(f"Successfully saved {num_frames_output} frames to {args.output_dir}")

if __name__ == '__main__':
    main()