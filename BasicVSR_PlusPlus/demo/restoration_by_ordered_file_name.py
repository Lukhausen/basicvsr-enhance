# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import shutil

import cv2
import mmcv
import numpy as np
import torch

from mmedit.apis import init_model, restoration_video_inference
from mmedit.core import tensor2img # This should be mmedit.core.utils.image_utils.tensor2img
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
        print(f"Error: Input directory '{input_dir}' not found for preprocessing. Exiting.")
        exit() # Exit if input dir not found
    except NotADirectoryError:
        print(f"Error: '{input_dir}' is not a directory for preprocessing. Exiting.")
        exit() # Exit if not a directory

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
            new_filename = new_filename_stem + original_ext if original_ext else new_filename_stem + ".png" # Default to .png if no ext
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
    if not image_files and not (skipped_count > 0 and renamed_count == 0):
        print(f"No image files processed in {input_dir}.")


def main():
    args = parse_args()

    print("--- Starting Pre-processing: Renaming Input Frames (if applicable) ---")
    preprocess_rename_input_frames(args.input_dir, args.start_idx, args.filename_tmpl)
    print("--- Finished Pre-processing ---")

    if args.device < 0 or not torch.cuda.is_available():
        device = torch.device('cpu')
        print("Warning: CUDA not available or device < 0. Using CPU. This might be very slow.")
    else:
        device = torch.device('cuda', args.device)
        print(f"Using CUDA device: {args.device}")

    model = init_model(
        args.config, args.checkpoint, device=device)

    print(f"Starting video restoration inference for input: '{args.input_dir}'")
    # Ensure that filename_tmpl and start_idx are correctly used by restoration_video_inference
    # to read the (potentially) renamed files.
    output = restoration_video_inference(model, args.input_dir,
                                         args.window_size, args.start_idx,
                                         args.filename_tmpl, args.max_seq_len)
    print("Video restoration inference complete.")

    if output is None or output.numel() == 0:
        print("Error: Output tensor from model is None or empty. Cannot proceed to save.")
        return
    
    # CRUCIAL DEBUGGING: Check the range of the output tensor from the model
    print(f"Output tensor stats: min={output.min().item():.4f}, max={output.max().item():.4f}, mean={output.mean().item():.4f}, std={output.std().item():.4f}, dtype={output.dtype}, shape={output.shape}")

    if not os.path.splitext(args.output_dir)[1] in VIDEO_EXTENSIONS:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Ensured output directory for frames exists: {args.output_dir}")

    file_extension = os.path.splitext(args.output_dir)[1].lower()
    if file_extension in VIDEO_EXTENSIONS:
        print(f"Saving output as video to: {args.output_dir}")
        
        if output.ndim == 5: # (1, T, C, H, W)
            num_frames = output.size(1)
            h, w = output.shape[-2:]
        elif output.ndim == 4: # (T, C, H, W)
            num_frames = output.size(0)
            h, w = output.shape[-2:]
            output = output.unsqueeze(0) # Ensure 5D for consistent slicing output[:, i, ...]
            print("Info: Model output was 4D, unsqueezed to 5D for video saving.")
        else:
            print(f"Error: Unexpected output tensor shape: {output.shape}. Expected 4D or 5D for video saving.")
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if file_extension == '.mov':
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
        video_writer = cv2.VideoWriter(args.output_dir, fourcc, 25, (w, h)) # Assuming 25 FPS
        
        for i in range(0, num_frames): # i is 0 to T-1
            img_tensor_slice = output[:, i, :, :, :] # Slice is (1, C, H, W)
            
            # --- Using default tensor2img call, like in a simpler script ---
            # This assumes mmedit.core.tensor2img handles the model's output range correctly by default
            # or that the output range is compatible with its default expectations (e.g. [0,255] float or [0,1] float)
            img_uint8 = tensor2img(img_tensor_slice)
            # The tensor2img from mmedit.core.utils.image_utils should return np.uint8 by default
            
            video_writer.write(img_uint8) 
        
        cv2.destroyAllWindows()
        video_writer.release()
        print(f"Successfully saved video to {args.output_dir}")
    else: # save as frames
        print(f"Saving output as individual frames to directory: {args.output_dir}")
        
        if output.ndim == 5: # (1, T, C, H, W)
            num_model_output_frames = output.size(1) # T
        elif output.ndim == 4: # (T, C, H, W)
            num_model_output_frames = output.size(0) # T
            output = output.unsqueeze(0) # Ensure 5D for consistent slicing
            print("Info: Model output was 4D for frame saving, unsqueezed to 5D.")
        else:
            print(f"Error: Unexpected output tensor shape: {output.shape}. Expected 4D or 5D for frame saving.")
            return

        # Loop to match original frame indexing for filenames,
        # but use 0-based index for slicing the output tensor.
        for frame_num_for_filename in range(args.start_idx, args.start_idx + num_model_output_frames):
            tensor_slice_idx = frame_num_for_filename - args.start_idx # This is 0, 1, 2, ... T-1

            output_slice = output[:, tensor_slice_idx, :, :, :] # Slice is (1, C, H, W)
            
            # --- Using default tensor2img call, like in a simpler script ---
            img_to_save_uint8 = tensor2img(output_slice)
            # The tensor2img from mmedit.core.utils.image_utils should return np.uint8 by default

            # Debug: print stats for the slice before tensor2img
            # print(f"Frame {tensor_slice_idx} slice stats: min={output_slice.min().item():.4f}, max={output_slice.max().item():.4f}")
            # Debug: print stats for the image after tensor2img
            # print(f"Frame {tensor_slice_idx} img_to_save_uint8 stats: min={img_to_save_uint8.min()}, max={img_to_save_uint8.max()}, dtype={img_to_save_uint8.dtype}")

            save_path_i = os.path.join(args.output_dir, args.filename_tmpl.format(frame_num_for_filename))
            
            mmcv.imwrite(img_to_save_uint8, save_path_i)
        print(f"Successfully saved {num_model_output_frames} frames to {args.output_dir}")

if __name__ == '__main__':
    main()