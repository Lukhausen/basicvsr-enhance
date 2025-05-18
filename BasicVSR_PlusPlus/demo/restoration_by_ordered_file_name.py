# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import glob # Added
import tempfile # Added
import shutil # Added for copy fallback

import cv2
import mmcv
import numpy as np
import torch

from mmedit.apis import init_model, restoration_video_inference
from mmedit.core import tensor2img # tensor2img might be in mmedit.utils in newer mmedit versions
from mmedit.utils import modify_args

VIDEO_EXTENSIONS = ('.mp4', '.mov')
IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')


def parse_args():
    modify_args()
    parser = argparse.ArgumentParser(description='Restoration demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('input_dir', help='directory of the input video or image frames')
    parser.add_argument('output_dir', help='directory of the output video or image frames')
    parser.add_argument(
        '--start-idx',
        type=int,
        default=0,
        help='index for the first frame of the output sequence (if saving frames)')
    parser.add_argument(
        '--filename-tmpl',
        default='{:08d}.png',
        help='template for the file names of output frames (if saving frames)')
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


def main():
    args = parse_args()

    model = init_model(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))

    # Parameters for restoration_video_inference, determined by input type
    current_input_path = args.input_dir
    current_start_idx = args.start_idx
    current_filename_tmpl = args.filename_tmpl
    
    temp_dir_manager = None  # To manage TemporaryDirectory object for cleanup

    # Path 1: Input is a directory, try to process as image frames
    if os.path.isdir(args.input_dir):
        print(f"Input path '{args.input_dir}' is a directory. Scanning for image frames...")
        
        all_items_in_dir = glob.glob(os.path.join(args.input_dir, '*'))
        image_files = sorted([
            f for f in all_items_in_dir
            if os.path.isfile(f) and f.lower().endswith(IMG_EXTENSIONS)
        ])

        if image_files:
            print(f"Found {len(image_files)} image frames. Processing them in alphabetical order.")
            
            temp_dir_manager = tempfile.TemporaryDirectory()
            temp_dir_path = temp_dir_manager.name
            
            # Standardized template for the symlinked/copied files for the model's consumption
            TEMP_INPUT_FILENAME_TMPL = '{:08d}.png' 
            print(f"Preparing temporary input files in '{temp_dir_path}' using template '{TEMP_INPUT_FILENAME_TMPL}'.")

            for idx, original_filepath in enumerate(image_files):
                temp_file_path = os.path.join(temp_dir_path, TEMP_INPUT_FILENAME_TMPL.format(idx))
                abs_original_filepath = os.path.abspath(original_filepath)
                try:
                    os.symlink(abs_original_filepath, temp_file_path)
                except OSError as e_symlink: # Fallback to copying if symlink fails
                    print(f"Warning: Symlink creation failed for '{abs_original_filepath}' (Error: {e_symlink}). Attempting to copy file...")
                    try:
                        shutil.copy2(abs_original_filepath, temp_file_path)
                    except Exception as e_copy:
                        if temp_dir_manager:
                            temp_dir_manager.cleanup()
                        raise RuntimeError(f"Failed to create symlink or copy for '{abs_original_filepath}' to '{temp_file_path}'. Error: {e_copy}") from e_copy
            
            current_input_path = temp_dir_path
            current_start_idx = 0  # Files in temp dir are 0-indexed
            current_filename_tmpl = TEMP_INPUT_FILENAME_TMPL # Template used for temp files
        else:
            # Directory exists but no image files found.
            # It might be an actual video file, or an empty/irrelevant directory.
            # We'll let the next conditions try to handle it or raise an error.
            print(f"Input path '{args.input_dir}' is a directory, but no supported image frames found. "
                  f"Will check if it's a video file or raise an error.")

    # Path 2: Input is a video file
    # This condition is checked if Path 1 did not set up a temp_dir_manager (i.e., no image frames were processed from a dir)
    # AND args.input_dir points to an actual file with a video extension.
    if temp_dir_manager is None and \
       os.path.isfile(args.input_dir) and \
       args.input_dir.lower().endswith(VIDEO_EXTENSIONS):
        print(f"Input path '{args.input_dir}' is a video file.")
        # current_input_path, current_start_idx, current_filename_tmpl are already args.input_dir etc by default.
        # This is correct for video file input.
        pass
    
    # Path 3: Invalid input (not a dir with frames, not a video file)
    elif temp_dir_manager is None: 
        # This means:
        # 1. args.input_dir was not a directory with processable image frames.
        # 2. AND args.input_dir was not a recognized video file.
        raise ValueError(
            f"Input path '{args.input_dir}' is not a recognized video file (supported: {VIDEO_EXTENSIONS}) "
            f"and not a directory containing supported image frames (supported: {IMG_EXTENSIONS}). Please check the path.")

    # Perform restoration
    print(f"Starting video restoration inference with input: '{current_input_path}'...")
    output = restoration_video_inference(model, current_input_path,
                                         args.window_size, current_start_idx,
                                         current_filename_tmpl, args.max_seq_len)
    print("Inference complete.")

    # Save output
    output_file_or_dir_path = args.output_dir
    output_file_extension = os.path.splitext(output_file_or_dir_path)[1].lower()

    if output_file_extension in VIDEO_EXTENSIONS:
        print(f"Saving output as video to: {output_file_or_dir_path}")
        
        output_video_parent_dir = os.path.dirname(output_file_or_dir_path)
        if output_video_parent_dir and not os.path.exists(output_video_parent_dir):
             os.makedirs(output_video_parent_dir, exist_ok=True)

        h, w = output.shape[-2:] # Assumes output tensor is (N, T, C, H, W)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Consider making codec configurable
        video_fps = 25 # Consider making FPS configurable or inferring
        video_writer = cv2.VideoWriter(output_file_or_dir_path, fourcc, video_fps, (w, h))
        
        num_output_frames = output.size(1)
        for i in range(num_output_frames):
            img = tensor2img(output[:, i, :, :, :]) 
            video_writer.write(img.astype(np.uint8))
         
        video_writer.release()
        # cv2.destroyAllWindows() # Usually called after cv2.imshow, may not be needed here
        print(f"Output video saved successfully to {output_file_or_dir_path}")
    else:
        print(f"Saving output as image frames to directory: {output_file_or_dir_path}")
        mmcv.mkdir_or_exist(output_file_or_dir_path)

        num_output_frames = output.size(1)
        for i in range(num_output_frames): 
            output_tensor_slice = output[:, i, :, :, :]
            img = tensor2img(output_tensor_slice)
               
            output_frame_filename = args.filename_tmpl.format(args.start_idx + i)
            save_path_i = os.path.join(output_file_or_dir_path, output_frame_filename)
            
            mmcv.imwrite(img, save_path_i)
        print(f"{num_output_frames} output frames saved in '{output_file_or_dir_path}' "
              f"using template '{args.filename_tmpl}' starting from index {args.start_idx}.")

    # Clean up temporary directory if one was used
    if temp_dir_manager:
        try:
            temp_dir_manager.cleanup()
            print(f"Temporary directory '{temp_dir_manager.name}' cleaned up successfully.")
        except Exception as e:
            print(f"Warning: Failed to clean up temporary directory '{temp_dir_manager.name}'. Error: {e}")


if __name__ == '__main__':
    main()