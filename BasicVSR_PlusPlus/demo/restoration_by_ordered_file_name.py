# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import glob # For finding files
import shutil # For moving/renaming

import cv2
import mmcv
import numpy as np
import torch

from mmedit.apis import init_model, restoration_video_inference
from mmedit.core import tensor2img # Ensure this import path is correct for your mmedit version
from mmedit.utils import modify_args

VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv')
IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')

def parse_args():
    modify_args()
    parser = argparse.ArgumentParser(description='Restoration demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('input_path', help='path to the input video or directory of image frames')
    parser.add_argument('output_path', help='path to the output video or directory for image frames')
    parser.add_argument(
        '--start-idx',
        type=int,
        default=0,
        help='index corresponds to the first frame of the input sequence (if input is frames) or the desired output start index (if saving frames)')
    parser.add_argument(
        '--filename-tmpl',
        default='{:08d}.png',
        help='template for file names (used for reading input frames from a dir and for writing output frames)')
    parser.add_argument(
        '--window-size',
        type=int,
        default=0,
        help='window size if sliding-window framework is used (0 for recurrent framework)')
    parser.add_argument(
        '--max-seq-len',
        type=int,
        default=None,
        help='maximum sequence length if recurrent framework is used')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    parser.add_argument(
        '--rename-back',
        action='store_true',
        help='Attempt to rename input files back to their original names after processing (use with caution).')

    args = parser.parse_args()
    return args


def main():
    """ Demo for video restoration models.
    Handles video file input or a directory of image frames.
    If input_path is a directory, images within will be renamed in-place.
    """
    args = parse_args()

    model = init_model(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))

    original_filenames_map = {} # To store original names if --rename-back is used
    effective_input_path = os.path.abspath(args.input_path)
    effective_start_idx = args.start_idx
    effective_filename_tmpl = args.filename_tmpl

    is_input_dir_processed = False

    if os.path.isdir(effective_input_path):
        print(f"Input path '{effective_input_path}' is a directory. Scanning for image frames...")
        
        all_items_in_dir = glob.glob(os.path.join(effective_input_path, '*'))
        image_files = sorted([
            os.path.abspath(f) # Use absolute paths for original files
            for f in all_items_in_dir
            if os.path.isfile(f) and f.lower().endswith(IMG_EXTENSIONS)
        ])

        if image_files:
            print(f"Found {len(image_files)} image frames. Renaming them in-place for inference.")
            
            # --- RENAME FILES IN-PLACE ---
            temp_renamed_files = [] # Keep track of files we renamed to attempt renaming back
            for idx, original_filepath in enumerate(image_files):
                original_dir = os.path.dirname(original_filepath)
                # New name based on the template, starting from index 0 for the model
                new_filename_for_model = effective_filename_tmpl.format(idx)
                new_filepath_for_model = os.path.join(original_dir, new_filename_for_model)

                if original_filepath == new_filepath_for_model:
                    print(f"Skipping rename for '{original_filepath}', already matches target.")
                    original_filenames_map[new_filepath_for_model] = original_filepath # Still track for potential rename-back consistency
                    temp_renamed_files.append(new_filepath_for_model) # Track it as if it was renamed
                    continue

                if os.path.exists(new_filepath_for_model):
                    # This is a conflict. We should not overwrite.
                    # For simplicity, we'll error out. More complex logic could backup the conflicting file.
                    print(f"ERROR: Target rename path '{new_filepath_for_model}' already exists. "
                          f"Cannot rename '{original_filepath}'. Please clear conflicting files or use a different directory.")
                    # Attempt to revert any renames done so far if --rename-back was intended
                    if args.rename_back and original_filenames_map:
                        print("Attempting to revert previous renames...")
                        for new_name, orig_name in reversed(list(original_filenames_map.items())):
                            if new_name != orig_name and os.path.exists(new_name):
                                try:
                                    shutil.move(new_name, orig_name)
                                    print(f"Reverted '{new_name}' to '{orig_name}'")
                                except Exception as e_revert:
                                    print(f"Failed to revert '{new_name}' to '{orig_name}': {e_revert}")
                    return # Exit
                
                try:
                    shutil.move(original_filepath, new_filepath_for_model)
                    print(f"Renamed '{original_filepath}' to '{new_filepath_for_model}'")
                    original_filenames_map[new_filepath_for_model] = original_filepath
                    temp_renamed_files.append(new_filepath_for_model)
                except Exception as e_rename:
                    print(f"ERROR: Failed to rename '{original_filepath}' to '{new_filepath_for_model}'. Error: {e_rename}")
                    # Attempt to revert any renames done so far
                    if args.rename_back and original_filenames_map:
                        print("Attempting to revert previous renames due to error...")
                        for new_name, orig_name in reversed(list(original_filenames_map.items())):
                             if new_name != orig_name and os.path.exists(new_name) and new_name not in temp_renamed_files: # only revert files not part of current failed batch
                                try:
                                    shutil.move(new_name, orig_name)
                                    print(f"Reverted '{new_name}' to '{orig_name}'")
                                except Exception as e_revert:
                                    print(f"Failed to revert '{new_name}' to '{orig_name}': {e_revert}")
                    return # Exit

            # The model will read from the same directory, but with files named 00000000.png, etc.
            # effective_input_path remains the same.
            effective_start_idx = 0  # Model expects 0-indexed files based on tmpl
            # effective_filename_tmpl is already args.filename_tmpl
            is_input_dir_processed = True
            print(f"Frames renamed. Model will read from '{effective_input_path}' with start_idx=0 and tmpl='{effective_filename_tmpl}'.")
        
        elif not effective_input_path.lower().endswith(VIDEO_EXTENSIONS):
             raise ValueError(
                f"Input path '{args.input_path}' is a directory but contains no supported image frames "
                f"(supported: {IMG_EXTENSIONS}) and does not appear to be a video file path.")

    elif not (os.path.isfile(effective_input_path) and effective_input_path.lower().endswith(VIDEO_EXTENSIONS)):
        raise ValueError(
            f"Input path '{args.input_path}' is not a recognized video file (supported: {VIDEO_EXTENSIONS}) "
            f"and not a directory containing processable image frames. Please check the path.")
    else:
        print(f"Input path '{effective_input_path}' is a video file. Processing directly.")

    try:
        print(f"Starting inference with: input='{effective_input_path}', window_size={args.window_size}, "
              f"start_idx={effective_start_idx}, filename_tmpl='{effective_filename_tmpl}', max_seq_len={args.max_seq_len}")
        
        output = restoration_video_inference(model, effective_input_path,
                                             args.window_size, effective_start_idx,
                                             effective_filename_tmpl, args.max_seq_len)
        print("Inference complete.")

        output_target_abs_path = os.path.abspath(args.output_path)
        file_extension = os.path.splitext(output_target_abs_path)[1].lower()

        if file_extension in VIDEO_EXTENSIONS:
            output_video_parent_dir = os.path.dirname(output_target_abs_path)
            if output_video_parent_dir:
                 os.makedirs(output_video_parent_dir, exist_ok=True)

            if output.ndim == 5 and output.size(0) == 1:
                output_frames_tensor = output.squeeze(0)
            elif output.ndim == 4:
                output_frames_tensor = output
            else:
                raise ValueError(f"Unexpected output tensor dimensions from model: {output.shape}")

            num_out_frames, _, h, w = output_frames_tensor.shape
            video_fps = 25
            if not is_input_dir_processed and os.path.isfile(args.input_path) and args.input_path.lower().endswith(VIDEO_EXTENSIONS):
                try:
                    cap = cv2.VideoCapture(args.input_path)
                    fps_in = cap.get(cv2.CAP_PROP_FPS)
                    if fps_in and fps_in > 0: video_fps = fps_in
                    cap.release()
                    print(f"Using FPS from input video: {video_fps}")
                except Exception as e_fps:
                    print(f"Warning: Could not read FPS from input video '{args.input_path}'. Defaulting to {video_fps} FPS. Error: {e_fps}")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_target_abs_path, fourcc, video_fps, (w, h))
            
            for i in range(num_out_frames):
                img = tensor2img(output_frames_tensor[i, :, :, :])
                video_writer.write(img.astype(np.uint8))
            video_writer.release()
            print(f"Output video saved to '{output_target_abs_path}'")

        else:
            mmcv.mkdir_or_exist(output_target_abs_path)
            if output.ndim == 5 and output.size(0) == 1:
                output_frames_tensor = output.squeeze(0)
            elif output.ndim == 4:
                output_frames_tensor = output
            else:
                raise ValueError(f"Unexpected output tensor dimensions from model: {output.shape}")
            num_out_frames = output_frames_tensor.size(0)
            print(f"Saving {num_out_frames} output frames to directory: '{output_target_abs_path}'")
            for i in range(num_out_frames):
                output_i_tensor = output_frames_tensor[i, :, :, :]
                img_np = tensor2img(output_i_tensor)
                save_path_i = os.path.join(output_target_abs_path, args.filename_tmpl.format(args.start_idx + i))
                mmcv.imwrite(img_np, save_path_i)
            print(f"Output frames saved successfully.")

    finally:
        # --- RENAME FILES BACK (if applicable and requested) ---
        if is_input_dir_processed and args.rename_back and original_filenames_map:
            print("\nAttempting to rename input files back to their original names...")
            # Iterate in reverse order of renaming to avoid conflicts if original names were sequential
            # We need to ensure we're renaming the files that were actually temporarily renamed
            renamed_during_session = list(original_filenames_map.keys())

            for new_name_path in reversed(renamed_during_session):
                original_name_path = original_filenames_map.get(new_name_path)
                if original_name_path and new_name_path != original_name_path: # Only if a rename actually happened
                    if os.path.exists(new_name_path):
                        try:
                            # Before renaming back, check if the original path is now occupied by an output file (unlikely but possible if input_dir=output_dir)
                            if os.path.exists(original_name_path) and original_name_path not in renamed_during_session:
                                print(f"Warning: Cannot rename '{new_name_path}' back to '{original_name_path}' because the original path is now occupied by a different file. Skipping this revert.")
                                continue

                            shutil.move(new_name_path, original_name_path)
                            print(f"Renamed '{new_name_path}' back to '{original_name_path}'")
                        except Exception as e_revert:
                            print(f"ERROR: Failed to rename '{new_name_path}' back to '{original_name_path}'. Error: {e_revert}")
                    else:
                        print(f"Warning: Expected renamed file '{new_name_path}' not found. Cannot rename back.")
            print("File renaming revert process finished.")
        elif is_input_dir_processed and not args.rename_back:
            print(f"\nInput files in '{effective_input_path}' were renamed for processing and were NOT renamed back as --rename-back was not specified.")
            print("They are currently named like '00000000.png', '00000001.png', etc.")


if __name__ == '__main__':
    main()