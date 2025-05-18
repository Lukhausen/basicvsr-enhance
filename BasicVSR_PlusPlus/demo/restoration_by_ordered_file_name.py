# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import glob
import tempfile
import shutil
import subprocess # For ls debug

import cv2
import mmcv
import numpy as np
import torch

from mmedit.apis import init_model, restoration_video_inference
# Try to import tensor2img from the most likely locations for mmedit 0.14.0
try:
    from mmedit.core import tensor2img
except ImportError:
    try:
        from mmedit.utils import tensor2img # Older versions might have it here
    except ImportError:
        # As a last resort, define a simple one if not found.
        print("Warning: mmedit.core.tensor2img or mmedit.utils.tensor2img not found. Using a basic fallback.")
        def tensor2img(tensor, min_max=(0, 1)): # Basic implementation
            output = tensor.clone().detach().cpu().squeeze() # Ensure it's on CPU, squeezed
            if output.ndim == 3: # C, H, W
                output = output.permute(1, 2, 0) # H, W, C
            output = output.numpy()
            output = (output - np.min(output)) / (np.max(output) - np.min(output) + 1e-5) # Normalize
            output = np.clip(output, 0, 1)
            output = (output * 255.0).round().astype(np.uint8)
            return output

from mmedit.utils import modify_args

VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv') # Added more common video extensions
IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')


def parse_args():
    # modify_args() from mmedit.utils mainly handles downloading remote configs/checkpoints
    # and making their paths local. It should not affect input_dir.
    modify_args()
    parser = argparse.ArgumentParser(description='Restoration demo by ordered file name or video')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('input_path', help='path to the input video or directory of image frames')
    parser.add_argument('output_path', help='path to the output video or directory for image frames')
    parser.add_argument(
        '--start-idx',
        type=int,
        default=0,
        help='index for the first frame of the output sequence (if saving frames to a dir)')
    parser.add_argument(
        '--filename-tmpl',
        default='{:08d}.png',
        help='template for the file names of output frames (if saving frames to a dir)')
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
    parser.add_argument('--device', type=int, default=0, help='CUDA device id (e.g., 0, 1)')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Ensure the output directory exists if saving frames
    # If output_path is a video file, its parent directory will be checked later when opening VideoWriter
    if not args.output_path.lower().endswith(VIDEO_EXTENSIONS):
        mmcv.mkdir_or_exist(args.output_path)

    model = init_model(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))

    # These will be passed to restoration_video_inference
    effective_input_path = os.path.abspath(args.input_path) # Ensure input_path is absolute from the start
    effective_start_idx = args.start_idx
    effective_filename_tmpl = args.filename_tmpl
    
    temp_dir_manager = None
    # Standardized template for files inside the temporary directory for model's consumption
    TEMP_DIR_INTERNAL_FILENAME_TMPL = '{:08d}.png' 

    if os.path.isdir(effective_input_path):
        print(f"Input path '{effective_input_path}' is a directory. Scanning for image frames...")
        
        # Glob for all files/dirs, then filter for image files
        all_items_in_dir = glob.glob(os.path.join(effective_input_path, '*'))
        image_files = sorted([
            f for f in all_items_in_dir
            if os.path.isfile(f) and f.lower().endswith(IMG_EXTENSIONS)
        ])

        if image_files:
            print(f"Found {len(image_files)} image frames. Processing them in alphabetical order.")
            
            temp_dir_manager = tempfile.TemporaryDirectory(prefix="basicvsr_frames_")
            temp_dir_abs_path = temp_dir_manager.name # This is guaranteed to be an absolute path
            
            print(f"Preparing temporary input files in '{temp_dir_abs_path}' using internal template '{TEMP_DIR_INTERNAL_FILENAME_TMPL}'.")

            for idx, original_filepath in enumerate(image_files):
                # Ensure original_filepath is absolute for symlinking/copying
                abs_original_filepath = os.path.abspath(original_filepath)
                temp_file_link_path = os.path.join(temp_dir_abs_path, TEMP_DIR_INTERNAL_FILENAME_TMPL.format(idx))
                
                try:
                    os.symlink(abs_original_filepath, temp_file_link_path)
                except OSError as e_symlink:
                    print(f"Warning: Symlink creation failed for '{abs_original_filepath}' -> '{temp_file_link_path}' (Error: {e_symlink}). Attempting to copy file...")
                    try:
                        shutil.copy2(abs_original_filepath, temp_file_link_path) # copy2 preserves metadata
                    except Exception as e_copy:
                        if temp_dir_manager: temp_dir_manager.cleanup()
                        raise RuntimeError(f"Failed to create symlink or copy for '{abs_original_filepath}' to '{temp_file_link_path}'. Error: {e_copy}") from e_copy
            
            effective_input_path = temp_dir_abs_path # Use the absolute path of the temporary directory
            effective_start_idx = 0  # Files in temp dir are 0-indexed for the model
            effective_filename_tmpl = TEMP_DIR_INTERNAL_FILENAME_TMPL # Template used for temp files
        else: # Directory exists but no image files found.
            if temp_dir_manager: temp_dir_manager.cleanup() # Should not happen here, but good practice
            raise ValueError(
                f"Input path '{effective_input_path}' is a directory but contains no supported image frames "
                f"(supported: {IMG_EXTENSIONS}). If it's a video, provide the direct file path.")
    
    elif os.path.isfile(effective_input_path) and effective_input_path.lower().endswith(VIDEO_EXTENSIONS):
        print(f"Input path '{effective_input_path}' is a video file.")
        # effective_input_path, effective_start_idx, effective_filename_tmpl are already set from args
        # and effective_input_path was made absolute.
    elif not os.path.exists(effective_input_path):
        raise FileNotFoundError(f"Input path '{effective_input_path}' does not exist.")
    else: # Exists, but not a dir with frames, not a video file
        raise ValueError(
            f"Input path '{effective_input_path}' is not a recognized video file (supported: {VIDEO_EXTENSIONS}) "
            f"and not a directory containing supported image frames (supported: {IMG_EXTENSIONS}). Please check the path.")

    print(f"EFFECTIVE PARAMS for inference -> input_dir: '{effective_input_path}', "
          f"start_idx: {effective_start_idx}, filename_tmpl: '{effective_filename_tmpl}'")

    # --- DEBUG PRINTS for temporary directory case ---
    if temp_dir_manager:
        print(f"DEBUG: Listing contents of temporary input directory: {effective_input_path}")
        try:
            ls_output = subprocess.check_output(['ls', '-lha', effective_input_path], text=True, stderr=subprocess.STDOUT)
            print(ls_output)
        except subprocess.CalledProcessError as e_ls:
            print(f"DEBUG: 'ls -lha {effective_input_path}' failed. Output:\n{e_ls.output}")
        except Exception as e_ls_other:
            print(f"DEBUG: Failed to list directory with 'ls -lha' due to: {e_ls_other}")

        first_temp_file_to_check = os.path.join(effective_input_path, TEMP_DIR_INTERNAL_FILENAME_TMPL.format(0))
        print(f"DEBUG: Checking existence of first temp file: {first_temp_file_to_check}")
        if os.path.exists(first_temp_file_to_check): # Checks symlink itself
            print(f"DEBUG: File/Symlink {first_temp_file_to_check} exists.")
            if os.path.islink(first_temp_file_to_check):
                link_target = os.readlink(first_temp_file_to_check)
                print(f"DEBUG: It is a symlink. Target: {link_target}")
                if os.path.exists(link_target): # Checks symlink target
                    print(f"DEBUG: Symlink target {link_target} exists.")
                else:
                    print(f"DEBUG: CRITICAL WARNING! Symlink target {link_target} does NOT exist.")
            else:
                print(f"DEBUG: It is a regular file (copied).")
        else:
            print(f"DEBUG: CRITICAL WARNING! File/Symlink {first_temp_file_to_check} does NOT exist in temp dir.")
    # --- END DEBUG PRINTS ---

    output_tensor = restoration_video_inference(model, effective_input_path,
                                                args.window_size, effective_start_idx,
                                                effective_filename_tmpl, args.max_seq_len)
    print("Inference complete.")

    # --- Process and Save Output ---
    output_target_path = os.path.abspath(args.output_path) # Ensure output path is absolute
    output_file_extension = os.path.splitext(output_target_path)[1].lower()

    if output_tensor.ndim == 5 and output_tensor.size(0) == 1: # (N, T, C, H, W) -> (T, C, H, W)
        output_frames_tensor = output_tensor.squeeze(0)
    elif output_tensor.ndim == 4: # Already (T, C, H, W)
        output_frames_tensor = output_tensor
    else:
        if temp_dir_manager: temp_dir_manager.cleanup()
        raise ValueError(f"Unexpected output tensor dimensions from model: {output_tensor.shape}")

    num_output_frames, _, h, w = output_frames_tensor.shape

    if output_file_extension in VIDEO_EXTENSIONS:
        print(f"Saving output as video to: {output_target_path}")
        
        output_video_parent_dir = os.path.dirname(output_target_path)
        if output_video_parent_dir: # Ensure parent directory exists for the video file
             os.makedirs(output_video_parent_dir, exist_ok=True)
        
        video_fps = 25 
        if temp_dir_manager is None and os.path.isfile(effective_input_path): # Original input was a video
            try:
                cap = cv2.VideoCapture(effective_input_path)
                fps_in = cap.get(cv2.CAP_PROP_FPS)
                if fps_in and fps_in > 0: video_fps = fps_in
                cap.release()
            except Exception as e_fps:
                print(f"Warning: Could not read FPS from input video '{effective_input_path}'. Defaulting to {video_fps} FPS. Error: {e_fps}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Common, but consider XVID or avc1 for wider compatibility
        video_writer = cv2.VideoWriter(output_target_path, fourcc, video_fps, (w, h))
        
        for i in range(num_output_frames):
            img_np = tensor2img(output_frames_tensor[i]) 
            video_writer.write(img_np)
         
        video_writer.release()
        print(f"Output video saved successfully to {output_target_path}")
    else: # Output is a directory for image frames
        print(f"Saving output as image frames to directory: {output_target_path}")
        # mmcv.mkdir_or_exist(output_target_path) # Already done at the beginning of main for frame output dir
            
        for i in range(num_output_frames):
            img_np = tensor2img(output_frames_tensor[i])
            # Use args.start_idx for naming output files as per user's original intent for the sequence
            output_frame_filename = args.filename_tmpl.format(args.start_idx + i)
            save_path_i = os.path.join(output_target_path, output_frame_filename)
            mmcv.imwrite(img_np, save_path_i)
        print(f"{num_output_frames} output frames saved in '{output_target_path}' "
              f"using template '{args.filename_tmpl}' starting from index {args.start_idx}.")

    # --- Cleanup ---
    if temp_dir_manager:
        try:
            temp_dir_manager.cleanup()
            print(f"Temporary directory '{temp_dir_manager.name}' cleaned up successfully.")
        except Exception as e_cleanup: # Catch more specific exceptions if needed
            print(f"Warning: Failed to clean up temporary directory '{temp_dir_manager.name}'. Error: {e_cleanup}")

if __name__ == '__main__':
    main()