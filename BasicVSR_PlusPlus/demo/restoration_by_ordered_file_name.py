# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import glob # Added
import tempfile # Added
import shutil # Added

import cv2
import mmcv
import numpy as np
import torch

from mmedit.apis import init_model, restoration_video_inference
from mmedit.core import tensor2img # Ensure this import path is correct for your mmedit version
from mmedit.utils import modify_args

VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv') # Made more comprehensive
IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp') # Added for frame processing

def parse_args():
    modify_args()
    parser = argparse.ArgumentParser(description='Restoration demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('input_path', help='path to the input video or directory of image frames') # Renamed for clarity
    parser.add_argument('output_path', help='path to the output video or directory for image frames') # Renamed for clarity
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
    args = parser.parse_args()
    return args


def main():
    """ Demo for video restoration models.
    Handles video file input or a directory of image frames.
    """
    args = parse_args()

    model = init_model(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))

    temp_dir_manager = None
    # These variables will hold the actual parameters passed to the inference function
    effective_input_path = os.path.abspath(args.input_path) # Use absolute paths
    effective_start_idx = args.start_idx
    effective_filename_tmpl = args.filename_tmpl

    if os.path.isdir(effective_input_path):
        print(f"Input path '{effective_input_path}' is a directory. Scanning for image frames...")
        
        all_items_in_dir = glob.glob(os.path.join(effective_input_path, '*'))
        image_files = sorted([
            f for f in all_items_in_dir
            if os.path.isfile(f) and f.lower().endswith(IMG_EXTENSIONS)
        ])

        if image_files:
            print(f"Found {len(image_files)} image frames. Preparing them for inference.")
            
            temp_dir_manager = tempfile.TemporaryDirectory(prefix="basicvsr_input_frames_")
            temp_dir_abs_path = temp_dir_manager.name
            
            print(f"Copying and renaming frames to temporary directory: '{temp_dir_abs_path}'")
            for idx, original_filepath in enumerate(image_files):
                # Files in the temporary directory will be named according to effective_filename_tmpl, starting from 0
                temp_file_target_name = effective_filename_tmpl.format(idx)
                temp_file_target_path = os.path.join(temp_dir_abs_path, temp_file_target_name)
                
                try:
                    shutil.copy2(os.path.abspath(original_filepath), temp_file_target_path)
                except Exception as e_copy:
                    if temp_dir_manager: temp_dir_manager.cleanup()
                    raise RuntimeError(f"Failed to copy '{original_filepath}' to '{temp_file_target_path}'. Error: {e_copy}") from e_copy
            
            effective_input_path = temp_dir_abs_path # Model will read from temp dir
            effective_start_idx = 0  # Frames in temp dir are 0-indexed for the model
            # effective_filename_tmpl is already args.filename_tmpl, which is correct for reading these newly named files.
            print(f"Frames prepared. Model will read from '{effective_input_path}' with start_idx=0 and tmpl='{effective_filename_tmpl}'.")
        
        elif not effective_input_path.lower().endswith(VIDEO_EXTENSIONS): # Directory, but no images, and not a video extension
            # This case is tricky: if input_path was 'myvideo.mp4' but it's a dir, it's an error.
            # If it was 'myframes_dir' and empty, it's an error.
             if temp_dir_manager: temp_dir_manager.cleanup() # Should be None here anyway
             raise ValueError(
                f"Input path '{args.input_path}' is a directory but contains no supported image frames "
                f"(supported: {IMG_EXTENSIONS}) and does not appear to be a video file path.")
        # If it's a directory *and* ends with a video extension (e.g. "myvideo.mp4/"),
        # it's likely an error, but the next check will catch it if it's not a file.

    elif not (os.path.isfile(effective_input_path) and effective_input_path.lower().endswith(VIDEO_EXTENSIONS)):
        # Not a directory (from above) AND not a recognized video file.
        if temp_dir_manager: temp_dir_manager.cleanup()
        raise ValueError(
            f"Input path '{args.input_path}' is not a recognized video file (supported: {VIDEO_EXTENSIONS}) "
            f"and not a directory containing processable image frames. Please check the path.")
    else:
        print(f"Input path '{effective_input_path}' is a video file. Processing directly.")


    # Call the inference function
    print(f"Starting inference with: input='{effective_input_path}', window_size={args.window_size}, "
          f"start_idx={effective_start_idx}, filename_tmpl='{effective_filename_tmpl}', max_seq_len={args.max_seq_len}")
    
    output = restoration_video_inference(model, effective_input_path,
                                         args.window_size, effective_start_idx,
                                         effective_filename_tmpl, args.max_seq_len)
    print("Inference complete.")

    # Process and save output
    output_target_abs_path = os.path.abspath(args.output_path)
    file_extension = os.path.splitext(output_target_abs_path)[1].lower()

    if file_extension in VIDEO_EXTENSIONS:  # save as video
        output_video_parent_dir = os.path.dirname(output_target_abs_path)
        if output_video_parent_dir: # Ensure parent directory exists for the video file
             os.makedirs(output_video_parent_dir, exist_ok=True)

        # Ensure output is (T, C, H, W) or (N, T, C, H, W) where N=1
        if output.ndim == 5 and output.size(0) == 1:
            output_frames_tensor = output.squeeze(0) # (T, C, H, W)
        elif output.ndim == 4: # Already (T, C, H, W)
            output_frames_tensor = output
        else:
            if temp_dir_manager: temp_dir_manager.cleanup()
            raise ValueError(f"Unexpected output tensor dimensions from model: {output.shape}")

        num_out_frames, _, h, w = output_frames_tensor.shape
        
        # Try to get FPS from input if it was a video
        video_fps = 25 # Default FPS
        if temp_dir_manager is None and os.path.isfile(args.input_path) and args.input_path.lower().endswith(VIDEO_EXTENSIONS):
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
            img = tensor2img(output_frames_tensor[i, :, :, :]) # Pass a single frame tensor (C, H, W)
            video_writer.write(img.astype(np.uint8))
        # cv2.destroyAllWindows() # Generally not needed unless cv2.imshow was used
        video_writer.release()
        print(f"Output video saved to '{output_target_abs_path}'")

    else: # Save as frames
        mmcv.mkdir_or_exist(output_target_abs_path) # Ensure output directory for frames exists
        
        # Ensure output is (T, C, H, W) or (N, T, C, H, W) where N=1
        if output.ndim == 5 and output.size(0) == 1:
            output_frames_tensor = output.squeeze(0)
        elif output.ndim == 4:
            output_frames_tensor = output
        else:
            if temp_dir_manager: temp_dir_manager.cleanup()
            raise ValueError(f"Unexpected output tensor dimensions from model: {output.shape}")
            
        num_out_frames = output_frames_tensor.size(0)

        print(f"Saving {num_out_frames} output frames to directory: '{output_target_abs_path}'")
        # Use the *original* args.start_idx for numbering the output files
        for i in range(num_out_frames):
            output_i_tensor = output_frames_tensor[i, :, :, :] # Get the i-th frame from model output
            img_np = tensor2img(output_i_tensor)
            
            # Name output files based on the original user-specified start_idx and filename_tmpl
            save_path_i = os.path.join(output_target_abs_path, args.filename_tmpl.format(args.start_idx + i))
            mmcv.imwrite(img_np, save_path_i)
        print(f"Output frames saved successfully.")

    # Cleanup temporary directory if one was created
    if temp_dir_manager:
        try:
            temp_dir_manager.cleanup()
            print(f"Temporary directory '{temp_dir_manager.name}' cleaned up successfully.")
        except Exception as e:
            print(f"Warning: Failed to clean up temporary directory '{temp_dir_manager.name}'. Error: {e}")

if __name__ == '__main__':
    main()