# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import shutil # Added for potential backup, though not fully implemented here

import cv2
import mmcv
import numpy as np
import torch

from mmedit.apis import init_model, restoration_video_inference
from mmedit.core import tensor2img
from mmedit.utils import modify_args

VIDEO_EXTENSIONS = ('.mp4', '.mov')
# Added common image extensions for the preprocessing step
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
    """
    Renames image files in the input_dir to match the expected format.
    Only acts if input_dir is a directory.
    """
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
    except NotADirectoryError: # Should be caught by the first check, but good to have
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
        # Ensure the new filename uses the extension from the template if provided, or keeps original if not.
        # For simplicity, we assume the template includes the extension (e.g. '{:08d}.png')
        new_filename_stem = filename_tmpl.split('.')[0].format(current_frame_idx)
        template_ext = os.path.splitext(filename_tmpl)[1]
        if not template_ext: # If template has no extension, try to keep original (might be problematic)
            original_ext = os.path.splitext(old_filename)[1]
            new_filename = new_filename_stem + original_ext
        else:
            new_filename = filename_tmpl.format(current_frame_idx)


        old_filepath = os.path.join(input_dir, old_filename)
        new_filepath = os.path.join(input_dir, new_filename)

        if old_filepath == new_filepath:
            # print(f"Skipping '{old_filename}': already matches target name '{new_filename}'.")
            skipped_count +=1
            continue

        try:
            os.rename(old_filepath, new_filepath)
            # print(f"Renamed '{old_filename}' to '{new_filename}'")
            renamed_count += 1
        except OSError as e:
            print(f"Error renaming '{old_filename}' to '{new_filename}': {e}")
            print("Please check file permissions and if the target filename already exists (though it shouldn't with proper sorting and unique new names).")
            # Potentially offer to stop or continue
    
    if renamed_count > 0:
        print(f"Successfully renamed {renamed_count} files in '{input_dir}'.")
    if skipped_count > 0 and renamed_count == 0 :
        print(f"All {skipped_count} files in '{input_dir}' already matched the target naming scheme. No files were renamed.")
    elif skipped_count > 0 :
         print(f"{skipped_count} files already matched the target naming scheme and were skipped.")
    if not image_files:
        print(f"No image files processed in {input_dir}.")


def main():
    """ Demo for video restoration models.

    Note that we accept video as input/output, when 'input_dir'/'output_dir'
    is set to the path to the video. But using videos introduces video
    compression, which lowers the visual quality. If you want actual quality,
    please save them as separate images (.png).
    """

    args = parse_args()

    # --- Pre-processing step to rename input files ---
    # IMPORTANT: Backup your original files before running if input_dir contains loose frames!
    print("--- Starting Pre-processing: Renaming Input Frames (if applicable) ---")
    preprocess_rename_input_frames(args.input_dir, args.start_idx, args.filename_tmpl)
    print("--- Finished Pre-processing ---")
    # --- End of Pre-processing step ---

    # Check if CUDA is available and set the device
    if args.device < 0 or not torch.cuda.is_available():
        device = torch.device('cpu')
        print("Warning: CUDA not available or device < 0. Using CPU. This might be very slow.")
    else:
        device = torch.device('cuda', args.device)
        print(f"Using CUDA device: {args.device}")


    model = init_model(
        args.config, args.checkpoint, device=device)

    print(f"Starting video restoration inference for input: '{args.input_dir}'")
    output = restoration_video_inference(model, args.input_dir,
                                         args.window_size, args.start_idx,
                                         args.filename_tmpl, args.max_seq_len)
    print("Video restoration inference complete.")

    # Ensure output directory exists
    if not os.path.splitext(args.output_dir)[1] in VIDEO_EXTENSIONS: # if output is a directory for frames
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Ensured output directory exists: {args.output_dir}")


    file_extension = os.path.splitext(args.output_dir)[1].lower()
    if file_extension in VIDEO_EXTENSIONS:  # save as video
        print(f"Saving output as video to: {args.output_dir}")
        if output is None or output.numel() == 0:
            print("Error: Output from model is empty. Cannot save video.")
            return

        # Output shape is expected to be (1, T, C, H, W) or (T, C, H, W)
        # We need to get H, W from the tensor
        if output.ndim == 5: # (1, T, C, H, W)
            num_frames = output.size(1)
            h, w = output.shape[-2:]
        elif output.ndim == 4: # (T, C, H, W) - let's adjust if model outputs this
             num_frames = output.size(0)
             h, w = output.shape[-2:]
             output = output.unsqueeze(0) # Add batch dimension for tensor2img compatibility
        else:
            print(f"Error: Unexpected output tensor shape: {output.shape}. Expected 4D or 5D.")
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # For .mp4
        if file_extension == '.mov':
            fourcc = cv2.VideoWriter_fourcc(*'avc1') # Common for .mov, or use 'jpeg'

        video_writer = cv2.VideoWriter(args.output_dir, fourcc, 25, (w, h)) # Assuming 25 FPS
        
        for i in range(0, num_frames):
            # tensor2img expects (1, C, H, W) or (C, H, W)
            # output is (1, T, C, H, W), so select frame: output[:, i, :, :, :]
            img_tensor_frame = output[:, i, :, :, :]
            img = tensor2img(img_tensor_frame, min_max=(-1, 1) if model.cfg.model.get('bgr_mean') else (0, 255)) # Heuristic for min_max
            video_writer.write(img.astype(np.uint8))
        
        cv2.destroyAllWindows()
        video_writer.release()
        print(f"Successfully saved video to {args.output_dir}")
    else: # save as frames
        print(f"Saving output as individual frames to directory: {args.output_dir}")
        if output is None or output.numel() == 0:
            print("Error: Output from model is empty. Cannot save frames.")
            return

        # Determine number of frames from output tensor
        if output.ndim == 5: # (1, T, C, H, W)
            num_frames_output = output.size(1)
        elif output.ndim == 4: # (T, C, H, W)
            num_frames_output = output.size(0)
            output = output.unsqueeze(0) # Add batch dimension for tensor2img compatibility
        else:
            print(f"Error: Unexpected output tensor shape: {output.shape}. Expected 4D or 5D.")
            return

        # The loop for saving output frames should use the actual number of output frames
        # and the original start_idx for consistent naming if desired.
        # The `restoration_video_inference` might return frames starting from 0 internally,
        # regardless of `args.start_idx` passed to it (which it uses for *reading*).
        # So, the output frames are indexed 0 to N-1.
        # We will save them using `args.start_idx` as the base for naming.
        for i in range(0, num_frames_output):
            # output_i is the i-th frame from the *model's output sequence*
            output_frame_tensor = output[:, i, :, :, :]
            img = tensor2img(output_frame_tensor, min_max=(-1, 1) if model.cfg.model.get('bgr_mean') else (0, 255)) # Heuristic for min_max
            
            # The filename for the saved frame should correspond to its sequence position
            # If args.start_idx was 10, the first output frame (index 0) becomes 10.png, second (index 1) becomes 11.png etc.
            save_frame_idx = args.start_idx + i 
            save_path_i = os.path.join(args.output_dir, args.filename_tmpl.format(save_frame_idx))
            
            mmcv.imwrite(img.astype(np.uint8), save_path_i)
        print(f"Successfully saved {num_frames_output} frames to {args.output_dir}")


if __name__ == '__main__':
    main()