# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import shutil # Added for file operations
import tempfile # Added for temporary directory

import cv2
import mmcv
import numpy as np
import torch

from mmedit.apis import init_model, restoration_video_inference
from mmedit.core import tensor2img
from mmedit.utils import modify_args

VIDEO_EXTENSIONS = ('.mp4', '.mov')


def parse_args():
    modify_args()
    parser = argparse.ArgumentParser(description='Restoration demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('input_dir', help='directory of the input video or image sequence')
    parser.add_argument('output_dir', help='directory of the output video or image sequence')
    parser.add_argument(
        '--start-idx',
        type=int,
        default=0,
        help='index corresponds to the first frame of the sequence')
    parser.add_argument(
        '--filename-tmpl',
        default='{:08d}.png',
        help='template of the file names for input and output image sequences')
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
    """ Demo for video restoration models.

    Note that we accept video as input/output, when 'input_dir'/'output_dir'
    is set to the path to the video. But using videos introduces video
    compression, which lowers the visual quality. If you want actual quality,
    please save them as separate images (.png).
    """

    args = parse_args()

    # --- Start of added preprocessing ---
    temp_input_dir_obj = None # For managing the lifecycle of temp dir
    processed_input_dir = args.input_dir # By default, use original input_dir

    # Check if input_dir is a directory and not a video file path
    is_input_dir_a_directory = os.path.isdir(args.input_dir)
    is_input_dir_a_video = any(args.input_dir.lower().endswith(ext) for ext in VIDEO_EXTENSIONS)

    if is_input_dir_a_directory and not is_input_dir_a_video:
        print(f"Input '{args.input_dir}' is a directory. Preprocessing files...")

        # List all files in the input directory, sort them alphabetically
        # This assumes natural sort order is desired for frame sequences
        try:
            filenames = sorted([
                f for f in os.listdir(args.input_dir)
                if os.path.isfile(os.path.join(args.input_dir, f))
            ])
        except FileNotFoundError:
            print(f"Error: Input directory '{args.input_dir}' not found.")
            return
        except Exception as e:
            print(f"Error listing files in '{args.input_dir}': {e}")
            return


        if not filenames:
            print(f"Warning: No files found in '{args.input_dir}'. Proceeding with empty input.")
        else:
            # Create a temporary directory
            temp_input_dir_obj = tempfile.TemporaryDirectory(prefix="mmedit_preprocessed_")
            processed_input_dir = temp_input_dir_obj.name
            print(f"Created temporary directory for preprocessed input: {processed_input_dir}")

            for i, original_filename in enumerate(filenames):
                source_path = os.path.join(args.input_dir, original_filename)

                # Determine the new filename based on start_idx and filename_tmpl
                new_frame_number = args.start_idx + i
                # Ensure the extension from filename_tmpl is used, or infer if not present
                _, original_ext = os.path.splitext(original_filename)
                if '.' not in args.filename_tmpl.split('{')[0]: # If template doesn't specify extension
                    new_filename_formatted = args.filename_tmpl.format(new_frame_number) + original_ext
                else:
                    new_filename_formatted = args.filename_tmpl.format(new_frame_number)

                destination_path = os.path.join(processed_input_dir, new_filename_formatted)

                try:
                    # print(f"Copying '{source_path}' to '{destination_path}'")
                    shutil.copy2(source_path, destination_path) # copy2 preserves metadata
                except Exception as e:
                    print(f"Error copying '{source_path}' to '{destination_path}': {e}")
                    if temp_input_dir_obj:
                        temp_input_dir_obj.cleanup() # Clean up temp dir on error
                    return
            print(f"Finished preprocessing {len(filenames)} files into {processed_input_dir}")
    elif is_input_dir_a_video:
        print(f"Input '{args.input_dir}' is a video file. Skipping file preprocessing.")
    else: # Not a directory, not a video -> likely an error or specific single image case not handled by this preprocessing
        print(f"Input '{args.input_dir}' is not a directory of frames or a recognized video file. Skipping preprocessing.")
    # --- End of added preprocessing ---

    try:
        model = init_model(
            args.config, args.checkpoint, device=torch.device('cuda', args.device))

        # Use the potentially modified input directory path
        output = restoration_video_inference(model, processed_input_dir,
                                             args.window_size, args.start_idx,
                                             args.filename_tmpl, args.max_seq_len)

        file_extension = os.path.splitext(args.output_dir)[1]
        if file_extension.lower() in VIDEO_EXTENSIONS:  # save as video
            h, w = output.shape[-2:]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or 'XVID', 'MJPG', etc.
            # Ensure output directory for video exists
            os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
            video_writer = cv2.VideoWriter(args.output_dir, fourcc, 25, (w, h)) # Assuming 25 FPS
            for i in range(0, output.size(1)):
                img = tensor2img(output[:, i, :, :, :])
                video_writer.write(img.astype(np.uint8))
            cv2.destroyAllWindows()
            video_writer.release()
            print(f"Output video saved to {args.output_dir}")
        else: # save as image sequence
            # Ensure output directory for frames exists
            os.makedirs(args.output_dir, exist_ok=True)
            for i in range(args.start_idx, args.start_idx + output.size(1)):
                output_i = output[:, i - args.start_idx, :, :, :]
                output_i = tensor2img(output_i)
                save_path_i = os.path.join(args.output_dir, args.filename_tmpl.format(i))
                mmcv.imwrite(output_i, save_path_i)
            print(f"Output frames saved to directory {args.output_dir} with template {args.filename_tmpl}")

    finally:
        # Clean up the temporary directory if it was created
        if temp_input_dir_obj:
            print(f"Cleaning up temporary directory: {temp_input_dir_obj.name}")
            temp_input_dir_obj.cleanup()


if __name__ == '__main__':
    main()