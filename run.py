#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess

def parse_args():
    p = argparse.ArgumentParser(
        description='BasicVSR++ 4K enhancement (official code)')
    p.add_argument('--input_folder',  type=str, required=True,
                   help='Folder with N 4K frames (PNG/JPG)')
    p.add_argument('--output_path',   type=str, required=True,
                   help='Path for the single enhanced central frame')
    p.add_argument('--checkpoint',    type=str,
                   default='basicvsr_plusplus/checkpoints/'
                           'basicvsr_plusplus_c64n7_8x1_600k_reds4.pth',
                   help='Path to the downloaded REDS4 checkpoint')
    p.add_argument('--num_feat',      type=int, default=64,
                   help='Number of feature channels (should match checkpoint)')
    p.add_argument('--device',        type=str, default='cuda',
                   help='“cuda” or “cpu”')
    return p.parse_args()

def main():
    args = parse_args()
    # Ensure the official inference script is there
    script = os.path.join('basicvsr_plusplus', 'inference.py')
    if not os.path.exists(script):
        sys.exit(f'ERROR: cannot find {script}; did you run install.sh?')

    # Temporary folder for all the frame outputs
    tmp_out = 'tmp_enhanced_frames'
    os.makedirs(tmp_out, exist_ok=True)

    # Call the official inference.py
    cmd = [
        sys.executable, script,
        '--input_folder',  args.input_folder,
        '--output_folder', tmp_out,
        '--checkpoint',    args.checkpoint,
        '--num_feat',      str(args.num_feat),
        '--device',        args.device
    ]
    print('Running official BasicVSR++ inference…')
    subprocess.check_call(cmd)

    # Pick out the central frame
    allf = sorted(f for f in os.listdir(tmp_out)
                  if f.lower().endswith(('.png','.jpg','jpeg')))
    if not allf:
        sys.exit('ERROR: no frames were written by inference.py')
    mid = len(allf) // 2
    src = os.path.join(tmp_out, allf[mid])
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    os.replace(src, args.output_path)
    print(f'✔️  Central enhanced frame saved to {args.output_path}')

if __name__ == '__main__':
    main()
