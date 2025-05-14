#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess

def parse_args():
    p = argparse.ArgumentParser(
        description='BasicVSR++ 4K enhancement (central-frame fusion)')
    p.add_argument('--input_folder', type=str, required=True,
                   help='Folder with N 4K frames (PNG/JPG)')
    p.add_argument('--output_path', type=str, required=True,
                   help='Where to save the enhanced central frame')
    p.add_argument('--checkpoint', type=str,
                   default='basicvsr_plusplus/checkpoints/'
                           'basicvsr_plusplus_c64n7_8x1_600k_reds4.pth',
                   help='Path to the REDS4 checkpoint')
    p.add_argument('--device', type=str, default='cuda',
                   help='Device: "cuda" or "cpu"')
    return p.parse_args()

def main():
    args = parse_args()
    demo_script = os.path.join('basicvsr_plusplus', 'demo', 'restoration_video_demo.py')
    if not os.path.isfile(demo_script):
        sys.exit(f'ERROR: demo script not found: {demo_script}')

    # Temporary out folder
    tmp_out = 'tmp_enhanced'
    os.makedirs(tmp_out, exist_ok=True)

    cmd = [
        sys.executable, demo_script,
        '--input_folder',    args.input_folder,
        '--output_folder',   tmp_out,
        '--checkpoint_file', args.checkpoint,
        '--device',          args.device
    ]

    print('📡  Running BasicVSR++ official demo…')
    subprocess.check_call(cmd)

    # Move the central frame to the desired output
    frames = sorted(f for f in os.listdir(tmp_out)
                    if f.lower().endswith(('.png','jpg','jpeg')))
    if not frames:
        sys.exit('ERROR: demo produced no frames.')

    central = frames[len(frames)//2]
    src = os.path.join(tmp_out, central)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    os.replace(src, args.output_path)

    print(f'✔️  Saved enhanced central frame to {args.output_path}')

if __name__ == '__main__':
    main()
