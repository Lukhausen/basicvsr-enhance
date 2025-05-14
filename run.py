#!/usr/bin/env python3
import os
import sys
import argparse

import cv2
import numpy as np
import torch

from mmcv import Config
from mmengine.runner import load_checkpoint
from basicsr.archs.basicvsrpp_arch import BasicVSRPlusPlus

def parse_args():
    parser = argparse.ArgumentParser(
        description='BasicVSR++ 4K enhancement (central frame only)')
    parser.add_argument('--input_folder',  type=str, required=True,
                        help='Folder containing N 4K frames (PNG/JPG)')
    parser.add_argument('--output_path',   type=str, required=True,
                        help='Where to save the enhanced central frame')
    parser.add_argument('--config',        type=str,
                        default='configs/basicvsr_plusplus_reds4.py',
                        help='Path to model config file')
    parser.add_argument('--checkpoint',    type=str,
                        default='checkpoints/basicvsr_plusplus_reds4.pth',
                        help='Path to pretrained weights')
    parser.add_argument('--device',        type=str, default='cuda',
                        help='“cuda” or “cpu”')
    return parser.parse_args()

def main():
    args = parse_args()
    # Validate config file
    if not os.path.isfile(args.config):
        sys.exit(f"❌ Config file not found: {args.config}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load model
    cfg = Config.fromfile(args.config)
    model = BasicVSRPlusPlus(**cfg.model)
    load_checkpoint(model, args.checkpoint, map_location=device)
    model.to(device).eval()

    # Read & sort frames
    files = sorted(
        f for f in os.listdir(args.input_folder)
        if f.lower().endswith(('.png','jpg','jpeg')))
    if not files:
        sys.exit(f"❌ No images found in {args.input_folder}")

    # Load BGR images → convert to RGB
    imgs = []
    for f in files:
        im = cv2.imread(os.path.join(args.input_folder, f), cv2.IMREAD_COLOR)
        if im is None:
            sys.exit(f"❌ Failed to read image: {f}")
        imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

    # Stack → [T, H, W, 3]
    arr = np.stack(imgs, axis=0)

    # → tensor [1, T, 3, H, W], float, normalized to [0,1]
    t = torch.from_numpy(arr).permute(0,3,1,2).unsqueeze(0).float().to(device) / 255.0

    # Inference
    with torch.no_grad():
        out = model(t)         # [1, T, C, H, W]
    mid = out.size(1) // 2     # central frame index
    frame = out[0, mid]        # [C, H, W]

    # Denormalize & convert back to BGR
    img = frame.permute(1,2,0).cpu().numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Save
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    success = cv2.imwrite(args.output_path, bgr)
    if not success:
        sys.exit(f"❌ Failed to write output to {args.output_path}")
    print(f"✔️  Saved enhanced frame to {args.output_path}")

if __name__ == '__main__':
    main()
