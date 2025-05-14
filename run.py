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
        description='BasicVSR++ 4K enhancement (central frame by fusing all neighbours)')
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

    # Validate files
    if not os.path.isfile(args.config):
        sys.exit(f"❌ Config not found: {args.config}")
    if not os.path.isfile(args.checkpoint):
        sys.exit(f"❌ Checkpoint not found: {args.checkpoint}")

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load config & model
    cfg = Config.fromfile(args.config)
    model_cfg = dict(cfg.model)
    model_cfg.pop('type', None)
    model = BasicVSRPlusPlus(**model_cfg)
    load_checkpoint(model, args.checkpoint, map_location=device)
    model.to(device).eval()

    # Gather frames
    files = sorted(
        f for f in os.listdir(args.input_folder)
        if f.lower().endswith(('.png','jpg','jpeg')))
    if not files:
        sys.exit(f"❌ No images found in {args.input_folder}")

    # Read & convert to RGB
    imgs = []
    for fn in files:
        bgr = cv2.imread(os.path.join(args.input_folder, fn),
                         cv2.IMREAD_COLOR)
        if bgr is None:
            sys.exit(f"❌ Failed to read {fn}")
        imgs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    # Stack into numpy array [T, H, W, 3]
    arr = np.stack(imgs, axis=0)

    # **HERE’S THE FIX** → to tensor [1, C, T, H, W]
    tensor = (
        torch.from_numpy(arr)
             .permute(3, 0, 1, 2)    # from [T,H,W,3] to [3,T,H,W]
             .unsqueeze(0)           # to [1,3,T,H,W]
             .float()
             .to(device)
        / 255.0
    )

    # Inference
    with torch.no_grad():
        out = model(tensor)         # [1, T, C, H, W]
    mid = out.size(1) // 2          # central frame index
    frame = out[0, mid]             # [C, H, W]

    # Denormalize & back to BGR
    img = frame.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Save
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    if not cv2.imwrite(args.output_path, out_bgr):
        sys.exit(f"❌ Failed to write {args.output_path}")
    print(f"✔️  Saved enhanced frame to {args.output_path}")

if __name__ == '__main__':
    main()
