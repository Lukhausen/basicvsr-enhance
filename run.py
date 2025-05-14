#!/usr/bin/env python3
# run.py — Minimal BasicVSR++ demo (fixes color, normalization, tensor shape)

import os
import argparse

import cv2
import numpy as np
import torch

from mmcv import Config
from mmengine.runner import load_checkpoint
from basicsr.archs.basicvsrpp_arch import BasicVSRPlusPlus

def parse_args():
    parser = argparse.ArgumentParser(
        description='BasicVSR++ 4K enhancement (single central frame)')
    parser.add_argument('--input_folder',  type=str, required=True,
                        help='Folder with N 4K frames (PNG/JPG)')
    parser.add_argument('--output_path',   type=str, required=True,
                        help='Path to save the enhanced frame (PNG/JPG)')
    parser.add_argument('--config',        type=str,
                        default='configs/restorers/basicvsr_plusplus_reds4.py',
                        help='MMEditing model config file')
    parser.add_argument('--checkpoint',    type=str,
                        default='checkpoints/basicvsr_plusplus_reds4.pth',
                        help='Pretrained weights file')
    parser.add_argument('--device',        type=str, default='cuda',
                        help='“cuda” or “cpu”')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 1) Build the model from config
    cfg = Config.fromfile(args.config)
    model = BasicVSRPlusPlus(**cfg.model)
    load_checkpoint(model, args.checkpoint, map_location=device)
    model.to(device).eval()

    # 2) Gather, sort & read frames
    files = sorted(
        f for f in os.listdir(args.input_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    imgs = [
        cv2.imread(os.path.join(args.input_folder, f), cv2.IMREAD_COLOR)
        for f in files
    ]
    if len(imgs) == 0:
        raise RuntimeError(f"No images found in {args.input_folder}")

    # 3) BGR -> RGB, stack, normalize
    rgb_imgs = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in imgs]
    arr = np.stack(rgb_imgs, axis=0)            # (T, H, W, 3)
    tensor = torch.from_numpy(arr)              # (T, H, W, 3)
    tensor = tensor.permute(0, 3, 1, 2).float()  # (T, 3, H, W)
    tensor = tensor.unsqueeze(0).to(device)     # (1, T, C, H, W)
    tensor = tensor / 255.0                     # normalize to [0,1]

    # 4) Inference
    with torch.no_grad():
        out = model(tensor)                     # (1, T, C, H, W)
    mid = out.size(1) // 2                       # central time index
    frame = out[0, mid]                          # (C, H, W)

    # 5) Denormalize & save
    frame = frame.permute(1, 2, 0).cpu().numpy() # (H, W, C)
    frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    cv2.imwrite(args.output_path, bgr)
    print(f"✔️  Saved enhanced frame to {args.output_path}")


if __name__ == '__main__':
    main()
