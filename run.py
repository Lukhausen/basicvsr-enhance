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
    p = argparse.ArgumentParser(
        description='BasicVSR++ 4K enhancement (fuse N→1 central frame)')
    p.add_argument('--input_folder', type=str, required=True,
                   help='Folder with N 4K frames (PNG/JPG)')
    p.add_argument('--output_path', type=str, required=True,
                   help='Where to save the enhanced central frame')
    p.add_argument('--config', type=str,
                   default='configs/basicvsr_plusplus_reds4.py',
                   help='Model config file')
    p.add_argument('--checkpoint', type=str,
                   default='checkpoints/basicvsr_plusplus_reds4.pth',
                   help='Pretrained weights')
    p.add_argument('--device', type=str, default='cuda',
                   help='cuda or cpu')
    return p.parse_args()

def main():
    args = parse_args()

    # sanity checks
    if not os.path.isfile(args.config):
        sys.exit(f"❌ Config not found: {args.config}")
    if not os.path.isfile(args.checkpoint):
        sys.exit(f"❌ Checkpoint not found: {args.checkpoint}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # load model
    cfg = Config.fromfile(args.config)
    model_cfg = dict(cfg.model)
    model_cfg.pop('type', None)            # remove stray 'type' field
    model = BasicVSRPlusPlus(**model_cfg)
    load_checkpoint(model, args.checkpoint, map_location=device)
    model.to(device).eval()

    # load & sort frames
    ims = sorted([
        f for f in os.listdir(args.input_folder)
        if f.lower().endswith(('.png','jpg','jpeg'))
    ])
    if not ims:
        sys.exit(f"❌ No images in {args.input_folder}")

    imgs = []
    for fn in ims:
        bgr = cv2.imread(os.path.join(args.input_folder, fn), cv2.IMREAD_COLOR)
        if bgr is None:
            sys.exit(f"❌ Failed to read {fn}")
        imgs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    # stack → numpy [T,H,W,3]
    arr = np.stack(imgs, axis=0)

    # **FIXED**: to tensor [1, T, 3, H, W]
    tensor = (
        torch.from_numpy(arr)
             .permute(0, 3, 1, 2)  # [T,3,H,W]
             .unsqueeze(0)         # [1,T,3,H,W]
             .float()
             .to(device)
        / 255.0
    )

    # inference
    with torch.no_grad():
        out = model(tensor)            # [1, T, C, H, W]
    mid  = out.size(1) // 2             # central index
    frm  = out[0, mid]                  # [C, H, W]

    # back to image
    img = frm.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    if not cv2.imwrite(args.output_path, bgr):
        sys.exit(f"❌ Could not write {args.output_path}")
    print(f"✔️  Saved enhanced frame to {args.output_path}")

if __name__ == '__main__':
    main()
