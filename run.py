#!/usr/bin/env python3
import os
import argparse
import torch
import cv2
import numpy as np
from mmcv import Config
from mmengine.runner import load_checkpoint
from basicsr.archs.basicvsrpp_arch import BasicVSRPlusPlus
from basicsr.utils.img_util import img2tensor, tensor2img

def parse_args():
    p = argparse.ArgumentParser(
        description='BasicVSR++ 4K enhancement (central frame)')
    p.add_argument('--input_folder',  required=True, help='input frames')
    p.add_argument('--output_path',   required=True, help='saved image')
    p.add_argument('--config',
                   default='configs/basicvsr_plusplus_reds4.py',
                   help='model config')
    p.add_argument('--checkpoint',
                   default='checkpoints/basicvsr_plusplus_reds4.pth',
                   help='pretrained weights')
    p.add_argument('--device',
                   default='cuda',
                   help='cuda or cpu')
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # --- build model ---
    cfg = Config.fromfile(args.config)
    mcfg = cfg.model.copy()
    mcfg.pop('type', None)  # drop the config's 'type' key

    model = BasicVSRPlusPlus(**mcfg)
    load_checkpoint(model, args.checkpoint, map_location=device)
    model.to(device).eval()

    # --- load & convert frames ---
    files = sorted(f for f in os.listdir(args.input_folder)
                   if f.lower().endswith(('.png','jpg','jpeg')))
    imgs = []
    for fname in files:
        bgr = cv2.imread(os.path.join(args.input_folder, fname),
                         cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"cannot read {fname}")
        # convert to RGB *without* negative‐stride slicing
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        imgs.append(rgb)

    # --- to tensors ---
    seq = [
        img2tensor(im, bgr2rgb=False, float32=True)
        .unsqueeze(0).to(device)
        for im in imgs
    ]

    # --- inference & pick central frame ---
    with torch.no_grad():
        out_seq = model(seq)                 # (T, C, H, W)
        mid = out_seq.size(0) // 2
        out_img = tensor2img(
            out_seq[mid].cpu(),
            rgb2bgr=False,
            out_type=np.uint8
        )

    # --- save result ---
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    # convert back to BGR for OpenCV
    bgr_out = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output_path, bgr_out)
    print(f"✔️  saved → {args.output_path}")

if __name__ == '__main__':
    main()
