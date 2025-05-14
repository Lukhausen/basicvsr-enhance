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
    p.add_argument('--input_folder', required=True,
                   help='folder with input frames')
    p.add_argument('--output_path', required=True,
                   help='where to save the enhanced frame')
    p.add_argument('--config', default='configs/basicvsr_plusplus_reds4.py',
                   help='model config file')
    p.add_argument('--checkpoint', default='checkpoints/basicvsr_plusplus_reds4.pth',
                   help='path to pretrained weights')
    p.add_argument('--device', default='cuda',
                   help='cuda or cpu')
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # — load model config & weights —
    cfg = Config.fromfile(args.config)
    model_kwargs = cfg.model.copy()
    model_kwargs.pop('type', None)   # remove the config’s 'type' key
    model = BasicVSRPlusPlus(**model_kwargs)
    load_checkpoint(model, args.checkpoint, map_location=device)
    model.to(device).eval()

    # — read & convert frames to RGB (no negative strides) —
    fnames = sorted(f for f in os.listdir(args.input_folder)
                    if f.lower().endswith(('.png','jpg','jpeg')))
    frames = []
    for fn in fnames:
        bgr = cv2.imread(os.path.join(args.input_folder, fn), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read {fn}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)

    # — to list of 4-D tensors (1,C,H,W) —
    tensors = [
        img2tensor(im, bgr2rgb=False, float32=True)
        .unsqueeze(0)
        for im in frames
    ]

    # — stack into one 5-D tensor (1, T, C, H, W) —
    lqs = torch.stack(tensors, dim=1).to(device)

    # — inference & extract central frame —
    with torch.no_grad():
        outs = model(lqs)  # returns tensor of shape (1, T, C, H, W)
    n, t, c, h, w = outs.size()
    mid = t // 2
    out_tensor = outs[0, mid]  # shape (C, H, W)

    # — convert to uint8 image —
    out_img = tensor2img(out_tensor.cpu(), rgb2bgr=False, out_type=np.uint8)

    # — back to BGR & save —
    bgr_out = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    cv2.imwrite(args.output_path, bgr_out)
    print(f"✔️  Saved enhanced frame: {args.output_path}")

if __name__ == '__main__':
    main()
