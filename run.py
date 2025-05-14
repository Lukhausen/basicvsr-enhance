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
    p = argparse.ArgumentParser()
    p.add_argument('--input_folder',  required=True)
    p.add_argument('--output_path',   required=True)
    p.add_argument('--config',  default='configs/basicvsr_plusplus_reds4.py')
    p.add_argument('--checkpoint',
                   default='checkpoints/basicvsr_plusplus_reds4.pth')
    p.add_argument('--device', default='cuda')
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # load config & model
    cfg = Config.fromfile(args.config)
    model = BasicVSRPlusPlus(**cfg.model)
    load_checkpoint(model, args.checkpoint, map_location=device)
    model.to(device).eval()

    # read frames
    files = sorted(f for f in os.listdir(args.input_folder)
                   if f.lower().endswith(('.png','.jpg','.jpeg')))
    imgs  = [cv2.imread(os.path.join(args.input_folder, f))[..., ::-1]
             for f in files]

    # to tensors
    seq = [img2tensor(im, bgr2rgb=False, float32=True)
           .unsqueeze(0).to(device) for im in imgs]

    # inference
    with torch.no_grad():
        out_seq = model(seq)                    # (T, C, H, W)
        mid_idx = out_seq.size(0)//2
        out_img = tensor2img(out_seq[mid_idx].cpu(),
                             rgb2bgr=False,
                             out_type=np.uint8)

    # save
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    cv2.imwrite(args.output_path, out_img[..., ::-1])
    print(f"✔️  saved → {args.output_path}")

if __name__ == '__main__':
    main() 