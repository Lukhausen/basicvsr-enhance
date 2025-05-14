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

    # load config & prepare model args
    cfg = Config.fromfile(args.config)
    mcfg = cfg.model.copy()
    mcfg.pop('type', None)  # strip the MMEditing "type" key

    # build & load
    model = BasicVSRPlusPlus(**mcfg)
    load_checkpoint(model, args.checkpoint, map_location=device)
    model.to(device).eval()

    # read input frames
    files = sorted(f for f in os.listdir(args.input_folder)
                   if f.lower().endswith(('.png','jpg','jpeg')))
    imgs = [cv2.imread(os.path.join(args.input_folder, f),
                       cv2.IMREAD_COLOR)[..., ::-1] for f in files]

    # to tensors
    seq = [img2tensor(im, bgr2rgb=False, float32=True)
           .unsqueeze(0).to(device) for im in imgs]

    # forward & grab central output
    with torch.no_grad():
        out_seq = model(seq)                 # (T, C, H, W)
        mid = out_seq.size(0) // 2
        out_img = tensor2img(out_seq[mid].cpu(),
                             rgb2bgr=False,
                             out_type=np.uint8)

    # save
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    cv2.imwrite(args.output_path, out_img[..., ::-1])
    print(f"✔️  saved → {args.output_path}")

if __name__ == '__main__':
    main()
