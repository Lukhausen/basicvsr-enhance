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
    parser = argparse.ArgumentParser(description='BasicVSR++ 4K enhancement')
    parser.add_argument('--input_folder', type=str, required=True,
                        help='Folder with 5–10 4K frames')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save the enhanced frame (PNG/JPG)')
    parser.add_argument('--config', type=str, default='configs/basicvsr_plusplus_reds4.py',
                        help='Model config file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/basicvsr_plusplus_reds4.pth',
                        help='Pretrained weights file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load config and model
    cfg = Config.fromfile(args.config)
    model = BasicVSRPlusPlus(**cfg.model)
    load_checkpoint(model, args.checkpoint, map_location=device)
    model.to(device).eval()

    # Read and sort frames
    files = sorted([os.path.join(args.input_folder, f)
                    for f in os.listdir(args.input_folder)
                    if f.lower().endswith(('.png','.jpg','.jpeg'))])
    imgs = [cv2.imread(f, cv2.IMREAD_COLOR)[..., ::-1] for f in files]

    # Prepare tensors
    inputs = [img2tensor(im, bgr2rgb=False, float32=True)
              .unsqueeze(0).to(device) for im in imgs]

    # Inference
    with torch.no_grad():
        out_seq = model(inputs)            # (num_frames, C, H, W)
        mid = out_seq.size(0) // 2         # central frame
        out_img = tensor2img(out_seq[mid].cpu(), rgb2bgr=False, out_type=np.uint8)

    # Save result
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    cv2.imwrite(args.output_path, out_img[..., ::-1])
    print(f"Saved enhanced frame to {args.output_path}")


if __name__ == '__main__':
    main()
