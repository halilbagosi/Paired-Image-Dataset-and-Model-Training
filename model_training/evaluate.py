import os
import argparse
import json
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from dataset import get_test_dataloader
from metrics import MetricsCalculator
from models.unet import UNet
from models.pix2pix import Pix2Pix
from models.pynet import PyNET
from models.restormer import Restormer


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',    type=str, default='dataset')
    parser.add_argument('--model',      type=str, choices=['unet', 'pix2pix', 'pynet', 'restormer'], required=True)
    parser.add_argument('--weights',    type=str, required=True, help='Path to .pth checkpoint')
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device()

    # Evaluate one image at a time for accurate per-image metrics
    test_loader = get_test_dataloader(args.dataset, batch_size=1)

    model_cls = {'unet': UNet, 'pix2pix': Pix2Pix, 'pynet': PyNET, 'restormer': Restormer}
    model = model_cls[args.model]().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    calculator = MetricsCalculator(device)

    total_psnr = total_ssim = total_lpips = 0.0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Inference direction: iPhone (input) → Fuji (target)
            iphone = batch['iphone'].to(device)   # input:  iPhone photo
            fuji   = batch['fuji'].to(device)     # target: Fuji reference
            name   = batch['name'][0]

            # model(iphone) calls the generator internally for pix2pix
            fake_fuji = model(iphone)

            psnr, ssim, lpips_val = calculator.evaluate_batch(fake_fuji, fuji)
            total_psnr  += psnr
            total_ssim  += ssim
            total_lpips += lpips_val

            # Denormalise [-1, 1] → [0, 1] before saving
            out_img = (fake_fuji + 1) / 2
            save_image(out_img, os.path.join(args.output_dir, f"{args.model}_{name}"))

    N = len(test_loader)
    metrics_dict = {
        "PSNR":  total_psnr  / N,
        "SSIM":  total_ssim  / N,
        "LPIPS": total_lpips / N,
    }

    metrics_path = os.path.join(args.output_dir, f"{args.model}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)

    print(f"Results for {args.model} saved to {metrics_path}:")
    print(f"PSNR:  {metrics_dict['PSNR']:.4f}")
    print(f"SSIM:  {metrics_dict['SSIM']:.4f}")
    print(f"LPIPS: {metrics_dict['LPIPS']:.4f}")

    # FID requires the real Fuji images as the reference distribution
    print(f"\nTo compute FID, run:")
    print(f"python -m pytorch_fid {args.dataset}/test/fuji {args.output_dir} --device {device}")


if __name__ == '__main__':
    evaluate()