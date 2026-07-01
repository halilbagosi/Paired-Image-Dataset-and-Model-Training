import os
import re
import csv
import argparse
import torch
import torch.optim as optim

from models.unet import UNet
from models.pix2pix import Pix2Pix
from models.pynet import PyNET
from models.restormer import Restormer


def build_resume():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',    type=str, choices=['unet', 'pix2pix', 'pynet', 'restormer'], required=True)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--epochs',   type=int, default=100,  help='Must match --epochs used during original training')
    parser.add_argument('--lr',       type=float, default=2e-4, help='Must match --lr used during original training')
    args = parser.parse_args()

    # ── Find the latest epoch checkpoint ────────────────────────────────────
    pattern = re.compile(rf"^{re.escape(args.model)}_epoch_(\d+)\.pth$")
    epoch_files = []
    for fname in os.listdir(args.save_dir):
        m = pattern.match(fname)
        if m:
            epoch_files.append((int(m.group(1)), fname))

    if not epoch_files:
        print(f"No epoch checkpoints found in '{args.save_dir}' for model '{args.model}'.")
        return

    epoch_files.sort()
    last_epoch, last_file = epoch_files[-1]
    weights_path = os.path.join(args.save_dir, last_file)
    print(f"Latest checkpoint : {last_file}  (epoch {last_epoch})")

    # ── Read best PSNR and early-stop counter from the training log CSV ────────
    best_psnr                  = -float('inf')
    epochs_without_improvement = 0
    csv_path = os.path.join(args.save_dir, f"{args.model}_training_log.csv")

    if os.path.exists(csv_path):
        rows = []
        with open(csv_path, 'r') as f:
            for row in csv.DictReader(f):
                try:
                    rows.append((int(row['epoch']), float(row['val_psnr'])))
                except (KeyError, ValueError):
                    pass

        if rows:
            # Best PSNR seen across all logged epochs
            best_epoch, best_psnr = max(rows, key=lambda r: r[1])
            # Epochs since that best — this is the correct early-stop counter
            epochs_without_improvement = last_epoch - best_epoch
            print(f"Best PSNR from log : {best_psnr:.4f}  (epoch {best_epoch})")
            print(f"Epochs since best  : {epochs_without_improvement}")
        else:
            print("Could not parse CSV rows — best_psnr set to -inf, counter set to 0.")
    else:
        print("No training log found — best_psnr set to -inf, counter set to 0.")

    # ── Build model and load weights ─────────────────────────────────────────
    model_cls = {'unet': UNet, 'pix2pix': Pix2Pix, 'pynet': PyNET, 'restormer': Restormer}
    model     = model_cls[args.model]()
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))

    # ── Build optimizers and schedulers, step scheduler to correct position ──
    # Optimizer momentum history is lost (unavoidable from old checkpoints) but
    # the scheduler LR is reconstructed exactly by replaying the same steps.
    if args.model == 'pix2pix':
        optimizer_G = optim.AdamW(model.generator.parameters(),     lr=args.lr, betas=(0.5, 0.999), weight_decay=1e-4)
        optimizer_D = optim.AdamW(model.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=1e-4)
        scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=args.epochs, eta_min=args.lr * 0.01)
        scheduler_D = optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=args.epochs, eta_min=args.lr * 0.01)

        for _ in range(last_epoch):
            scheduler_G.step()
            scheduler_D.step()

        print(f"LR after replay   : {scheduler_G.get_last_lr()[0]:.2e}")

        resume_state = {
            'epoch':                      last_epoch,
            'model_state_dict':           model.state_dict(),
            'optimizer_G_state_dict':     optimizer_G.state_dict(),
            'optimizer_D_state_dict':     optimizer_D.state_dict(),
            'scheduler_G_state_dict':     scheduler_G.state_dict(),
            'scheduler_D_state_dict':     scheduler_D.state_dict(),
            'best_psnr':                  best_psnr,
            'epochs_without_improvement': epochs_without_improvement,
        }

    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

        for _ in range(last_epoch):
            scheduler.step()

        print(f"LR after replay   : {scheduler.get_last_lr()[0]:.2e}")

        resume_state = {
            'epoch':                      last_epoch,
            'model_state_dict':           model.state_dict(),
            'optimizer_state_dict':       optimizer.state_dict(),
            'scheduler_state_dict':       scheduler.state_dict(),
            'best_psnr':                  best_psnr,
            'epochs_without_improvement': epochs_without_improvement,
        }

    # ── Save ─────────────────────────────────────────────────────────────────
    resume_path = os.path.join(args.save_dir, f"{args.model}_resume.pth")
    torch.save(resume_state, resume_path)

    print(f"\nResume checkpoint saved to: {resume_path}")
    print(f"Training will continue from epoch {last_epoch + 1}.")
    print(f"\nNow run:")
    print(f"  python train.py --model {args.model} ... --resume")


if __name__ == '__main__':
    build_resume()