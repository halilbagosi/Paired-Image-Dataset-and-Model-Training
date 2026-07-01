import os
import argparse
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from dataset import get_dataloaders
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


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',         type=str,   default='dataset', help='Dataset root path')
    parser.add_argument('--model',           type=str,   choices=['unet', 'pix2pix', 'pynet', 'restormer'], required=True)
    parser.add_argument('--epochs',          type=int,   default=50)
    parser.add_argument('--batch_size',      type=int,   default=8)
    parser.add_argument('--lr',              type=float, default=2e-4)
    parser.add_argument('--save_dir',        type=str,   default='checkpoints')
    parser.add_argument('--patience',        type=int,   default=5,         help='Patience for early stopping')
    parser.add_argument('--num_workers',     type=int,   default=4,         help='DataLoader workers')
    parser.add_argument('--grad_accum',      type=int,   default=1,         help='Gradient accumulation steps')
    parser.add_argument('--prefetch_factor', type=int,   default=2,         help='Batches each worker pre-loads into RAM (2-4 recommended)')
    parser.add_argument('--compile',         action='store_true',           help='torch.compile() the model')
    parser.add_argument('--resume',          action='store_true',           help='Resume from the last saved checkpoint')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device  = get_device()
    is_cuda = device.type == 'cuda'
    use_amp = device.type in ('cuda', 'mps')

    print(f"\n{'─'*50}")
    print(f"  Device          : {device}")
    print(f"  Training task   : iPhone  ->  Fuji")
    print(f"  Mixed precision : {'float16 v' if use_amp else 'disabled'}")
    print(f"  Batch size      : {args.batch_size}  (effective: {args.batch_size * args.grad_accum})")
    print(f"  Workers         : {args.num_workers}  prefetch x{args.prefetch_factor}")
    print(f"  Grad accum      : {args.grad_accum}")
    print(f"  torch.compile   : {'will attempt v' if args.compile else 'off'}")
    print(f"{'─'*50}\n")

    # ── DataLoaders ─────────────────────────────────────────────────────────
    train_loader, val_loader = get_dataloaders(
        args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=is_cuda,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor,
    )

    # ── Metrics + CSV log ───────────────────────────────────────────────────
    metrics_calculator = MetricsCalculator(device)

    csv_path = os.path.join(args.save_dir, f"{args.model}_training_log.csv")
    if not args.resume or not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'train_loss', 'val_psnr', 'val_ssim', 'val_lpips', 'lr'])

    # ── Model ───────────────────────────────────────────────────────────────
    model_cls = {'unet': UNet, 'pix2pix': Pix2Pix, 'pynet': PyNET, 'restormer': Restormer}
    model = model_cls[args.model]().to(device)

    if args.compile:
        try:
            model = torch.compile(model)
            print("torch.compile() active - first epoch slower while kernels compile.\n")
        except Exception as e:
            print(f"torch.compile() unavailable ({e}), continuing without it.\n")

    # ── Optimizers + Schedulers ─────────────────────────────────────────────
    if args.model == 'pix2pix':
        optimizer_G = optim.AdamW(model.generator.parameters(),     lr=args.lr, betas=(0.5, 0.999), weight_decay=1e-4)
        optimizer_D = optim.AdamW(model.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=1e-4)
        scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=args.epochs, eta_min=args.lr * 0.01)
        scheduler_D = optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=args.epochs, eta_min=args.lr * 0.01)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    scaler = torch.cuda.amp.GradScaler() if is_cuda else None

    best_psnr                  = -float('inf')
    epochs_without_improvement = 0
    start_epoch                = 1

    # ── Resume ──────────────────────────────────────────────────────────────
    resume_path = os.path.join(args.save_dir, f"{args.model}_resume.pth")

    if args.resume:
        if not os.path.exists(resume_path):
            print(f"  No resume checkpoint found at {resume_path}. Starting from scratch.\n")
        else:
            print(f"  Resuming from {resume_path}")
            ckpt = torch.load(resume_path, map_location=device)

            model.load_state_dict(ckpt['model_state_dict'])

            if args.model == 'pix2pix':
                optimizer_G.load_state_dict(ckpt['optimizer_G_state_dict'])
                optimizer_D.load_state_dict(ckpt['optimizer_D_state_dict'])
                scheduler_G.load_state_dict(ckpt['scheduler_G_state_dict'])
                scheduler_D.load_state_dict(ckpt['scheduler_D_state_dict'])
            else:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])

            best_psnr                  = ckpt['best_psnr']
            epochs_without_improvement = ckpt['epochs_without_improvement']
            start_epoch                = ckpt['epoch'] + 1

            print(f"  Resuming from epoch {start_epoch}  |  best PSNR so far: {best_psnr:.4f}\n")

    # ── Epoch loop ──────────────────────────────────────────────────────────
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            train_loss = 0.0

            if args.model == 'pix2pix':
                optimizer_G.zero_grad(set_to_none=True)
                optimizer_D.zero_grad(set_to_none=True)
            else:
                optimizer.zero_grad(set_to_none=True)

            pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{args.epochs}")

            for step, batch in pbar:
                iphone = batch['iphone'].to(device, non_blocking=is_cuda)
                fuji   = batch['fuji'].to(device,   non_blocking=is_cuda)

                if args.model == 'pix2pix':
                    # ── Discriminator step ────────────────────────────────
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                        fake_fuji = model.generator(iphone)

                        pred_real = model.discriminator(torch.cat([iphone, fuji],               dim=1))
                        pred_fake = model.discriminator(torch.cat([iphone, fake_fuji.detach()], dim=1))
                        loss_D = (
                            F.binary_cross_entropy_with_logits(pred_real, torch.ones_like(pred_real)) +
                            F.binary_cross_entropy_with_logits(pred_fake, torch.zeros_like(pred_fake))
                        ) * 0.5 / args.grad_accum

                    (scaler.scale(loss_D) if scaler else loss_D).backward()

                    # ── Generator step ────────────────────────────────────
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                        pred_fake  = model.discriminator(torch.cat([iphone, fake_fuji], dim=1))
                        loss_G_GAN = F.binary_cross_entropy_with_logits(pred_fake, torch.ones_like(pred_fake))
                        loss_G_pix = F.l1_loss(fake_fuji, fuji) * 100.0
                        loss_G     = (loss_G_GAN + loss_G_pix) / args.grad_accum

                    (scaler.scale(loss_G) if scaler else loss_G).backward()

                    if (step + 1) % args.grad_accum == 0:
                        if scaler:
                            scaler.unscale_(optimizer_D)
                            scaler.unscale_(optimizer_G)
                            nn.utils.clip_grad_norm_(model.discriminator.parameters(), 1.0)
                            nn.utils.clip_grad_norm_(model.generator.parameters(),     1.0)
                            scaler.step(optimizer_D)
                            scaler.step(optimizer_G)
                            scaler.update()
                        else:
                            nn.utils.clip_grad_norm_(model.discriminator.parameters(), 1.0)
                            nn.utils.clip_grad_norm_(model.generator.parameters(),     1.0)
                            optimizer_D.step()
                            optimizer_G.step()
                        optimizer_D.zero_grad(set_to_none=True)
                        optimizer_G.zero_grad(set_to_none=True)

                    train_loss += loss_G.item() * args.grad_accum
                    pbar.set_postfix({
                        'D':  f"{loss_D.item() * args.grad_accum:.3f}",
                        'G':  f"{loss_G.item() * args.grad_accum:.3f}",
                        'px': f"{loss_G_pix.item():.3f}",
                    })

                else:
                    # ── Standard supervised (L1) training ─────────────────
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                        fake_fuji = model(iphone)
                        loss      = F.l1_loss(fake_fuji, fuji) / args.grad_accum

                    (scaler.scale(loss) if scaler else loss).backward()

                    if (step + 1) % args.grad_accum == 0:
                        if scaler:
                            scaler.unscale_(optimizer)
                            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                    train_loss += loss.item() * args.grad_accum
                    pbar.set_postfix({'loss': f"{loss.item() * args.grad_accum:.4f}"})

            # ── LR scheduler step ────────────────────────────────────────────
            if args.model == 'pix2pix':
                scheduler_G.step()
                scheduler_D.step()
                current_lr = scheduler_G.get_last_lr()[0]
            else:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]

            # ── Validation ───────────────────────────────────────────────────
            model.eval()
            val_psnr = val_ssim = val_lpips = 0.0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="  Validating", leave=False):
                    iphone = batch['iphone'].to(device)
                    fuji   = batch['fuji'].to(device)

                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                        fake_fuji = model.generator(iphone) if args.model == 'pix2pix' else model(iphone)

                    fake_fuji = fake_fuji.float()
                    p, s, l   = metrics_calculator.evaluate_batch(fake_fuji, fuji.float())
                    val_psnr  += p
                    val_ssim  += s
                    val_lpips += l

            N         = len(val_loader)
            val_psnr  /= N
            val_ssim  /= N
            val_lpips /= N

            print(f"Epoch {epoch:3d} | lr {current_lr:.1e} | "
                  f"PSNR {val_psnr:.4f} | SSIM {val_ssim:.4f} | LPIPS {val_lpips:.4f}")

            # ── Log to CSV ───────────────────────────────────────────────────
            avg_train_loss = train_loss / len(train_loader)
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([epoch, avg_train_loss, val_psnr, val_ssim, val_lpips, current_lr])

            # ── Per-epoch model weights (for rollback) ───────────────────────
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"{args.model}_epoch_{epoch}.pth"))

            # ── Early stopping counters ──────────────────────────────────────
            if val_psnr > best_psnr:
                best_psnr                  = val_psnr
                epochs_without_improvement = 0
                torch.save(model.state_dict(), os.path.join(args.save_dir, f"{args.model}_best.pth"))
                print(f"  New best PSNR: {best_psnr:.4f}")
            else:
                epochs_without_improvement += 1
                print(f"  Early stop counter: {epochs_without_improvement}/{args.patience}")

            # ── Resume checkpoint (saved AFTER counters updated) ─────────────
            if args.model == 'pix2pix':
                resume_state = {
                    'epoch':                      epoch,
                    'model_state_dict':           model.state_dict(),
                    'optimizer_G_state_dict':     optimizer_G.state_dict(),
                    'optimizer_D_state_dict':     optimizer_D.state_dict(),
                    'scheduler_G_state_dict':     scheduler_G.state_dict(),
                    'scheduler_D_state_dict':     scheduler_D.state_dict(),
                    'best_psnr':                  best_psnr,
                    'epochs_without_improvement': epochs_without_improvement,
                }
            else:
                resume_state = {
                    'epoch':                      epoch,
                    'model_state_dict':           model.state_dict(),
                    'optimizer_state_dict':       optimizer.state_dict(),
                    'scheduler_state_dict':       scheduler.state_dict(),
                    'best_psnr':                  best_psnr,
                    'epochs_without_improvement': epochs_without_improvement,
                }
            torch.save(resume_state, resume_path)

            if epochs_without_improvement >= args.patience:
                print("Early stopping triggered. Training stopped.")
                break

    except KeyboardInterrupt:
        if os.path.exists(resume_path):
            last = torch.load(resume_path, map_location='cpu')['epoch']
            print(f"\n  Training interrupted after epoch {last}.")
            print(f"  To resume, add --resume to your command.")
        else:
            print(f"\n  Training interrupted before any epoch completed — nothing to resume from.")


if __name__ == '__main__':
    train()