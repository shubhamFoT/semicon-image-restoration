import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pytorch_msssim import ssim
from tqdm import tqdm

from model import NAFNetMVP
from dataset import SemiconductorDataset


class RestorationLoss(nn.Module):
    """
    Multi-objective loss combining:
    1. Charbonnier Loss (robust edge-preserving spatial reconstruction)
    2. SSIM Loss (structural fidelity)
    3. 2D FFT Loss (high-frequency spectral geometry preservation)
    """
    def __init__(self, eps=1e-3, w_char=1.0, w_ssim=0.2, w_fft=0.1):
        super().__init__()
        self.eps2 = eps ** 2
        self.w_char = w_char
        self.w_ssim = w_ssim
        self.w_fft = w_fft

    def forward(self, pred, target):
        # Charbonnier loss (smooth L1)
        char = torch.sqrt((pred - target) ** 2 + self.eps2).mean()

        # Structural Dissimilarity Loss
        ssim_val = ssim(pred, target, data_range=1.0, size_average=True)
        ssim_loss = 1.0 - ssim_val

        # 2D Real Fast Fourier Transform Loss (orthonormal norm)
        pf = torch.fft.rfft2(pred, norm='ortho')
        tf = torch.fft.rfft2(target, norm='ortho')
        fft_loss = (pf.abs() - tf.abs()).abs().mean()

        return (self.w_char * char) + (self.w_ssim * ssim_loss) + (self.w_fft * fft_loss)


def calculate_psnr(pred, target, max_val=1.0):
    """Computes Peak Signal-to-Noise Ratio (PSNR) in dB."""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20.0 * torch.log10(max_val / torch.sqrt(mse))


def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0

    for noisy, gt in tqdm(loader, desc="Training", leave=False):
        noisy = noisy.to(device)
        gt = gt.to(device)

        optimizer.zero_grad()

        if scaler is not None and device.type == 'cuda':
            with torch.cuda.amp.autocast():
                pred = model(noisy)
                loss = criterion(pred, gt)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(noisy)
            loss = criterion(pred, gt)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * noisy.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_metrics(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0

    for noisy, gt in tqdm(loader, desc="Validation", leave=False):
        noisy = noisy.to(device)
        gt = gt.to(device)

        pred = model(noisy)
        pred = torch.clamp(pred, 0.0, 1.0)

        loss = criterion(pred, gt)
        total_loss += loss.item() * noisy.size(0)

        psnr_val = calculate_psnr(pred, gt)
        ssim_val = ssim(pred, gt, data_range=1.0, size_average=True)

        total_psnr += psnr_val.item() * noisy.size(0)
        total_ssim += ssim_val.item() * noisy.size(0)

    dataset_size = len(loader.dataset)
    return total_loss / dataset_size, total_psnr / dataset_size, total_ssim / dataset_size


def main():
    parser = argparse.ArgumentParser(description="Train NAFNet for Semiconductor Restorations")
    parser.add_argument('--noisy_dir', type=str, default='./data/train/noisy', help='Path to degraded images')
    parser.add_argument('--gt_dir', type=str, default='./data/train/gt', help='Path to ground-truth images')
    parser.add_argument('--epochs', type=int, default=50, help='Total training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Mini-batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--val_split', type=float, default=0.15, help='Fraction of data used for validation')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to store model checkpoints')
    parser.add_argument('--use_amp', action='store_true', help='Enable Automatic Mixed Precision')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Execution Target Device: {device}")

    # 1. Dataset Loading & Validation Split
    full_dataset = SemiconductorDataset(args.noisy_dir, args.gt_dir)
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )

    pin_mem = (device.type == 'cuda')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=pin_mem)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=pin_mem)

    print(f"Dataset summary: {train_size} train samples, {val_size} validation samples.")

    # 2. Pipeline Initialization
    model = NAFNetMVP().to(device)
    criterion = RestorationLoss(eps=1e-3, w_char=1.0, w_ssim=0.2, w_fft=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler() if (args.use_amp and device.type == 'cuda') else None

    # 3. Execution Loop
    best_val_psnr = -1.0
    best_model_path = os.path.join(args.save_dir, 'best_model.pth')

    print("\n--- Commencing Training Loop ---")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_psnr, val_ssim = evaluate_metrics(model, val_loader, criterion, device)
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch:03d}/{args.epochs:03d}] | LR: {current_lr:.6f} | "
              f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | "
              f"Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f}")

        # Checkpoint based strictly on validation score
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save(model.state_dict(), best_model_path)
            # Save a copy in repository root for standalone evaluation scripts
            torch.save(model.state_dict(), 'best_model.pth')
            print(f" -> Checkpoint updated: best_model.pth (PSNR: {best_val_psnr:.2f} dB)")

    # Save final epoch weights
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'latest_model.pth'))
    print("\n--- Training Completed ---")
    print(f"Optimal Validation PSNR: {best_val_psnr:.2f} dB")
    print(f"Optimal Checkpoint: {best_model_path}")


if __name__ == '__main__':
    main()
