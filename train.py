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
    def __init__(self, eps=1e-3, w_char=1.0, w_ssim=0.2, w_fft=0.1):
        super().__init__()
        self.eps2 = eps ** 2
        self.w_char = w_char
        self.w_ssim = w_ssim
        self.w_fft = w_fft

    def forward(self, pred, target):
        char = torch.sqrt((pred - target) ** 2 + self.eps2).mean()
        ssim_loss = 1.0 - ssim(pred, target, data_range=1.0, size_average=True)
        pf = torch.fft.rfft2(pred, norm='ortho')
        tf = torch.fft.rfft2(target, norm='ortho')
        fft_loss = (pf.abs() - tf.abs()).abs().mean()
        return (self.w_char * char) + (self.w_ssim * ssim_loss) + (self.w_fft * fft_loss)

def calculate_psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20.0 * torch.log10(max_val / torch.sqrt(mse))

def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    for noisy, gt in tqdm(loader, desc="Training", leave=False):
        noisy, gt = noisy.to(device), gt.to(device)
        optimizer.zero_grad()
        if scaler is not None and device.type == 'cuda':
            with torch.amp.autocast('cuda'):
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
    total_loss, total_psnr, total_ssim = 0.0, 0.0, 0.0
    for noisy, gt in tqdm(loader, desc="Validation", leave=False):
        noisy, gt = noisy.to(device), gt.to(device)
        pred = torch.clamp(model(noisy), 0.0, 1.0)
        loss = criterion(pred, gt)
        total_loss += loss.item() * noisy.size(0)
        total_psnr += calculate_psnr(pred, gt).item() * noisy.size(0)
        total_ssim += ssim(pred, gt, data_range=1.0, size_average=True).item() * noisy.size(0)
    return total_loss / len(loader.dataset), total_psnr / len(loader.dataset), total_ssim / len(loader.dataset)

def main():
    parser = argparse.ArgumentParser(description="Train NAFNet")
    parser.add_argument('--noisy_dir', type=str, default='./data/train/noisy')
    parser.add_argument('--gt_dir', type=str, default='./data/train/gt')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val_split', type=float, default=0.15)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--use_amp', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    full_dataset = SemiconductorDataset(args.noisy_dir, args.gt_dir)
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    pin_mem = (device.type == 'cuda')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=pin_mem)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=pin_mem)

    model = NAFNetMVP().to(device)
    criterion = RestorationLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') if (args.use_amp and device.type == 'cuda') else None

    best_val_psnr = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_psnr, val_ssim = evaluate_metrics(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:03d}/{args.epochs:03d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val PSNR: {val_psnr:.2f} | Val SSIM: {val_ssim:.4f}")
        
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            torch.save(model.state_dict(), 'best_model.pth')

if __name__ == '__main__':
    main()