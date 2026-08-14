import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ImageRestorationDataset
from model import NAFNetMVP

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # Configuration
    batch_size = 16
    num_epochs = 10
    lr = 1e-4

    # Setup DataLoader
    dataset = ImageRestorationDataset("./data/train/noisy/", "./data/train/gt/")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize Model & Optimizer
    model = NAFNetMVP().to(device)
    criterion = nn.L1Loss() # Baseline loss; upgrade to Charbonnier later
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for noisy, gt in loader:
            noisy, gt = noisy.to(device), gt.to(device)
            
            optimizer.zero_grad()
            outputs = model(noisy)
            loss = criterion(outputs, gt)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {running_loss/len(loader):.4f}")

    # Save the trained weights
    torch.save(model.state_dict(), "best_model.pth")
    print("Model saved to best_model.pth")

if __name__ == "__main__":
    train()