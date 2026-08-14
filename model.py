import torch
import torch.nn as nn

class SimpleGate(nn.Module):
    def forward(self, x):
        # Splits the tensor in half across channels and multiplies them
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(c, c*2, kernel_size=1),
            nn.Conv2d(c*2, c*2, kernel_size=3, padding=1, groups=c*2), # Depthwise
            SimpleGate(),
            nn.Conv2d(c, c, kernel_size=1)
        )
    def forward(self, x):
        return x + self.seq(x) # Residual connection

class NAFNetMVP(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, dim=32):
        super().__init__()
        # 1. Feature Extraction (1 channel for Grayscale)
        self.intro = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)
        self.encoder_block = NAFBlock(dim)
        
        # 2. Upsampling (Resolving the down-sampled resolution loss)
        self.upsample = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2)
        
        # 3. Final Reconstruction
        self.decoder_block = NAFBlock(dim)
        self.outro = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid() # Bounds output strictly to [0, 1]

    def forward(self, x):
        x = self.intro(x)
        x = self.encoder_block(x)
        x = self.upsample(x)
        x = self.decoder_block(x)
        x = self.outro(x)
        return self.sigmoid(x)