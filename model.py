import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class SimplifiedChannelAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(c, c, 1, 1, 0)
        
    def forward(self, x):
        attn = self.conv(self.squeeze(x))
        return x * attn

class NAFBlock(nn.Module):
    def __init__(self, c, dw_expand=2, ffn_expand=2):
        super().__init__()
        self.norm1 = nn.InstanceNorm2d(c, affine=True)
        self.conv1 = nn.Conv2d(c, c * dw_expand, 1, 1, 0)
        self.dwconv = nn.Conv2d(c * dw_expand, c * dw_expand, 3, 1, 1, groups=c * dw_expand)
        self.sg = SimpleGate()
        self.sca = SimplifiedChannelAttention(c * dw_expand // 2)
        self.conv2 = nn.Conv2d(c * dw_expand // 2, c, 1, 1, 0)

        self.norm2 = nn.InstanceNorm2d(c, affine=True)
        self.conv3 = nn.Conv2d(c, c * ffn_expand, 1, 1, 0)
        self.sg2 = SimpleGate()
        self.conv4 = nn.Conv2d(c * ffn_expand // 2, c, 1, 1, 0)

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.sg(x)
        x = self.sca(x)
        x = self.conv2(x)
        x = x + identity

        identity = x
        x = self.norm2(x)
        x = self.conv3(x)
        x = self.sg2(x)
        x = self.conv4(x)
        x = x + identity
        return x

class NAFNetMVP(nn.Module):
    def __init__(self, in_channels=1, width=32):
        super().__init__()
        self.intro = nn.Conv2d(in_channels, width, 3, 1, 1)
        
        self.enc_block = NAFBlock(width)
        self.down = nn.Conv2d(width, width * 2, 2, 2, 0)
        
        self.mid_block = NAFBlock(width * 2)
        
        self.up = nn.ConvTranspose2d(width * 2, width, 2, 2, 0)
        self.dec_block = NAFBlock(width)
        
        self.sr_up = nn.ConvTranspose2d(width, width, kernel_size=2, stride=2, padding=0)
        self.ending = nn.Conv2d(width, in_channels, 3, 1, 1)

    def forward(self, x):
        x_intro = self.intro(x)
        x_enc = self.enc_block(x_intro)
        x_down = self.down(x_enc)
        x_mid = self.mid_block(x_down)
        x_up = self.up(x_mid)
        x_dec = self.dec_block(x_up + x_enc) 
        x_sr = self.sr_up(x_dec)
        out = self.ending(x_sr)
        
        base_up = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        return out + base_up