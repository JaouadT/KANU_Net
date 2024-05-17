"""
Pytorch implementation of U-Net based on Kolmogorov-Arnold Network, based on the U-Net implementation in https://github.com/milesial/Pytorch-UNet.
The U-Net model is modified to use the FastKANConvLayer instead of the Conv2d layer in the original implementation.
The Convolution operation is implemented in https://github.com/XiangboGaoBarry/KA-Conv
"""

import torch
from torch import nn
import torch.nn.functional as F

from src.fastkanconv import FastKANConvLayer

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, device):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device

        self.double_conv = nn.Sequential(
            FastKANConvLayer(self.in_channels, self.out_channels//2, padding=1, kernel_size=3, stride=1, kan_type='RBF'),
            nn.BatchNorm2d(self.out_channels//2),
            nn.ReLU(inplace=True),
            FastKANConvLayer(self.out_channels//2, self.out_channels, padding=1, kernel_size=3, stride=1, kan_type='RBF'),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, device='mps'):
        super().__init__()
        self.device = device
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, device=self.device)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, device='mps'):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, device=device)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = FastKANConvLayer(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class KANU_Net(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, device='mps'):
        super(KANU_Net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.device = device

        self.channels = [64, 128, 256, 512, 1024]

        self.inc = (DoubleConv(n_channels, 64, device=self.device))
        
        self.down1 = (Down(self.channels[0], self.channels[1], self.device))
        self.down2 = (Down(self.channels[1], self.channels[2], self.device))
        self.down3 = (Down(self.channels[2], self.channels[3], self.device))
        factor = 2 if bilinear else 1
        self.down4 = (Down(self.channels[3], self.channels[4] // factor, self.device))
        self.up1 = (Up(self.channels[4], self.channels[3] // factor, bilinear, self.device))
        self.up2 = (Up(self.channels[3], self.channels[2] // factor, bilinear, self.device))
        self.up3 = (Up(self.channels[2], self.channels[1] // factor, bilinear, self.device))
        self.up4 = (Up(self.channels[1], self.channels[0], bilinear, self.device))
        self.outc = (OutConv(self.channels[0], n_classes))

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        #Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# if __name__ == "__main__":
#     device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
#     # print(device)
#     model = KANU_Net(3, 1, 'mps').to(device)
#     # print(model)
#     x = torch.randn((1, 3, 224, 224)).to(device)
#     print(model(x).shape)

