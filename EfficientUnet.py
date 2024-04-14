from torchsummary import summary
from torchvision.models import efficientnet_b0
from torch import nn
import torch
from Minerva.sslt.models.nets.unet import _OutConv, _DoubleConv
import torch.nn.functional as F
import torchvision.models as models

class _Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, cat_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )   # in_channels == out_channels
            self.conv = _DoubleConv(in_channels + cat_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = _DoubleConv(in_channels // 2 + cat_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW (channel, height, width)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # pad the input tensor on all sides with the given "pad" value
        x1 = F.pad(
            x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
        )
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class Up_no_cat(nn.Module):
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )
            self.conv = _DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = _DoubleConv(in_channels // 2, out_channels)
        
    def forward(self, x):
        x = self.up(x)
        return self.conv(x)
    
    
class EfficientUnet(nn.Module):
    
    def __init__(self, 
                    n_channels: int=2, 
                    n_classes: int=6, 
                    bilinear: bool=False,
                    pretrained_backbone=False):
        
        super().__init__()
        factor = 2 if bilinear else 1
        
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained_backbone else None
        
        net = efficientnet_b0(weights=weights)
        effnetb0_backbone = net.features[:6]
        
        in_layer = nn.Conv2d(n_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        
        # -- Encoder -- #
        
        self.encoder_block0 = effnetb0_backbone[0]      # (2, 256, 256) -> (32, 128, 128)
        self.encoder_block0[0] = in_layer               # n_channels
        self.encoder_block1 = effnetb0_backbone[1:3]    # (32, 128, 128) -> (24, 64, 64)
        self.encoder_block2 = effnetb0_backbone[3]      # (24, 64, 64) -> (40, 32, 32)
        self.encoder_block3 = effnetb0_backbone[4:6]    # (40, 32, 32) -> (112, 16, 16)
        
        # -- Decoder -- #
        
        # self.up1 = _Up(112, 40, 256 // factor, bilinear=bilinear)     # (112, 16, 16) -> (256, 32, 32)
        # self.up2 = _Up(256, 24, 128 // factor, bilinear=bilinear)     # (256, 32, 32) -> (128, 64, 64)
        # self.up3 = _Up(128, 32, 64 // factor, bilinear=bilinear)      # (128, 64, 64) -> (64, 128, 128)
        # self.up4 = _Up(64, 2, 32 // factor, bilinear=bilinear)        # (64, 128, 128) -> (32, 256, 256)
        
        self.up1 = _Up(112, 40, 256, bilinear=False)     # (112, 16, 16) -> (256, 32, 32)
        self.up2 = _Up(256, 24, 128, bilinear=False)     # (256, 32, 32) -> (128, 64, 64)
        self.up3 = _Up(128, 32, 64, bilinear=False)      # (128, 64, 64) -> (64, 128, 128)
        # self.up4 = _Up(64, 2, 32, bilinear=False)        # (64, 128, 128) -> (32, 256, 256)
        self.up4 = Up_no_cat(64, 32, bilinear=False)
        
        
        # -- Output -- #
        
        self.outc = _OutConv(32, n_classes)                      # (32, 256, 256) -> (n_classes, 256, 256)
        
        
    def forward(self, x):                       # x -> (2, 256, 256)
        # x0 = x
        x1 = self.encoder_block0(x)             # x1 -> (32, 128, 128)    
        x2 = self.encoder_block1(x1)            # x2 -> (24, 64, 64)
        x3 = self.encoder_block2(x2)            # x3 -> (40, 32, 32)
        x4 = self.encoder_block3(x3)            # x4 -> (112, 16, 16)
        
        x = self.up1(x4, x3)                    # x -> (256, 32, 32)
        x = self.up2(x, x2)                     # x -> (128, 64, 64)
        x = self.up3(x, x1)                     # x -> (64, 128, 128)
        # x = self.up4(x, x0)                      # x -> (32, 256, 256)
        x = self.up4(x)
        
        logits = self.outc(x)                    # logits -> (n_classes, 256, 256)
        
        return logits