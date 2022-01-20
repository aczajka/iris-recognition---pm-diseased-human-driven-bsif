import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import math 
import numpy as np

class UNetUp(nn.Module):

    def __init__(self, in_channels, features, out_channels):
        super().__init__()

        self.up = nn.Sequential(
            nn.Conv2d(in_channels, features, 3, padding = 1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, padding = 1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(features, out_channels, 2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)

class UNetDown(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        layers = [
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(in_channels, out_channels, 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ]

        self.down = nn.Sequential(*layers)

    def forward(self, x):
        return self.down(x)

class UNet(nn.Module):

    def __init__(self, num_classes, num_channels):
        super().__init__()

        self.first = nn.Sequential(
            nn.Conv2d(num_channels, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True)
        )
        self.dec2 = UNetDown(4, 8)
        self.dec3 = UNetDown(8, 16)
        self.dec4 = UNetDown(16, 32)

        self.center = nn.MaxPool2d(2, stride=2)
        self.center1 = nn.Conv2d(32, 64, 3, padding = 1)
        self.center_bn = nn.BatchNorm2d(64)
        self.center_relu = nn.ReLU(inplace=True)
        self.center2 = nn.Conv2d(64, 64, 3, padding = 1)
        self.center2_bn = nn.BatchNorm2d(64)
        self.center2relu = nn.ReLU(inplace=True)
        self.center3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.center3_bn = nn.BatchNorm2d(32)
        self.center3_relu = nn.ReLU(inplace=True)
        
        self.enc4 = UNetUp(64, 32, 16)
        self.enc3 = UNetUp(32, 16, 8)
        self.enc2 = UNetUp(16, 8, 4)
        self.enc1 = nn.Sequential(
            nn.Conv2d(8, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(4, num_classes, 1)
        self._initialize_weights()
     
    def _initialize_weights(self):
            for m in self.modules():
                 if isinstance(m, nn.Conv2d):
                     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                     m.weight.data.normal_(0, np.sqrt(2. / n))
                     #print(m.bias)
                     #if m.bias:
                     init.constant_(m.bias, 0)
                 elif isinstance(m, nn.BatchNorm2d):
                     m.weight.data.fill_(1)
                     m.bias.data.zero_()

    def forward(self, x):
        dec1 = self.first(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)

        center1 = self.center(dec4)
        center2 = self.center1(center1)
        center3 = self.center_bn(center2)
        center4 = self.center_relu(center3)
        center5 = self.center2(center4)
        center6 = self.center2_bn(center5)
        center7 = self.center2relu(center6)
        center8 = self.center3(center7)
        center9 = self.center3_bn(center8)
        center10 = self.center3_relu(center9)

        enc4 = self.enc4(torch.cat([center10, dec4], 1))
        enc3 = self.enc3(torch.cat([enc4, dec3], 1))
        enc2 = self.enc2(torch.cat([enc3, dec2], 1))
        enc1 = self.enc1(torch.cat([enc2, dec1], 1))

        return self.final(enc1)

class UNet_radius_center_conv10(nn.Module):

    def __init__(self, num_classes, num_channels, residual=False):
        super().__init__()
        self.residual = residual
        self.first = nn.Sequential(
            nn.Conv2d(num_channels, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True)
        )
        self.dec2 = UNetDown(4, 8)
        self.dec3 = UNetDown(8, 16)
        self.dec4 = UNetDown(16, 32)

        self.center = nn.MaxPool2d(2, stride=2)
        self.center1 = nn.Conv2d(32, 64, 3, padding = 1)
        self.center_bn = nn.BatchNorm2d(64)
        self.center_relu = nn.ReLU(inplace=True)
        self.center2 = nn.Conv2d(64, 64, 3, padding = 1)
        self.center2_bn = nn.BatchNorm2d(64)
        self.center2relu = nn.ReLU(inplace=True)
        self.center3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.center3_bn = nn.BatchNorm2d(32)
        self.center3_relu = nn.ReLU(inplace=True)
        
        #self.center4 = nn.AdaptiveAvgPool2d((1,1)
        #self.center4_relu = nn.ReLU(inplace=True)
        self.xyr1_conv = nn.Conv2d(64, 64, 3, padding = 1)
        self.xyr1_bn = nn.BatchNorm2d(64)
        self.xyr1_relu = nn.ReLU(inplace=True)
        # 64 x 20 x 15
        self.xyr2_conv = nn.Conv2d(64, 64, 3, padding = 1)
        self.xyr2_bn = nn.BatchNorm2d(64)
        self.xyr2_relu = nn.ReLU(inplace=True)
        # 64 x 20 x 15
        self.xyr3_conv = nn.Conv2d(64, 64, 3, padding = 1)
        self.xyr3_bn = nn.BatchNorm2d(64)
        self.xyr3_relu = nn.ReLU(inplace=True)
        # 64 x 20 x 15
        self.xyr4_conv = nn.Conv2d(64, 64, 3, padding = 1)
        self.xyr4_bn = nn.BatchNorm2d(64)
        self.xyr4_relu = nn.ReLU(inplace=True)
        
        self.xyr5_conv = nn.Conv2d(64, 64, 3, padding = 1)
        self.xyr5_bn = nn.BatchNorm2d(64)
        self.xyr5_relu = nn.ReLU(inplace=True)
        
        self.xyr6_conv = nn.Conv2d(64, 64, 3, padding = 1)
        self.xyr6_bn = nn.BatchNorm2d(64)
        self.xyr6_relu = nn.ReLU(inplace=True)
        
        self.xyr7_conv = nn.Conv2d(64, 64, 3, padding = 1)
        self.xyr7_bn = nn.BatchNorm2d(64)
        self.xyr7_relu = nn.ReLU(inplace=True)
        
        self.xyr8_conv = nn.Conv2d(64, 64, 3, padding = 1)
        self.xyr8_bn = nn.BatchNorm2d(64)
        self.xyr8_relu = nn.ReLU(inplace=True)
        
        self.xyr9_conv = nn.Conv2d(64, 64, 3, padding = 1)
        self.xyr9_bn = nn.BatchNorm2d(64)
        self.xyr9_relu = nn.ReLU(inplace=True)
        
        self.xyr10_conv = nn.Conv2d(64, 64, 3, padding = 1)
        self.xyr10_bn = nn.BatchNorm2d(64)
        self.xyr10_relu = nn.ReLU(inplace=True)
        
        # 64 x 20 x 15
        self.xyr11_input = nn.Flatten()
        self.xyr11_linear = nn.Linear(64 * 20 * 15, 6)
        
        self.enc4 = UNetUp(64, 32, 16)
        self.enc3 = UNetUp(32, 16, 8)
        self.enc2 = UNetUp(16, 8, 4)
        self.enc1 = nn.Sequential(
            nn.Conv2d(8, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(4, num_classes, 1)
        self._initialize_weights()
     
    def _initialize_weights(self):
            for m in self.modules():
                 if isinstance(m, nn.Conv2d):
                     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                     m.weight.data.normal_(0, np.sqrt(2. / n))
                     #print(m.bias)
                     #if m.bias:
                     init.constant_(m.bias, 0)
                 elif isinstance(m, nn.BatchNorm2d):
                     m.weight.data.fill_(1)
                     m.bias.data.zero_()

    def forward(self, x):
        dec1 = self.first(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)

        center1 = self.center(dec4)
        center2 = self.center1(center1)
        center3 = self.center_bn(center2)
        center4 = self.center_relu(center3)
        center5 = self.center2(center4)
        center6 = self.center2_bn(center5)
        center7 = self.center2relu(center6)
        center8 = self.center3(center7)
        center9 = self.center3_bn(center8)
        center10 = self.center3_relu(center9)
        
        xyr1 = self.xyr1_conv(center7)
        xyr2 = self.xyr1_bn(xyr1)
        xyr3 = self.xyr1_relu(xyr2)
        
        if self.residual:
            xyr3 = xyr3 + center7
        
        xyr4 = self.xyr2_conv(xyr3)
        xyr5 = self.xyr2_bn(xyr4)
        xyr6 = self.xyr2_relu(xyr5)
        
        if self.residual:
            xyr6 = xyr6 + xyr3
        
        xyr7 = self.xyr3_conv(xyr6)
        xyr8 = self.xyr3_bn(xyr7)
        xyr9 = self.xyr3_relu(xyr8)
        
        if self.residual:
            xyr9 = xyr9 + xyr6
        
        xyr10 = self.xyr4_conv(xyr9)
        xyr11 = self.xyr4_bn(xyr10)
        xyr12 = self.xyr4_relu(xyr11)
        
        if self.residual:
            xyr12 = xyr12 + xyr9
        
        xyr13 = self.xyr5_conv(xyr12)
        xyr14 = self.xyr5_bn(xyr13)
        xyr15 = self.xyr5_relu(xyr14)
        
        if self.residual:
            xyr15 = xyr15 + xyr12
            
        xyr16 = self.xyr6_conv(xyr15)
        xyr17 = self.xyr6_bn(xyr16)
        xyr18 = self.xyr6_relu(xyr17)
        
        if self.residual:
            xyr18 = xyr18 + xyr15
        
        xyr19 = self.xyr7_conv(xyr18)
        xyr20 = self.xyr7_bn(xyr19)
        xyr21 = self.xyr7_relu(xyr20)
        
        if self.residual:
            xyr21 = xyr21 + xyr18
        
        xyr22 = self.xyr8_conv(xyr21)
        xyr23 = self.xyr8_bn(xyr22)
        xyr24 = self.xyr8_relu(xyr23)
        
        if self.residual:
            xyr24 = xyr24 + xyr21
        
        xyr25 = self.xyr9_conv(xyr24)
        xyr26 = self.xyr9_bn(xyr25)
        xyr27 = self.xyr9_relu(xyr26)
        
        if self.residual:
            xyr27 = xyr27 + xyr24
        
        xyr28 = self.xyr10_conv(xyr27)
        xyr29 = self.xyr10_bn(xyr28)
        xyr30 = self.xyr10_relu(xyr29)
        
        if self.residual:
            xyr30 = xyr30 + xyr27
        
        xyr31 = self.xyr11_input(xyr30)
        xyr32 = self.xyr11_linear(xyr31)

        enc4 = self.enc4(torch.cat([center10, dec4], 1))
        enc3 = self.enc3(torch.cat([enc4, dec3], 1))
        enc2 = self.enc2(torch.cat([enc3, dec2], 1))
        enc1 = self.enc1(torch.cat([enc2, dec1], 1))

        return self.final(enc1), xyr32
        
    def encode_xyr(self, x):
        dec1 = self.first(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)

        center1 = self.center(dec4)
        center2 = self.center1(center1)
        center3 = self.center_bn(center2)
        center4 = self.center_relu(center3)
        center5 = self.center2(center4)
        center6 = self.center2_bn(center5)
        center7 = self.center2relu(center6)
        center8 = self.center3(center7)
        center9 = self.center3_bn(center8)
        center10 = self.center3_relu(center9)
        
        xyr1 = self.xyr1_conv(center7)
        xyr2 = self.xyr1_bn(xyr1)
        xyr3 = self.xyr1_relu(xyr2)
        
        if self.residual:
            xyr3 = xyr3 + center7
        
        xyr4 = self.xyr2_conv(xyr3)
        xyr5 = self.xyr2_bn(xyr4)
        xyr6 = self.xyr2_relu(xyr5)
        
        if self.residual:
            xyr6 = xyr6 + xyr3
        
        xyr7 = self.xyr3_conv(xyr6)
        xyr8 = self.xyr3_bn(xyr7)
        xyr9 = self.xyr3_relu(xyr8)
        
        if self.residual:
            xyr9 = xyr9 + xyr6
        
        xyr10 = self.xyr4_conv(xyr9)
        xyr11 = self.xyr4_bn(xyr10)
        xyr12 = self.xyr4_relu(xyr11)
        
        if self.residual:
            xyr12 = xyr12 + xyr9
        
        xyr13 = self.xyr5_conv(xyr12)
        xyr14 = self.xyr5_bn(xyr13)
        xyr15 = self.xyr5_relu(xyr14)
        
        if self.residual:
            xyr15 = xyr15 + xyr12
            
        xyr16 = self.xyr6_conv(xyr15)
        xyr17 = self.xyr6_bn(xyr16)
        xyr18 = self.xyr6_relu(xyr17)
        
        if self.residual:
            xyr18 = xyr18 + xyr15
        
        xyr19 = self.xyr7_conv(xyr18)
        xyr20 = self.xyr7_bn(xyr19)
        xyr21 = self.xyr7_relu(xyr20)
        
        if self.residual:
            xyr21 = xyr21 + xyr18
        
        xyr22 = self.xyr8_conv(xyr21)
        xyr23 = self.xyr8_bn(xyr22)
        xyr24 = self.xyr8_relu(xyr23)
        
        if self.residual:
            xyr24 = xyr24 + xyr21
        
        xyr25 = self.xyr9_conv(xyr24)
        xyr26 = self.xyr9_bn(xyr25)
        xyr27 = self.xyr9_relu(xyr26)
        
        if self.residual:
            xyr27 = xyr27 + xyr24
        
        xyr28 = self.xyr10_conv(xyr27)
        xyr29 = self.xyr10_bn(xyr28)
        xyr30 = self.xyr10_relu(xyr29)
        
        if self.residual:
            xyr30 = xyr30 + xyr27
        
        xyr31 = self.xyr11_input(xyr30)
        xyr32 = self.xyr11_linear(xyr31)
        
        return xyr32