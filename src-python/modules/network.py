import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import math 
import numpy as np

class UNetUp(nn.Module):

    def __init__(self, in_channels, features, out_channels, is_bn = True):
        super().__init__()
        if is_bn:
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
        else:
            self.up = nn.Sequential(
                nn.Conv2d(in_channels, features, 3, padding = 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(features, features, 3, padding = 1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(features, out_channels, 2, stride=2),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.up(x)


class UNetDown(nn.Module):

    def __init__(self, in_channels, out_channels, is_bn = True):
        super().__init__()

        if is_bn:
            self.down = nn.Sequential(
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(in_channels, out_channels, 3, padding = 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding = 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.down = nn.Sequential(
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(in_channels, out_channels, 3, padding = 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding = 1),
                nn.ReLU(inplace=True)
            )

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

class UNet_radius_center_denseconv(nn.Module):

    def __init__(self, num_classes, num_channels, num_params=6, width=4, n_convs=10, is_bn = True, dense_bn = True):
        super().__init__()
        self.n_convs = n_convs
        self.is_bn = is_bn
        self.dense_bn = dense_bn
        if self.is_bn:
            self.first = nn.Sequential(
                nn.Conv2d(num_channels, width, 3, padding = 1),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
                nn.Conv2d(width, width, 3, padding = 1),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True)
            )
        else:
            self.first = nn.Sequential(
                nn.Conv2d(num_channels, width, 3, padding = 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(width, width, 3, padding = 1),
                nn.ReLU(inplace=True)
            )
            
        self.dec2 = UNetDown(width, width*2, is_bn = self.is_bn)
        self.dec3 = UNetDown(width*2, width*4, is_bn = self.is_bn)
        self.dec4 = UNetDown(width*4, width*8, is_bn = self.is_bn)

        self.center = nn.MaxPool2d(2, stride=2)
        self.center1 = nn.Conv2d(width*8, width*16, 3, padding = 1)
        self.center_bn = nn.BatchNorm2d(width*16)
        self.center_relu = nn.ReLU(inplace=True)
        self.center2 = nn.Conv2d(width*16, width*16, 3, padding = 1)
        self.center2_bn = nn.BatchNorm2d(width*16)
        self.center2relu = nn.ReLU(inplace=True)
        self.center3 = nn.ConvTranspose2d(width*16, width*8, 2, stride=2)
        self.center3_bn = nn.BatchNorm2d(width*8)
        self.center3_relu = nn.ReLU(inplace=True)
        
        #self.center4 = nn.AdaptiveAvgPool2d((1,1)
        #self.center4_relu = nn.ReLU(inplace=True)
        self.xyr_convs = []
        self.xyr_bns = []
        self.xyr_relus = []
        
        for i in range(self.n_convs):
            self.xyr_convs.append(nn.Conv2d(width*16*(i+1), width*16, 3, padding = 1))
            self.xyr_bns.append(nn.BatchNorm2d(width*16))
            self.xyr_relus.append(nn.ReLU(inplace=True))
        
        self.xyr_convs = nn.ModuleList(self.xyr_convs)
        self.xyr_bns = nn.ModuleList(self.xyr_bns)
        self.xyr_relus = nn.ModuleList(self.xyr_relus)
        
        
        # 64 x 20 x 15
        self.xyr_input = nn.Flatten()
        self.xyr_linear = nn.Linear(width*16 * 20 * 15, num_params)
        
        self.enc4 = UNetUp(width*16, width*8, width*4, is_bn = self.is_bn)
        self.enc3 = UNetUp(width*8, width*4, width*2, is_bn = self.is_bn)
        self.enc2 = UNetUp(width*4, width*2, width, is_bn = self.is_bn)
        if self.is_bn:
            self.enc1 = nn.Sequential(
                nn.Conv2d(width*2, width, 3, padding = 1),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
                nn.Conv2d(width, width, 3, padding = 1),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
            )
        else:
            self.enc1 = nn.Sequential(
                nn.Conv2d(width*2, width, 3, padding = 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(width, width, 3, padding = 1),
                nn.ReLU(inplace=True),
            )
        self.final = nn.Conv2d(width, num_classes, 1)
        self._initialize_weights()
     
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
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
        
        xyr = [center7]
        dense_col = [center7]
        for i in range(self.n_convs):
            xyr.append(self.xyr_convs[i](torch.cat(dense_col, 1)))
            if self.dense_bn:
                xyr.append(self.xyr_bns[i](xyr[-1]))
            dense_col.append(self.xyr_relus[i](xyr[-1]))
                
        
        xyr_lin1 = self.xyr_input(dense_col[-1])
        xyr_lin2 = self.xyr_linear(xyr_lin1)

        enc4 = self.enc4(torch.cat([center10, dec4], 1))
        enc3 = self.enc3(torch.cat([enc4, dec3], 1))
        enc2 = self.enc2(torch.cat([enc3, dec2], 1))
        enc1 = self.enc1(torch.cat([enc2, dec1], 1))

        return self.final(enc1), xyr_lin2
        
    def encode_params(self, x):
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
        
        xyr = [center7]
        dense_col = [center7]
        for i in range(self.n_convs):
            xyr.append(self.xyr_convs[i](torch.cat(dense_col, 1)))
            if self.dense_bn:
                xyr.append(self.xyr_bns[i](xyr[-1]))
            dense_col.append(self.xyr_relus[i](xyr[-1]))
                
        
        xyr_lin1 = self.xyr_input(dense_col[-1])
        xyr_lin2 = self.xyr_linear(xyr_lin1)

        return xyr_lin2