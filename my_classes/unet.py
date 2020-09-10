""" I got help from:
Ref: https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py
https://github.com/hlamba28/UNET-TGS/blob/master/TGS%20UNET.ipynb
"""
from collections import OrderedDict

import torch
import torch.nn as nn


class UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, init_features=16, dropout_p= 0.2):
        super().__init__()

        features = init_features

        # Encoding layers
        self.encoder1 = UNet._block(in_channels, features)   
        self.encoder2 = UNet._block(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck layer
        self.bottleneck = UNet._block(features * 8, features * 16)

        # Decoding layers
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet._block((features * 8) * 2, features * 8)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet._block((features * 2) * 2, features * 2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet._block(features * 2, features)

        # output layer
        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # Max Pool
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout
        self.dropout = nn.Dropout(dropout_p)



    def forward(self, x):

        # Encoding path
        enc1 = self.encoder1(x)
        p1 = self.dropout(self.pool(enc1))
        enc2 = self.encoder2(p1)
        p2 = self.dropout(self.pool(enc2))
        enc3 = self.encoder3(p2)
        p3 = self.dropout(self.pool(enc3))
        enc4 = self.encoder4(p3)
        p4 = self.dropout(self.pool(enc4))

        # Bottleneck
        bottleneck = self.bottleneck(p4)

        # Decoding path
        dec4 = self.dropout(self.upconv4(bottleneck))
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.dropout(self.upconv3(dec4))
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.dropout(self.upconv2(dec3))
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.dropout(self.upconv1(dec2))
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        # Output
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_features, out_features):
        return nn.Sequential(OrderedDict([
                    ("conv1",nn.Conv2d(
                        in_channels=in_features,
                        out_channels=out_features,
                        kernel_size=3,
                        padding=1,
                        bias=False)),
                    ("norm1", nn.BatchNorm2d(num_features=out_features)),
                    ("relu1", nn.ReLU(inplace=True)),
                    ("conv2",nn.Conv2d(
                        in_channels=out_features,
                        out_channels=out_features,
                        kernel_size=3,
                        padding=1,
                        bias=False)),
                    ("norm2", nn.BatchNorm2d(num_features=out_features)),
                    ("relu2", nn.ReLU(inplace=True))
                ]))