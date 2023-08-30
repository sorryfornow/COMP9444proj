import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        # Encoding path
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.enc5 = self.conv_block(512, 1024)

        # Decoding path
        self.up1 = self.upconv_block(1024, 512)
        self.up2 = self.upconv_block(512, 256)
        self.up3 = self.upconv_block(256, 128)
        self.up4 = self.upconv_block(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, padding=0)
        self.out_activation = nn.Sigmoid()

    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return block

    def upconv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        return block

    def forward(self, x):
        # Encoder Path
        enc1 = self.enc1(x)
        enc2 = self.enc2(nn.MaxPool2d(2)(enc1))
        enc3 = self.enc3(nn.MaxPool2d(2)(enc2))
        enc4 = self.enc4(nn.MaxPool2d(2)(enc3))
        enc5 = self.enc5(nn.MaxPool2d(2)(enc4))

        # Decoder Path
        dec1 = self.up1(enc5)
        dec1 = torch.cat([dec1, enc4], dim=1)  # Double the channel size

        # Adjust the convolution block to handle the doubled channel size
        dec1 = self.conv_block(1024, 512)(dec1)

        dec2 = self.up2(dec1)
        dec2 = torch.cat([dec2, enc3], dim=1)
        dec2 = self.conv_block(512, 256)(dec2)

        dec3 = self.up3(dec2)
        dec3 = torch.cat([dec3, enc2], dim=1)
        dec3 = self.conv_block(256, 128)(dec3)

        dec4 = self.up4(dec3)
        dec4 = torch.cat([dec4, enc1], dim=1)
        dec4 = self.conv_block(128, 64)(dec4)

        ###########
        # for main_.py binary classification
        out = self.out_conv(dec4)
        # out = nn.AdaptiveAvgPool2d(1)(out)
        out = out.view(out.size(0), -1)  # Flatten the tensor
        # one-zero classification
        # out = nn.Sigmoid()(out)
        # out = torch.where(out > 0.5, torch.tensor([1.]).cuda(), torch.tensor([0.]).cuda())
        return self.out_activation(out)
        ###########

        # for main.py
        # return self.out_activation(self.out_conv(dec4))


model = UNet()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.BCELoss()
criterion = nn.BCELoss()  # binary cross entropy loss

print(model)

# pretrained weights, you can load them using:
# if pretrained_weights:
#     model.load_state_dict(torch.load("path_to_pretrained_weights.pth"))
