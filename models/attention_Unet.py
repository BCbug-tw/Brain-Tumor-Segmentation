import torch
from torch import nn

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class AttentionUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super(AttentionUNet, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(in_ch, 16)
        self.Conv2 = ConvBlock(16, 32)
        self.Conv3 = ConvBlock(32, 64)
        self.Conv4 = ConvBlock(64, 128)
        self.Conv5 = ConvBlock(128, 256)

        self.Up5 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.Att5 = AttentionBlock(128, 128, 64)
        self.Up_conv5 = ConvBlock(256, 128)

        self.Up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.Att4 = AttentionBlock(64, 64, 32)
        self.Up_conv4 = ConvBlock(128, 64)

        self.Up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.Att3 = AttentionBlock(32, 32, 16)
        self.Up_conv3 = ConvBlock(64, 32)

        self.Up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.Att2 = AttentionBlock(16, 16, 8)
        self.Up_conv2 = ConvBlock(32, 16)

        self.Conv_1x1 = nn.Conv2d(16, out_ch, kernel_size=1)

    def forward(self, x):
        # Encoding
        x1 = self.Conv1(x)
        x2 = self.Conv2(self.Maxpool(x1))
        x3 = self.Conv3(self.Maxpool(x2))
        x4 = self.Conv4(self.Maxpool(x3))
        x5 = self.Conv5(self.Maxpool(x4))

        # Decoding + Attention
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = self.Up_conv5(torch.cat((x4, d5), dim=1))

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = self.Up_conv4(torch.cat((x3, d4), dim=1))

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = self.Up_conv3(torch.cat((x2, d3), dim=1))

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = self.Up_conv2(torch.cat((x1, d2), dim=1))

        out = self.Conv_1x1(d2)
        return out
