import torch
import torch.nn as nn
from torchsummary import summary

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 5, padding=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 5, padding=2),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    def __init__(self, cc=16, num_class=3):
        super().__init__()

        self.dconv_down1 = double_conv(1, cc)
        self.dconv_down2 = double_conv(cc, 2*cc)
        self.dconv_down3 = double_conv(2*cc, 4*cc)
        self.dconv_down4 = double_conv(4*cc, 8*cc)
        self.dconv_down5 = double_conv(8*cc, 16*cc)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up4 = double_conv(16*cc + 8*cc, 8*cc)
        self.dconv_up3 = double_conv(8*cc + 4*cc, 4*cc)
        self.dconv_up2 = double_conv(4*cc + 2*cc, 2*cc)
        self.dconv_up1 = double_conv(2*cc + cc, cc)

        self.conv_last = nn.Conv2d(in_channels=cc, out_channels=num_class, kernel_size=1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)

        conv5 = self.dconv_down5(x)
        #x = self.maxpool(conv5)

        x = self.upsample(conv5)
        x = torch.cat([x, conv4], dim=1)

        x = self.dconv_up4(x)
        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out

if __name__ == '__main__':
    model = UNet()
    summary(model, (1, 192, 640))

    print(model)