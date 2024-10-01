import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)
    

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x, skip_connection):
        x = self.up(x)
        x = torch.cat([x, skip_connection], dim=1)  # Concatenate along channel dimension
        return self.conv(x)
    


class UNet(nn.Module):
    def __init__(self, preceding_rainfall_days, forecast_rainfall_days, dropout_prob, base_channels=16):
        super(UNet, self).__init__()
        self.preceding_rainfall_days = preceding_rainfall_days
        self.forecast_rainfall_days = forecast_rainfall_days
        self.rainfall_sequence_length = (preceding_rainfall_days + forecast_rainfall_days)*8

        input_dim = self.rainfall_sequence_length + 1 + 1 #1 for soil moisture, 1 for topology
        self.dropout_prob = dropout_prob
        self.name = None
        self.base_channels = base_channels
        

        # Encoder path
        self.conv1 = ConvBlock(input_dim, base_channels)
        self.conv2 = ConvBlock(base_channels, base_channels * 2)
        self.conv3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.conv4 = ConvBlock(base_channels * 4, base_channels * 8)

        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder path
        self.upconv4 = UpBlock(base_channels * 8, base_channels * 4)
        self.upconv3 = UpBlock(base_channels * 4, base_channels * 2)
        self.upconv2 = UpBlock(base_channels * 2, base_channels)

        # Final layer to get the desired number of output channels (e.g. number of classes)
        self.finalconv = nn.Sequential(
            nn.Conv2d(base_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder path
        skip1 = self.conv1(x)
        x = self.downsample(skip1)
        skip2 = self.conv2(x)
        x = self.downsample(skip2)
        skip3 = self.conv3(x)
        x = self.downsample(skip3)
        x = self.conv4(x)

        # Decoder path
        x = self.upconv4(x, skip3)
        x = self.upconv3(x, skip2)
        x = self.upconv2(x, skip1)

        # Final convolution layer
        x = self.finalconv(x)

        return x

