from torch import nn
import warnings


class WDCNN(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 pool_size,
                 conv_padding):
        # conv_padding = ((in_channels - 1) * stride + kernel_size - in_channels) / 2
        super(WDCNN, self).__init__()

        self.layer1 = nn.Conv1d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=conv_padding)
        self.layer2 = nn.BatchNorm1d(out_channels)
        self.layer3 = nn.ReLU(inplace=False)
        self.layer4 = nn.MaxPool1d(kernel_size=pool_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class FullNet(nn.Module):
    def __init__(self, pretrained=False, in_channel=1, out_channel=10):
        super(FullNet, self).__init__()
        if pretrained:
            warnings.warn("Pretrained model is not available")

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channel,
                      out_channels=16,
                      kernel_size=64,
                      stride=16,
                      padding=32),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=False),
            nn.MaxPool1d(kernel_size=2))
        self.layer2 = WDCNN(in_channels=16,
                            out_channels=32,
                            kernel_size=3,
                            stride=1,
                            pool_size=2,
                            conv_padding=1)
        self.layer3 = WDCNN(in_channels=32,
                            out_channels=64,
                            kernel_size=3,
                            stride=1,
                            pool_size=2,
                            conv_padding=1)
        self.layer4 = WDCNN(in_channels=64,
                            out_channels=64,
                            kernel_size=3,
                            stride=1,
                            pool_size=2,
                            conv_padding=1)
        self.layer5 = WDCNN(in_channels=64,
                            out_channels=64,
                            kernel_size=3,
                            stride=1,
                            pool_size=2,
                            conv_padding=1)
        self.layer6 = nn.Flatten()
        self.layer7 = nn.Sequential(
            nn.Linear(64, 100),
            nn.ReLU(),
            nn.Linear(100, out_channel)
        )

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        return x
