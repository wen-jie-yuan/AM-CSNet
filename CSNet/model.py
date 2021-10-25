import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes=64, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class AM(nn.Module):
    def __init__(self, in_planes=64, ratio=16, kernel_size=7):
        super(AM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


class CBAM(nn.Module):
    def __init__(self):
        super(CBAM, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
        self.body = AM()

    def forward(self, x):
        res1 = self.layers(x)
        res2 = self.body(res1)
        return res2


class RFA(nn.Module):
    def __init__(self):
        super(RFA, self).__init__()
        self.out = CBAM()
        self.conv_cat = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out1 = self.out(x) + x
        out2 = self.out(out1) + out1
        out3 = self.out(out2) + out2
        out4 = self.out(out3)

        local_features = [out1, out2, out3, out4]
        out5 = self.conv_cat(torch.cat(local_features, 1)) + x
        return out5


class RFAN(nn.Module):
    def __init__(self, num_channels, num_features, num_blocks):
        super(RFAN, self).__init__()
        self.num_blocks = num_blocks

        # Use convolutional layers to sample and compress the original image
        self.sample = torch.nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=num_features, kernel_size=32, stride=32, padding=0,
                      bias=False))
        # Initial reconstruction
        self.initialization = torch.nn.Sequential(
            nn.ConvTranspose2d(in_channels=num_features, out_channels=1, kernel_size=32, stride=32, padding=0,
                               bias=True)
        )
        # shallow feature extraction
        self.getFactor = torch.nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        self.RFAs = nn.ModuleList([RFA()])
        for _ in range(self.num_blocks - 1):
            self.RFAs.append(RFA())

        self.gff = nn.Sequential(
            nn.Conv2d(64 * self.num_blocks, 64, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        sample_number = self.sample(x)
        initial_number = self.initialization(sample_number)
        factor = self.getFactor(initial_number)
        x = factor

        local_features = []
        for i in range(self.num_blocks):
            x = self.RFAs[i](x)
            local_features.append(x)

        out = self.gff(torch.cat(local_features, 1)) + initial_number
        return out

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


if __name__ == '__main__':
    import torch
    from thop import profile

    img = torch.randn(1, 1, 256, 256)
    net = RFAN(1, 102, num_blocks=1)
    flops, params = profile(net, (img,))
    print('flops: ', flops, 'params: ', params)
