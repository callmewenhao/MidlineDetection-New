import torch
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x


class block(nn.Module):
    def __init__(self, in_dim, out_dim, stride=2):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=out_dim,
            out_channels=out_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_dim)
        # 使用 1*1 卷积核进行下采样
        if stride == 2 and in_dim != out_dim:  # 只有 stride = 2 同时 in_dim != out_dim 时
            self.downsampling = nn.Sequential(nn.Conv2d(in_dim, out_dim, 1, stride=stride, padding=0),
                                            nn.BatchNorm2d(out_dim))
        else:
            self.downsampling = Identity()

    def forward(self, x):
        h = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        identity = self.downsampling(h)
        x = x + identity
        return x


class ResNet18(nn.Module):
    def __init__(self, in_dim, num_classes=10):
        super().__init__()
        # head of model
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.MaxPool = nn.MaxPool2d(3, 2, padding=1)
        # body of model
        self.blocks1 = self._blocks(64, 64, 1, 2)
        self.blocks2 = self._blocks(64, 128, 2, 2)
        self.blocks3 = self._blocks(128, 256, 2, 2)
        self.blocks4 = self._blocks(256, 512, 2, 2)
        # classifier of model
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        self.initialize_weights()

    def _blocks(self, in_dim, out_dim, stride, num_blocks=2):
        blocks_list = []
        blocks_list.append(block(in_dim, out_dim, stride))
        for i in range(num_blocks-1):
            blocks_list.append(block(out_dim, out_dim, 1))
        return nn.Sequential(*blocks_list)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.MaxPool(x)

        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)
        # [1, 512, 2, 3]
        x = self.avg(x)
        # [1, 512, 1, 1]
        x = x.flatten(1)  # x = torch.flatten(x, 1)  # 从第1维开始拉平
        x = self.fc(x)
        return x


def main():
    x = torch.randn([1, 1, 60, 94])
    print(f"input shape is {x.shape}")
    model = ResNet18(in_dim=1, num_classes=10)
    out = model(x)
    print(f"output shape is {out.shape}")


if __name__ == "__main__":
    main()
