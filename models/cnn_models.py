import torch.nn as nn
import torch.nn.functional as F
from models.custom_layers import CustomConv2d, SimpleSpatialAttention, CustomActivation, CustomMaxAbsPool2d

class SimpleCNN(nn.Module):
    """Простая CNN с 3 сверточными слоями для MNIST."""
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ResNetLike(nn.Module):
    """Простая CNN с Residual Block для MNIST."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, 3, padding=1)
        self.res1 = ResidualBlock(32, 64, downsample=True)
        self.res2 = ResidualBlock(64, 128, downsample=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        """Выполняет прямое распространение входных данных через сеть с residual блоками."""
        x = F.relu(self.conv(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ResidualBlock(nn.Module):
    """Residual блок для ResNet."""
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0, use_bn=True):
        super().__init__()
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
            )

    def forward(self, x):
        """Выполняет прямое распространение через residual блок."""
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)

class ResNetCIFAR(nn.Module):
    """ResNet с 3 residual блоками для CIFAR-10."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.layer1 = ResidualBlock(32, 64, stride=2)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 128, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        """Выполняет прямое распространение входных данных через сеть с residual блоками."""
        x = F.relu(self.bn(self.conv(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ResNetCIFARRegularized(nn.Module):
    """ResNet с Dropout и BatchNorm для CIFAR-10."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.layer1 = ResidualBlock(32, 64, stride=2, dropout=0.3, use_bn=True)
        self.layer2 = ResidualBlock(64, 128, stride=2, dropout=0.3, use_bn=True)
        self.layer3 = ResidualBlock(128, 128, stride=2, dropout=0.3, use_bn=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        """Выполняет прямое распространение входных данных через сеть с регуляризацией."""
        x = F.relu(self.bn(self.conv(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ShallowCNN(nn.Module):
    """Неглубокая CNN: 2 conv слоя."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class MediumCNN(nn.Module):
    """Средняя CNN: 4 conv слоя."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class DeepCNN(nn.Module):
    """Глубокая CNN: 6 conv слоев."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
class ResidualCNN(nn.Module):
    """CNN с Residual связями (4 residual блока)."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3, padding=1)
        self.res1 = ResidualBlock(32, 64)
        self.res2 = ResidualBlock(64, 128)
        self.res3 = ResidualBlock(128, 128)
        self.res4 = ResidualBlock(128, 128)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
class CNNKernel3x3(nn.Module):
    """CNN с 3x3 ядрами, число фильтров подобрано для равенства параметров."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class CNNKernel5x5(nn.Module):
    """CNN с 5x5 ядрами, число фильтров уменьшено для равенства параметров."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 19, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(19, 38, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(38, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class CNNKernel7x7(nn.Module):
    """CNN с 7x7 ядрами, число фильтров ещё меньше для равенства параметров."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 13, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(13, 26, kernel_size=7, padding=3)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(26, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class CNNKernelCombo(nn.Module):
    """CNN с комбинацией 1x1 и 3x3 ядер, число фильтров подобрано для равенства параметров."""
    def __init__(self, in_channels=1):  # по умолчанию MNIST
        super().__init__()
        self.conv1 = CustomConv2d(in_channels, 16, 3, padding=1, scale=1.5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class CNNWithCustomLayers(nn.Module):
    """CNN с кастомными слоями для сравнения."""
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv1 = CustomConv2d(in_channels, 16, 3, padding=1, scale=1.5)
        self.act1 = CustomActivation()
        self.attn = SimpleSpatialAttention(16)
        self.pool = CustomMaxAbsPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.attn(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
class BasicResidualBlock(nn.Module):
    """Базовый Residual блок: 2 conv слоя."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class BottleneckResidualBlock(nn.Module):
    """Bottleneck Residual блок: 1x1 -> 3x3 -> 1x1."""
    def __init__(self, in_channels, out_channels, stride=1, bottleneck_ratio=4):
        super().__init__()
        bottleneck_channels = out_channels // bottleneck_ratio
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)

class WideResidualBlock(nn.Module):
    """Wide Residual блок: увеличенное число каналов."""
    def __init__(self, in_channels, out_channels, stride=1, widen_factor=3):
        super().__init__()
        mid_channels = out_channels * widen_factor
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class BasicResNet(nn.Module):
    """ResNet с базовыми блоками."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.layer1 = BasicResidualBlock(16, 32, stride=2)
        self.layer2 = BasicResidualBlock(32, 64, stride=2)
        self.layer3 = BasicResidualBlock(64, 128, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class BottleneckResNet(nn.Module):
    """ResNet с bottleneck блоками."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.layer1 = BottleneckResidualBlock(16, 32, stride=2)
        self.layer2 = BottleneckResidualBlock(32, 64, stride=2)
        self.layer3 = BottleneckResidualBlock(64, 128, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class WideResNet(nn.Module):
    """ResNet с wide residual блоками."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.layer1 = WideResidualBlock(16, 32, stride=2, widen_factor=3)
        self.layer2 = WideResidualBlock(32, 64, stride=2, widen_factor=3)
        self.layer3 = WideResidualBlock(64, 128, stride=2, widen_factor=3)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)