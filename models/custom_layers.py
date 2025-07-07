import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomBatchNorm(nn.Module):
    """Кастомная реализация слоя BatchNorm2d."""
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        """Выполняет нормализацию входных данных."""
        return self.bn(x)

class CustomConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, scale=1.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = scale
    def forward(self, x):
        return self.conv(x) * self.scale

    def forward(self, x):
        out = self.conv(x)
        # Дополнительная логика: масштабируем выход
        return out * self.scale

class SimpleSpatialAttention(nn.Module):
    """Attention механизм для CNN (простая spatial attention карта)."""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        # Получаем attention map через 1x1 conv, softmax по spatial
        attn_map = torch.sigmoid(self.conv(x))
        return x * attn_map

class CustomActivation(nn.Module):
    """Кастомная функция активации: Swish с learnable beta."""
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

class CustomMaxAbsPool2d(nn.Module):
    """Кастомный pooling: максимальный по абсолютному значению."""
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x):
        # Берём max по abs(x), но возвращаем исходный знак
        abs_x = torch.abs(x)
        max_abs = F.max_pool2d(abs_x, self.kernel_size, self.stride, self.padding)
        mask = (abs_x == F.interpolate(max_abs, size=x.shape[2:], mode='nearest'))
        # Восстанавливаем знак
        out = x * mask.float()
        # Агрегируем по пулу
        out = F.max_pool2d(out, self.kernel_size, self.stride, self.padding)
        return out