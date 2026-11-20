import torch
import torch.nn as nn
from torchvision import models

# 1. طبقة Attention مكتوبة يدوياً (Custom Component)
class CustomSpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        attn = self.softmax(torch.bmm(q, k))
        v = self.value(x).view(B, -1, H * W)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x

# 2. Classic Model (CNN)
class ClassicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(32 * 4 * 4, 2)
        )
    def forward(self, x):
        return self.net(x)

# 3. Modern Model (ResNet-18 Pretrained / Scratch)
class ModernResNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)
    def forward(self, x):
        return self.backbone(x)

# 4. Advanced Model (ResNet-18 + Custom Attention)
class AdvancedAttentionNet(nn.Module):
    def __init__(self, use_attention=True):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(base.children())[:-2])
        self.use_attention = use_attention
        if use_attention:
            self.attention = CustomSpatialAttention(512)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        feat = self.encoder(x)
        if self.use_attention:
            feat = self.attention(feat)
        return self.fc(self.pool(feat).flatten(1))