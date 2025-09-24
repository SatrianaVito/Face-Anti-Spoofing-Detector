import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18Binary(nn.Module):
    def __init__(self, pretrained=True, num_classes=2):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, num_classes)

    def forward(self, x):
        return self.backbone(x)