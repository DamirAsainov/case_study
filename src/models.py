"""Model definitions: a compact CNN trained from scratch and a pretrained ResNet-18."""
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SimpleCNN(nn.Module):
    """4 conv blocks (32-64-128-256) + global average pooling + dropout + linear head."""

    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(conv_block(3, 32), conv_block(32, 64),
                                      conv_block(64, 128), conv_block(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
                                  nn.Linear(256, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


class ResNet18Transfer(nn.Module):
    """ImageNet-pretrained ResNet-18. Inputs are upsampled to `input_size` because
    ResNet downsamples 32x and 64px inputs would leave only a 2x2 feature map."""

    def __init__(self, n_classes, input_size=128):
        super().__init__()
        self.input_size = input_size
        self.net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.net.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.net.fc.in_features, n_classes))

    def forward(self, x):
        if x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        return self.net(x)

    @property
    def gradcam_layer(self):
        return self.net.layer4


def build_model(name, n_classes):
    if name == "cnn":
        return SimpleCNN(n_classes)
    if name == "resnet18":
        return ResNet18Transfer(n_classes)
    raise ValueError(name)
