"""Model backbones for CIFAR/Tiny-ImageNet-scale (32x32-64x64) classification."""

from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

# --- ResNet-50 (adapted from torchvision) ------------------------------------


def build_resnet50(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """Build a ResNet-50 adapted for 32x32 inputs.

    torchvision's ResNet-50 stem (7x7 conv stride 2 + maxpool) is designed for
    224x224 ImageNet inputs and would downsample a 32x32 CIFAR image too
    aggressively. Following common practice, the stem is replaced with a 3x3
    stride-1 conv and the maxpool is dropped.
    """
    weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = tv_models.resnet50(weights=weights)

    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# --- WideResNet-28-10 (CIFAR-native, matches RobustBench's architecture) ----
#
# Attribute names (bn1, relu1, conv1, ..., convShortcut) are kept identical to
# https://github.com/RobustBench/robustbench/blob/master/robustbench/model_zoo/architectures/wide_resnet.py
# on purpose, so that RobustBench's "Standard" WRN-28-10 CIFAR-10 checkpoint
# (see checkpoints/download.sh) loads via a plain strict state_dict match.


class _WRNBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = (
            None
            if self.equalInOut
            else nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        )

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
            out = x
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class _WRNNetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super().__init__()
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(in_planes if i == 0 else out_planes, out_planes, stride if i == 0 else 1, dropRate))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    """Wide Residual Network (Zagoruyko & Komodakis, 2016)."""

    def __init__(self, depth=28, num_classes=10, widen_factor=10, dropRate=0.0, bias_last=True):
        super().__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6
        block = _WRNBasicBlock

        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = _WRNNetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        self.block2 = _WRNNetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        self.block3 = _WRNNetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes, bias=bias_last)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / fan))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


def build_wideresnet2810(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """Build a WideResNet-28-10. `pretrained` is ignored (no ImageNet variant exists);
    use the `checkpoint` config option to load RobustBench's CIFAR-10 "Standard" weights.
    """
    if pretrained:
        print(
            "[build_wideresnet2810] WARNING: pretrained_backbone=True has no effect "
            "for wideresnet2810 (no ImageNet variant). Use `checkpoint` instead."
        )
    return WideResNet(depth=28, num_classes=num_classes, widen_factor=10)


# --- registry -----------------------------------------------------------------

MODEL_BUILDERS = {
    "resnet50": build_resnet50,
    "wideresnet2810": build_wideresnet2810,
}


def _strip_prefix(state_dict: dict, prefix: str) -> dict:
    if not any(k.startswith(prefix) for k in state_dict):
        return state_dict
    return OrderedDict(
        (k[len(prefix):] if k.startswith(prefix) else k, v) for k, v in state_dict.items()
    )


def build_model(cfg) -> nn.Module:
    """Build a model from a config `model = dict(...)` section and optionally load a checkpoint."""
    builder = MODEL_BUILDERS.get(cfg.type)
    if builder is None:
        raise ValueError(f"Unknown model type '{cfg.type}'. Available: {list(MODEL_BUILDERS)}")

    model = builder(
        num_classes=cfg.get("num_classes", 10),
        pretrained=cfg.get("pretrained_backbone", False),
    )

    checkpoint = cfg.get("checkpoint", None)
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
        # Some third-party checkpoints (e.g. trained with nn.DataParallel, or wrapped
        # in a `model.` submodule) prefix every key; strip those if present.
        state_dict = _strip_prefix(state_dict, "module.")
        state_dict = _strip_prefix(state_dict, "model.")
        model.load_state_dict(state_dict)
    else:
        print(
            "[build_model] WARNING: no checkpoint given, using randomly initialized "
            "(or ImageNet-pretrained) weights. Train a model first with train.py."
        )

    return model
