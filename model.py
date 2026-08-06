"""ResNet backbones adapted for small (32x32) CIFAR-style inputs."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models


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


MODEL_BUILDERS = {
    "resnet50": build_resnet50,
}


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
        model.load_state_dict(state_dict)
    else:
        print(
            "[build_model] WARNING: no checkpoint given, using randomly initialized "
            "(or ImageNet-pretrained) weights. Train a model first with train.py."
        )

    return model
