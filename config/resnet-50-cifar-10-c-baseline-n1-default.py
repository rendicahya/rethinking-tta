# Reference point: no TTA, single forward pass per sample.

model = dict(
    type="resnet50",
    num_classes=10,
    pretrained_backbone=False,
    checkpoint="checkpoints/resnet50-cifar10.pt",  # produced by train.py (not built yet)
)

data = dict(
    root="data",
    batch_size=256,
    num_workers=4,
    corruptions="all",  # or a list, e.g. ["gaussian_noise", "fog"]
    severities=[1, 2, 3, 4, 5],
)

tta = dict(
    type="baseline",
)

output = dict(
    dir="results",
    tag="resnet-50-cifar-10-c-baseline-n1-default",
)
