# Reference point: no TTA, single forward pass per sample.

model = dict(
    type="wideresnet2810",
    num_classes=10,
    pretrained_backbone=False,
    checkpoint="checkpoints/wideresnet-28-10-cifar-10.pt",  # bash checkpoints/download.sh
)

data = dict(
    root="data",
    batch_size=256,
    num_workers=4,
    corruptions="all",  # or a list, e.g. ["gaussian_noise", "fog"]
    severities=[1, 2, 3, 4, 5],
    # RobustBench's "Standard" WRN-28-10 checkpoint expects raw [0, 1] pixels
    # (no normalization), unlike the ResNet-50 checkpoint.
    mean=(0.0, 0.0, 0.0),
    std=(1.0, 1.0, 1.0),
)

tta = dict(
    type="baseline",
)

output = dict(
    dir="results",
    tag="wideresnet-28-10-cifar-10-c-baseline-n1-default",
)
