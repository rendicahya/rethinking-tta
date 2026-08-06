# Horizontal-flip TTA: averages predictions over the image and its flip (2 forward passes).

model = dict(
    type="wideresnet2810",
    num_classes=10,
    pretrained_backbone=False,
    checkpoint="checkpoints/wideresnet-28-10-cifar-10.pt",
)

data = dict(
    root="data",
    batch_size=256,
    num_workers=4,
    corruptions="all",
    severities=[1, 2, 3, 4, 5],
    mean=(0.0, 0.0, 0.0),
    std=(1.0, 1.0, 1.0),
)

tta = dict(
    type="flip",
)

output = dict(
    dir="results",
    tag="wideresnet-28-10-cifar-10-c-flip-n2-default",
)
