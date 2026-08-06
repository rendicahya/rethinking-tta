# TENT: 1 entropy-minimization adaptation step (lr=1e-3) + 1 prediction forward pass.

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
    type="tent",
    lr=1e-3,
    steps=1,
)

output = dict(
    dir="results",
    tag="wideresnet-28-10-cifar-10-c-tent-n1-lr1e-3",
)
