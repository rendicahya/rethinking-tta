# Random-crop + flip TTA averaged over 10 views (10 forward passes).

model = dict(
    type="resnet50",
    num_classes=10,
    pretrained_backbone=False,
    checkpoint="checkpoints/resnet50-cifar10.pt",
)

data = dict(
    root="data",
    batch_size=256,
    num_workers=4,
    corruptions="all",
    severities=[1, 2, 3, 4, 5],
)

tta = dict(
    type="multicrop",
    n_views=10,
    crop_size=28,
)

output = dict(
    dir="results",
    tag="resnet-50-cifar-10-c-multicrop-n10-default",
)
