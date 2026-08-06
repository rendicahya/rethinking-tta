# Config naming scheme

```
{model}-cifar-10-c-{tta_method}-{budget}-{variant}.py
```

- `{model}` — the backbone: `resnet-50` or `wideresnet-28-10` (see main README for
  which pretrained checkpoint each one loads).
- `{tta_method}` — the TTA method under test: `baseline` (no TTA), `flip`, `multicrop`, `tent`, ...
- `{budget}` — the compute budget, expressed as the number of forward passes per
  sample, e.g. `n1`, `n2`, `n10`. This is the main knob for cost-effectiveness
  comparisons (accuracy gain vs. extra forward passes).
- `{variant}` — a free-form tag distinguishing configs that share the same method
  and budget but differ in other hyperparameters, e.g. `default`, `lr1e-3`,
  `crop28`.

Examples:

| Config file | Model | TTA method | Budget | Notes |
|---|---|---|---|---|
| `resnet-50-cifar-10-c-baseline-n1-default.py` | ResNet-50 | none | 1 forward pass | reference point |
| `resnet-50-cifar-10-c-flip-n2-default.py` | ResNet-50 | horizontal flip averaging | 2 forward passes | |
| `resnet-50-cifar-10-c-multicrop-n10-default.py` | ResNet-50 | random-crop + flip averaging | 10 forward passes | |
| `resnet-50-cifar-10-c-tent-n1-lr1e-3.py` | ResNet-50 | entropy minimization (TENT) | 1 adaptation step + 1 forward pass | lr=1e-3 |
| `wideresnet-28-10-cifar-10-c-baseline-n1-default.py` | WRN-28-10 | none | 1 forward pass | reference point |
| `wideresnet-28-10-cifar-10-c-flip-n2-default.py` | WRN-28-10 | horizontal flip averaging | 2 forward passes | |
| `wideresnet-28-10-cifar-10-c-multicrop-n10-default.py` | WRN-28-10 | random-crop + flip averaging | 10 forward passes | |
| `wideresnet-28-10-cifar-10-c-tent-n1-lr1e-3.py` | WRN-28-10 | entropy minimization (TENT) | 1 adaptation step + 1 forward pass | lr=1e-3 |

Each config is a plain Python file defining top-level dicts:

```python
model = dict(type="resnet50", num_classes=10, checkpoint="checkpoints/resnet-50-cifar-10.pt")
data = dict(root="data", batch_size=256, corruptions="all", severities=[1, 2, 3, 4, 5])
tta = dict(type="flip")
output = dict(dir="results", tag="resnet-50-cifar-10-c-flip-n2-default")
```

`model.type` must be one of `model.py`'s registered builders: `resnet50` or
`wideresnet2810`. The two backbones expect different input normalization, so
`data` also carries optional `mean`/`std` overrides:

- ResNet-50 configs omit `mean`/`std` (defaults to CIFAR-10 stats matching
  `checkpoints/resnet-50-cifar-10.pt`'s training normalization).
- WideResNet-28-10 configs set `data.mean=(0,0,0), data.std=(1,1,1)` — a no-op
  normalization — because RobustBench's "Standard" checkpoint was trained on
  raw `[0, 1]` pixels.

Run with:

```
uv run test.py config/resnet-50-cifar-10-c-flip-n2-default.py
uv run test.py config/wideresnet-28-10-cifar-10-c-flip-n2-default.py
```
