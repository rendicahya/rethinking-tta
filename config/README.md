# Config naming scheme

```
resnet-50-cifar-10-c-{tta_method}-{budget}-{variant}.py
```

- `{tta_method}` — the TTA method under test: `baseline` (no TTA), `flip`, `multicrop`, `tent`, ...
- `{budget}` — the compute budget, expressed as the number of forward passes per
  sample, e.g. `n1`, `n2`, `n10`. This is the main knob for cost-effectiveness
  comparisons (accuracy gain vs. extra forward passes).
- `{variant}` — a free-form tag distinguishing configs that share the same method
  and budget but differ in other hyperparameters, e.g. `default`, `lr1e-3`,
  `crop28`.

Examples:

| Config file | TTA method | Budget | Notes |
|---|---|---|---|
| `resnet-50-cifar-10-c-baseline-n1-default.py` | none | 1 forward pass | reference point |
| `resnet-50-cifar-10-c-flip-n2-default.py` | horizontal flip averaging | 2 forward passes | |
| `resnet-50-cifar-10-c-multicrop-n10-default.py` | random-crop + flip averaging | 10 forward passes | |
| `resnet-50-cifar-10-c-tent-n1-lr1e-3.py` | entropy minimization (TENT) | 1 adaptation step + 1 forward pass | lr=1e-3 |

Each config is a plain Python file defining top-level dicts:

```python
model = dict(type="resnet50", num_classes=10, checkpoint="checkpoints/resnet50-cifar10.pt")
data = dict(root="data", batch_size=256, corruptions="all", severities=[1, 2, 3, 4, 5])
tta = dict(type="flip")
output = dict(dir="results", tag="resnet-50-cifar-10-c-flip-n2-default")
```

Run with:

```
uv run test.py config/resnet-50-cifar-10-c-flip-n2-default.py
```
