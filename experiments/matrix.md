# Experiment Matrix

This matrix tracks the intended CIFAR-100 binary classification experiments for the local code and
the standalone notebooks.

## Task Axis

| Task type | Examples | Notes |
|---|---|---|
| Fine class vs. rest | `cow` vs. not `cow`, `apple` vs. not `apple` | Report class balance carefully |
| Superclass vs. rest | `aquatic mammals` vs. rest, `vehicles 1` vs. rest | Usually less imbalanced than one fine class |

## Model Axis

| Family | Models | Training mode |
|---|---|---|
| Sequential | RNN, LSTM, Bi-LSTM | From scratch, optional row masking |
| Attention | Small ViT | From scratch |
| Transfer learning | MobileNetV3, EfficientNet, ResNet | Frozen feature extraction, partial fine-tuning |

## Ablation Axis

| Axis | Values |
|---|---|
| Data augmentation | off, on |
| Transfer backbone state | frozen, partially unfrozen |
| Sequence masking | off, on for sequential models |
| Metrics | accuracy, precision, recall, F1, ROC-AUC, confusion matrix |

## Run Policy

Start with one fine-class task and one superclass task. Use smoke runs first, then promote only the
most informative configurations to longer training runs.
