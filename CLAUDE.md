# CLAUDE.md - CIFAR-100 Binary Deep Learning Benchmark

> Persistent context for coding agents working in this repo. Read this before writing or changing
> code. The design decisions below are locked unless the user explicitly revises them.

## 1. What This Project Is

A deep-learning evaluation framework for the MSc in Artificial Intelligence project at NCSR
Demokritos and the University of Piraeus. The project benchmarks different TensorFlow/Keras model
families on CIFAR-100 binary classification tasks derived from fine labels and coarse
superclasses.

The central comparison is still architectural:

- sequential processing with RNN, LSTM, and Bi-LSTM models;
- attention-based modeling with a small from-scratch ViT;
- transfer learning and fine-tuning with MobileNetV3, EfficientNet, and ResNet.

Images are 32x32. Sequential models consume images row-by-row as sequences; image models consume
standard image tensors.

## 2. Deliverables

This project has two parallel implementations:

1. **Local source-code implementation.** Reusable Python modules under `data/`, `models/`,
   `training/`, `evaluation/`, and `experiments/`. This implementation should be config-driven,
   testable, and suitable for local smoke tests and longer runs.
2. **Standalone Colab notebooks.** Notebooks under `notebooks/` should run independently in Colab.
   They must not import local source-code modules from this repository. Each notebook should
   contain the dataset loading, preprocessing, model definitions, training, evaluation, and plots
   needed for that workflow.

Keep the two deliverables aligned conceptually, but do not couple the notebooks to the local code.

## 3. Stack & Environment

- **Framework: TensorFlow / Keras.** Keep the project in TensorFlow/Keras.
- **Python: 3.11** for local development.
- **Hardware target:** single Colab T4 for notebook runs and final student-facing execution.
- **Local Mac support:** `requirements-macos.txt` may include Apple Silicon extras such as
  `tensorflow-metal`, but the core `requirements.txt` should stay portable and Colab-friendly.
- **No heavy downloads at import time.** Dataset loading must happen inside explicit functions,
  scripts, or notebooks.

## 4. Hard Constraints

1. **Primary dataset is CIFAR-100.** Keep new data work centered on CIFAR-100 fine labels and
   coarse superclasses.
2. **Primary task style is binary classification.** Use CIFAR-100 fine labels and coarse
   superclasses to define tasks such as `cow` vs. `not cow` or `aquatic mammals` vs. all other
   classes.
3. **Keep sequence length fixed at T = 32.** A 32x32x3 image becomes `(T=32, features=96)`.
4. **Use small, reproducible runs first.** Prefer subset/smoke runs before full training.
5. **Avoid OOM on Colab T4.** Use small batches, `tf.data`, resizing only for transfer-learning
   backbones, and conservative fine-tuning.
6. **Notebooks must be self-contained.** They should not fetch or import code from this local
   package.

## 5. Dataset and Task Design

- Load CIFAR-100 through TensorFlow/Keras or another explicit loader inside the data layer or
  notebook.
- Preserve both CIFAR-100 label levels:
  - **fine labels:** 100 object classes, useful for target-vs-rest tasks;
  - **coarse labels:** 20 superclasses, useful for group-vs-rest tasks.
- Binary task examples:
  - fine class: `cow` vs. `not cow`;
  - fine class: `apple` vs. `not apple`;
  - superclass: `aquatic mammals` vs. all other classes;
  - superclass: `vehicles 1` or `large carnivores` vs. all other classes.
- Keep task definitions explicit in config/notebook metadata: label names, label ids, positive
  class definition, negative class definition, train/validation/test split, and seed.
- For imbalanced target-vs-rest tasks, report class balance and prefer class weights, balanced
  sampling, or clear metric interpretation.

## 6. Data Views and Preprocessing

- **Image view:** `(batch, 32, 32, 3)` float tensors for CNN, ViT, MobileNetV3, EfficientNet, and
  ResNet branches.
- **Sequence view:** `(batch, 32, 96)` for RNN, LSTM, and Bi-LSTM branches. Each image row becomes
  one timestep.
- **Transfer-learning resize:** pretrained CNN backbones may resize images from 32x32 to a larger
  input size such as 96x96 or 224x224 inside the model/pipeline. This does not change the sequence
  model contract.
- **Augmentation:** use controlled image augmentation for CNN/transfer-learning experiments and
  document whether augmentation is active for each run.
- **Masking experiment:** optional for sequential models. If row masking is used, avoid sentinel
  collisions with valid normalized pixels and document the chosen `mask_value`.

## 7. Models To Implement

- `models/sequential.py`
  - `build_rnn`
  - `build_lstm`
  - `build_bilstm`
  - optional Keras `Masking` before recurrent layers
- `models/vit.py`
  - from-scratch small ViT with patch embedding, positional embeddings, attention blocks, and a
    binary head
- `models/transfer.py`
  - MobileNetV3 feature extraction and fine-tuning
  - EfficientNet feature extraction and fine-tuning
  - ResNet feature extraction and fine-tuning

Transfer-learning stages should include:

- frozen-backbone feature extraction;
- partial unfreezing of top layers;
- optional full fine-tuning only when it fits the compute budget;
- clear comparison of frozen vs. unfrozen performance.

## 8. Experiment Matrix

Keep local experiments and notebooks aligned to this matrix where possible:

| Axis | Values |
|---|---|
| Task definition | fine class vs. rest, superclass vs. rest |
| Architecture | RNN, LSTM, Bi-LSTM, small ViT, MobileNetV3, EfficientNet, ResNet |
| Training mode | from scratch, frozen feature extraction, partial fine-tuning |
| Data augmentation | off, on |
| Sequence masking | off, on for sequential models only |
| Metrics | accuracy, precision, recall, F1, ROC-AUC, confusion matrix |

Promote only the most informative runs to full training. Keep exploratory runs small.

## 9. Notebook Plan

Notebooks should be standalone Colab workflows. Suggested sequence:

1. CIFAR-100 data exploration and binary task construction.
2. Baseline CNN for selected binary tasks.
3. Data augmentation experiments.
4. Transfer-learning feature extraction with frozen backbones.
5. Fine-tuning with partially unfrozen EfficientNet/ResNet/MobileNetV3.
6. Row-as-timestep sequence models.
7. Small ViT and architecture comparison summary.

Notebook outputs should include plots, training curves, metric tables, and short interpretation
cells suitable for the course report.

## 10. Gotchas

- **Target-vs-rest imbalance.** A single CIFAR-100 class has far fewer positives than negatives.
  Accuracy alone can be misleading; always report precision, recall, F1, ROC-AUC, and class counts.
- **CIFAR-100 label mapping.** Keep fine and coarse label names/ids explicit. Do not rely on magic
  integers without a mapping table.
- **ReLU in vanilla RNNs can explode.** Use gradient clipping when recurrent activation is `relu`.
- **Transfer backbones expect larger images.** Resize inside the image branch and keep the
  sequential branch at T=32.
- **Notebook independence.** If a fix is made in local source code, mirror the relevant logic in
  notebooks manually instead of importing the source module.
- **Determinism.** Set and log seeds for every local run and notebook.

## 11. Commands

```bash
# Setup
pip install -r requirements.txt

# macOS Apple Silicon optional setup
pip install -r requirements-macos.txt

# Single local training run, config-driven
python -m training.train --config configs/lstm.yaml

# Local ablation sweep
python -m experiments.run_ablations
```

## 12. Conventions

- Config-driven local code; avoid hard-coded experiment choices in training scripts.
- Every local run writes to `results/<run_name>/` with config snapshot, metrics JSON, and training
  curves.
- Prefer `tf.data` pipelines over ad hoc full-memory processing for full runs.
- Type hints and short docstrings on public functions.
- Keep notebooks readable and course-report friendly: headings, concise markdown, reproducible
  cells, and clear plots.
