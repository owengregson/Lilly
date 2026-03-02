# Lilly — ML Typing Behavior Model

## Project Overview

Lilly is a machine learning pipeline that models realistic human typing behavior — including timing (inter-key intervals), errors, and corrections — trained on the [Aalto 136M Keystrokes](https://userinterfaces.aalto.fi/136Mkeystrokes/) dataset. The end goal is a model that can be exported to TensorFlow.js and run in a Chrome extension to simulate human-like typing in real time.

Two model versions:
- **V1 (trained):** LSTM-based next-keystroke predictor with three output heads
- **V2 (in progress):** Transformer encoder-decoder for phrase-level sequence generation

## Package Structure

```
Lilly/
├── lilly/                          # Main package
│   ├── core/                       # Shared utilities
│   │   ├── config.py               #   All paths, constants, V1/V2 model & training configs
│   │   ├── encoding.py             #   char_to_id, id_to_char, wpm_to_bucket
│   │   ├── keyboard.py             #   QWERTY layout, key distances, finger map
│   │   └── losses.py               #   LogNormalNLL, LogNormalNLLSeq, MaskedSparseCE
│   ├── data/                       # Data pipeline
│   │   ├── download.py             #   Aalto dataset downloader
│   │   ├── preprocess.py           #   Raw keystroke parsing & alignment
│   │   ├── features.py             #   V1 dense feature extraction (14 features)
│   │   ├── segment.py              #   V2 pause-based segmentation
│   │   └── pipeline.py             #   tf.data builders (build_v1_datasets, build_v2_datasets)
│   ├── models/                     # Model definitions
│   │   ├── lstm.py                 #   V1 LSTM (build_model, compile_model)
│   │   └── transformer.py          #   V2 Transformer (TypingTransformer, compute_loss)
│   ├── training/                   # Training logic
│   │   ├── callbacks.py            #   Keras callbacks (make_callbacks)
│   │   ├── trainer_v1.py           #   V1 Keras .fit() trainer
│   │   └── trainer_v2.py           #   V2 custom GradientTape trainer
│   ├── inference/                  # Generation & preview
│   │   ├── sampling.py             #   sample_lognormal, weighted_sample, weighted_sample_logits
│   │   ├── context.py              #   ContextWindow (V1 sliding window, 14 dense features)
│   │   ├── generator.py            #   generate_v1, generate_v2_segment, generate_v2_full
│   │   └── preview.py              #   LiveRenderer, live_generate_v1, play_v2_keystrokes
│   ├── evaluation/                 # Metrics & visualization
│   │   └── evaluator.py            #   evaluate_v1, teacher_forced_metrics, reconstruction_metrics
│   └── export/                     # Model export
│       └── converter.py            #   export_model (Keras → TF.js pipeline)
├── scripts/                        # CLI entry points (thin wrappers)
│   ├── download.py                 ├── train.py (--version v1|v2)
│   ├── preprocess.py               ├── evaluate.py (--version v1|v2)
│   ├── extract_features.py         ├── generate.py (--version v1|v2)
│   ├── live_preview.py             └── export.py (--version v1|v2)
├── tests/                          # Test suite
│   ├── test_encoding.py            ├── test_keyboard.py
│   └── test_segment.py
├── configs/                        # YAML config files
│   ├── v1.yaml                     └── v2.yaml
├── pyproject.toml                  # Package definition & dependencies
├── Makefile                        # Common commands (make train-v1, make test, etc.)
└── CLAUDE.md                       # This file
```

## Data Pipeline

```
scripts/download.py → scripts/preprocess.py → scripts/extract_features.py → scripts/train.py
```

1. **lilly.data.download** — Downloads and extracts the Aalto 136M Keystrokes zip (~15GB) to `data/raw/`
2. **lilly.data.preprocess** — Parses raw keystroke files, replays sessions to classify keystrokes as correct/error/backspace. Outputs Parquet to `data/processed/`. Uses `ProcessPoolExecutor`.
3. **lilly.data.features** — Computes 14 dense features, extracts sliding windows (SEQ_LEN=32), saves `.npz` to `data/tfrecords/`
4. **lilly.data.pipeline** — `build_v1_datasets()` and `build_v2_datasets()` create `tf.data.Dataset` pipelines
5. **lilly.training.trainer_v1/v2** — Training with callbacks, checkpointing, early stopping

## V1 Model Architecture (lilly.models.lstm)

LSTM predicting the **next keystroke** from 32 previous keystrokes.

### Inputs
- `typed_chars` — (batch, 32) int32 — character IDs of what was typed
- `target_chars` — (batch, 32) int32 — character IDs of what should have been typed
- `actions` — (batch, 32) int32 — action labels (0=correct, 1=error, 2=backspace)
- `dense_features` — (batch, 32, 14) float32 — engineered features
- `wpm_bucket` — (batch, 1) int32 — WPM persona bucket (10 buckets)

### Three Output Heads
1. **timing** — Dense(2) → `[mu, log_sigma]` for LogNormal IKI distribution
2. **action** — Dense(3, softmax) → correct/error/backspace
3. **error_char** — Dense(97, softmax) → which wrong key was typed

### Loss Functions (lilly.core.losses)
- **Timing:** `LogNormalNLL` — negative log-likelihood of LogNormal in log-space
- **Action:** Sparse categorical cross-entropy (weight 2.0)
- **Error char:** Sparse categorical cross-entropy (weight 0.5), masked to action=error

## V2 Model Architecture (lilly.models.transformer)

Transformer encoder-decoder generating entire keystroke segments.

- **Encoder:** Target text → char embedding (32) + sinusoidal PE + WPM conditioning → 2 encoder layers (d_model=64, nhead=4)
- **Decoder:** Autoregressive (char_id, delay) → 2 decoder layers → char_logits (99 classes) + delay params (mu, log_sigma)
- **Segmentation:** Pause-based (300ms threshold) via `lilly.data.segment`
- **Loss:** `compute_loss()` = masked char CE + masked timing NLL (`LogNormalNLLSeq`)

## Character Encoding (lilly.core.encoding)

| Range | Meaning |
|-------|---------|
| 0 | PAD |
| 1–95 | Printable ASCII (space 0x20 .. tilde 0x7E), `ord(c) - 31` |
| 96 | BACKSPACE |
| 97 | END (V2 only) |
| 98 | START (V2 only) |

## Key Config Classes (lilly.core.config)

- `V1ModelConfig` / `V1TrainConfig` — V1 LSTM configuration
- `V2ModelConfig` / `V2TrainConfig` — V2 Transformer configuration
- Path constants: `PROJECT_ROOT`, `DATA_DIR`, `RAW_DIR`, `PROCESSED_DIR`, `TFRECORD_DIR`, `V2_SEGMENT_DIR`, `MODEL_DIR`, `V2_MODEL_DIR`, `EXPORT_DIR`, `V2_EXPORT_DIR`

## Commands

```bash
# Install
pip install -e ".[dev]"

# Full V1 pipeline
python scripts/download.py
python scripts/preprocess.py --workers 8
python scripts/extract_features.py
python scripts/train.py --version v1 --epochs 30

# V2 pipeline
python scripts/train.py --version v2

# Evaluate
python scripts/evaluate.py models/run_XXX/final_model.keras
python scripts/evaluate.py --version v2 models/v2/run_XXX/best_model.keras

# Generate / Preview
python scripts/generate.py models/run_XXX/final_model.keras "The quick brown fox"
python scripts/live_preview.py --wpm 80 "Hello, world!"

# Export to TF.js
python scripts/export.py models/run_XXX/best_model.keras --quantize uint8

# Test & Lint
make test
make lint
```

## Known Issues

1. **V1 action prediction imbalance:** Heavily biased toward "correct" (~90%+ of keystrokes). Error/backspace predictions unreliable.
2. **V1 generate.py hardcodes correction:** Forces immediate backspace+retype after errors. Live preview lets model decide.
3. **V1 timing MAE:** ~117ms. Could improve with V2's richer context.
4. **Dataset scale:** Processing full 136M keystrokes requires significant disk space and time.

## Dependencies

Defined in `pyproject.toml`. Core: tensorflow, numpy, pandas, pyarrow, tqdm, requests.
Optional: tensorflowjs (export), matplotlib + scikit-learn (eval), pytest + ruff (dev).
