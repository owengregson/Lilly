# Lilly — ML Typing Behavior Model

## Project Overview

Lilly is a machine learning pipeline that models realistic human typing behavior — including timing (inter-key intervals), errors, and corrections — trained on the [Aalto 136M Keystrokes](https://userinterfaces.aalto.fi/136Mkeystrokes/) dataset. The end goal is a model that can be exported to TensorFlow.js and run in a Chrome extension to simulate human-like typing in real time.

**V3 Architecture:** Unified encoder-decoder transformer with action-gated decoder, per-action mixture density network (MDN) timing heads, FiLM style conditioning, and QWERTY-biased error character prediction.

## Package Structure

```
Lilly/
├── lilly/                          # Main package
│   ├── core/                       # Shared utilities
│   │   ├── config.py               #   Paths, constants, V3ModelConfig, V3TrainConfig
│   │   ├── encoding.py             #   char_to_id, id_to_char, wpm_to_bucket
│   │   └── keyboard.py             #   QWERTY layout, key distances, finger map
│   ├── data/                       # Data pipeline
│   │   ├── download.py             #   Aalto dataset downloader
│   │   ├── preprocess.py           #   Raw keystroke parsing & alignment
│   │   ├── segment.py              #   Inference-time text segmentation
│   │   ├── segment_v3.py           #   V3 training segment extraction
│   │   ├── style.py                #   Style vector computation & normalization
│   │   └── pipeline.py             #   tf.data builder (build_v3_datasets)
│   ├── models/                     # Model definitions
│   │   ├── components.py           #   FiLMModulation, MDNHead, ActionGate, ErrorCharHead
│   │   └── typing_model.py         #   TypingTransformerV3, EncoderLayer, DecoderLayer
│   ├── training/                   # Training logic
│   │   ├── losses.py               #   FocalLoss, mdn_mixture_nll, compute_v3_loss
│   │   ├── schedule.py             #   WarmupCosineDecay LR schedule
│   │   └── trainer.py              #   V3 custom GradientTape training loop
│   ├── inference/                  # Generation & preview
│   │   ├── sampling.py             #   sample_mdn, weighted_sample, weighted_sample_logits
│   │   └── generator.py            #   generate_v3_segment, generate_v3_full
│   ├── evaluation/                 # Metrics & visualization
│   │   ├── metrics.py              #   Tier 1 point metrics (accuracy, F1, MAE, NLL)
│   │   ├── distributional.py       #   Tier 2 distributional metrics (Wasserstein, KS)
│   │   ├── realism.py              #   Tier 3 realism metrics (discriminator, style)
│   │   └── visualization.py        #   Plotting (IKI, bursts, confusion, MDN, style)
│   └── export/                     # Model export
│       └── converter.py            #   export_model, get_v3_custom_objects (Keras → TF.js)
├── scripts/                        # CLI entry points (thin wrappers)
│   ├── download.py                 ├── train.py
│   ├── preprocess.py               ├── evaluate.py (--tier 1|2|3)
│   ├── segment_v3.py               ├── generate.py (--wpm, --error-rate)
│   ├── live_preview.py             └── export.py
├── tests/                          # Test suite
│   ├── test_encoding.py            ├── test_keyboard.py
│   ├── test_segment.py             ├── test_components.py
│   ├── test_losses.py              ├── test_typing_model.py
│   ├── test_style.py               └── test_generator.py
├── pyproject.toml                  # Package definition & dependencies
├── Makefile                        # Common commands
└── CLAUDE.md                       # This file
```

## Data Pipeline

```
scripts/download.py → scripts/preprocess.py → scripts/segment_v3.py → scripts/train.py
```

1. **lilly.data.download** — Downloads and extracts the Aalto 136M Keystrokes zip (~15GB) to `data/raw/`
2. **lilly.data.preprocess** — Parses raw keystroke files, replays sessions to classify keystrokes as correct/error/backspace. Outputs Parquet to `data/processed/`. Uses `ProcessPoolExecutor`.
3. **lilly.data.segment_v3** — Extracts V3 training segments with style vectors, context windows, and teacher-forced decoder I/O. Outputs `.npz` to `data/v3_segments/`.
4. **lilly.data.pipeline** — `build_v3_datasets()` creates `tf.data.Dataset` pipeline with train/val/test splits.
5. **lilly.training.trainer** — Custom GradientTape training with AdamW, WarmupCosineDecay, gradient clipping, checkpointing, early stopping.

## V3 Model Architecture (lilly.models.typing_model)

Unified encoder-decoder transformer with action-gated generation.

### Encoder
- Target text → char embedding (48) + sinusoidal PE + FiLM style conditioning
- 4 encoder layers (d_model=128, nhead=8, dim_feedforward=256)

### Decoder
- Autoregressive input: char_id + delay + action embeddings + sinusoidal PE
- Previous context (tail of last segment) prepended
- 4 decoder layers with FiLM conditioning, causal self-attention, cross-attention to encoder

### Output Heads
1. **ActionGate** — Dense(64, relu) → Dense(3, softmax) → correct/error/backspace
2. **3 MDN Timing Heads** — Per-action (correct, error, backspace), each 8-component LogNormal mixture → (pi, mu, log_sigma)
3. **ErrorCharHead** — Dense(97) with learnable QWERTY distance bias → error character prediction
4. **PositionHead** — Dense(max_encoder_len) → predicted position in target text

### Style Conditioning
- 16-dimensional style vector computed from session statistics
- FiLM (Feature-wise Linear Modulation): γ * x + β at every encoder/decoder layer
- Style features: mean_iki_log, std_iki_log, error_rate, burst_length, correction_latency, etc.

### Loss Functions (lilly.training.losses)
- **Action:** Focal loss with per-class alpha (0.25, 0.5, 0.5), gamma=2.0
- **Timing:** MDN mixture NLL (LogNormal), masked per action type
- **Error char:** Sparse CE masked to action=error
- **Position:** Sparse CE for target position prediction

## Character Encoding (lilly.core.encoding)

| Range | Meaning |
|-------|---------|
| 0 | PAD |
| 1–95 | Printable ASCII (space 0x20 .. tilde 0x7E), `ord(c) - 31` |
| 96 | BACKSPACE |
| 97 | END |
| 98 | START |

## Key Config Classes (lilly.core.config)

- `V3ModelConfig` — d_model=128, nhead=8, 4 layers, char_embed=48, action_embed=16, delay_embed=16, style_dim=16, mdn_components=8, max_decoder_len=80, max_encoder_len=32, context_tail_len=16
- `V3TrainConfig` — batch_size=128, epochs=50, lr=3e-4, warmup_steps=2000, focal_gamma=2.0, focal_alpha=(0.25, 0.5, 0.5)
- Path constants: `PROJECT_ROOT`, `DATA_DIR`, `RAW_DIR`, `PROCESSED_DIR`, `V3_SEGMENT_DIR`, `V3_MODEL_DIR`, `V3_EXPORT_DIR`

## Commands

```bash
# Install
pip install -e ".[dev]"

# Full pipeline
python scripts/download.py
python scripts/preprocess.py --workers 8
python scripts/segment_v3.py --workers 8
python scripts/train.py --epochs 50

# Evaluate (tiered)
python scripts/evaluate.py models/v3/run_XXX/best_model.keras --tier 1
python scripts/evaluate.py models/v3/run_XXX/best_model.keras --tier 2 --n-samples 500
python scripts/evaluate.py models/v3/run_XXX/best_model.keras --tier 3

# Generate / Preview
python scripts/generate.py models/v3/run_XXX/best_model.keras "The quick brown fox" --wpm 80
python scripts/live_preview.py --wpm 80 "Hello, world!"

# Export to TF.js
python scripts/export.py models/v3/run_XXX/best_model.keras --quantize uint8

# Test & Lint
make test
make lint
```

## Known Issues

1. **Dataset scale:** Processing full 136M keystrokes requires significant disk space and time.
2. **Style vector normalization:** StyleNormalizer should be fit on training data before inference. Without normalization, style dimensions have different scales.

## Dependencies

Defined in `pyproject.toml`. Core: tensorflow, numpy, pandas, pyarrow, tqdm, requests, scipy.
Optional: tensorflowjs (export), matplotlib + scikit-learn (eval), pytest + ruff (dev).
