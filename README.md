<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="images/LillyLogoDesignV4SE2Dark.svg">
  <source media="(prefers-color-scheme: dark)" srcset="images/LillyLogoDesignV4SE2Light.svg">
  <img alt="Lilly" src="images/LillyLogoDesignV4SE2Light.svg" width="320">
</picture>

<br><br>

<strong>ML Typing Behavior Model</strong>

<p>
A machine learning pipeline that models realistic human typing behavior — including timing,
errors, and corrections — trained on the
<a href="https://userinterfaces.aalto.fi/136Mkeystrokes/">Aalto 136M Keystrokes</a> dataset.
</p>

<p>
  <a href="#quickstart"><img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="#quickstart"><img src="https://img.shields.io/badge/tensorflow-2.15%2B-orange?logo=tensorflow&logoColor=white" alt="TensorFlow 2.15+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"></a>
</p>

<p>
  <a href="#features">Features</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#usage">Usage</a> •
  <a href="#contributing">Contributing</a>
</p>

</div>

---

## Features

-   **Two model versions** — V1 (LSTM next-keystroke predictor) and V2 (Transformer encoder-decoder for phrase-level generation)
-   **Realistic timing** — Inter-key intervals sampled from LogNormal distributions predicted by the model
-   **Natural errors & corrections** — The model generates typing mistakes and backspace corrections, not just correct characters
-   **WPM persona conditioning** — Generate typing patterns for any speed level (10 WPM buckets from hunt-and-peck to 170+ WPM)
-   **Live terminal preview** — Real-time visualization of model-generated typing with ANSI color coding
-   **TF.js export** — Quantized models ready for Chrome extension deployment via WASM backend
-   **Modular pipeline** — Clean separation of data processing, model definition, training, inference, evaluation, and export

## Quickstart

```bash
# Clone and install
git clone https://github.com/owengregson/Lilly.git
cd Lilly
pip install -e ".[dev]"

# Download dataset (~15GB)
python scripts/download.py

# Preprocess raw keystrokes
python scripts/preprocess.py --workers 8

# Extract features (V1)
python scripts/extract_features.py

# Train V1 model
python scripts/train.py --version v1 --epochs 30

# Live preview
python scripts/live_preview.py "Hello, world!"
```

## Project Structure

```
Lilly/
├── lilly/                      # Main package
│   ├── core/                   # Shared utilities
│   │   ├── config.py           #   Paths, constants, model/training configs
│   │   ├── encoding.py         #   Character ↔ ID encoding, WPM buckets
│   │   ├── keyboard.py         #   QWERTY layout, key distances, finger map
│   │   └── losses.py           #   LogNormal NLL, masked cross-entropy
│   ├── data/                   # Data pipeline
│   │   ├── download.py         #   Aalto dataset downloader
│   │   ├── preprocess.py       #   Raw keystroke parsing & alignment
│   │   ├── features.py         #   V1 dense feature extraction
│   │   ├── segment.py          #   V2 pause-based segmentation
│   │   └── pipeline.py         #   tf.data builders for V1 & V2
│   ├── models/                 # Model definitions
│   │   ├── lstm.py             #   V1: LSTM with 3 output heads
│   │   └── transformer.py      #   V2: Transformer encoder-decoder
│   ├── training/               # Training logic
│   │   ├── callbacks.py        #   Keras callbacks
│   │   ├── trainer_v1.py       #   V1 Keras .fit() trainer
│   │   └── trainer_v2.py       #   V2 GradientTape trainer
│   ├── inference/              # Generation & preview
│   │   ├── sampling.py         #   LogNormal sampling, temperature
│   │   ├── context.py          #   V1 sliding context window
│   │   ├── generator.py        #   V1 & V2 sequence generation
│   │   └── preview.py          #   Live terminal renderer
│   ├── evaluation/             # Metrics & visualization
│   │   └── evaluator.py        #   Timing MAE, action accuracy, plots
│   └── export/                 # Model export
│       └── converter.py        #   Keras → TF.js conversion
├── scripts/                    # CLI entry points
│   ├── download.py             ├── train.py
│   ├── preprocess.py           ├── evaluate.py
│   ├── extract_features.py     ├── generate.py
│   ├── live_preview.py         └── export.py
├── tests/                      # Test suite
├── configs/                    # YAML configs (v1.yaml, v2.yaml)
├── pyproject.toml              # Package definition
├── Makefile                    # Common commands
└── CLAUDE.md                   # AI assistant context
```

## Architecture

### V1: LSTM Next-Keystroke Predictor

Predicts the **next keystroke** from a sliding window of 32 previous keystrokes.

```
Inputs (per timestep × 32):
  typed_chars ─┐
  target_chars ├─→ Embeddings ─→ Concat ─→ LSTM(128) ─→ LSTM(64) ─→ Dropout
  actions     ─┘                    ↑                                    │
  dense_features ───────────────────┘                                    │
                                                                         ↓
  wpm_bucket ─→ Embedding ─────────────────────────────────→ Concat ─→ Dense(64)
                                                                         │
                                              ┌──────────────┬───────────┤
                                              ↓              ↓           ↓
                                        timing (2)    action (3)   error_char (97)
                                        [μ, log σ]    [softmax]    [softmax]
```

**Three output heads:**
| Head | Output | Loss |
|------|--------|------|
| Timing | μ, log σ for LogNormal IKI | Negative log-likelihood |
| Action | correct / error / backspace | Sparse categorical CE (weight 2.0) |
| Error char | Which wrong key was typed | Sparse categorical CE (weight 0.5, masked) |

### V2: Transformer Encoder-Decoder

Generates entire **keystroke segments** (phrase-level) autoregressively.

```
Encoder:                              Decoder:
  target_text ─→ Char Embed ─┐         [START, c₁, c₂, ...] ─→ Char Embed ─┐
  wpm_bucket ──→ WPM Embed  ─┤         [0, d₁, d₂, ...]    ─→ Delay Proj ──┤
  prev_context ───────────────┤                                               │
                              ↓                                               ↓
                   Positional Encoding              Positional Encoding + Causal Mask
                              ↓                                               ↓
                   2× Encoder Layers               2× Decoder Layers (self + cross attn)
                              │                                               │
                              └──────── Cross-Attention ──────────────────────┘
                                                                              │
                                                              ┌───────────────┤
                                                              ↓               ↓
                                                      char_logits (99)  delay_params (2)
```

## Usage

### Training

```bash
# V1 LSTM model
python scripts/train.py --version v1 --epochs 30 --batch-size 512

# V2 Transformer model
python scripts/train.py --version v2 --epochs 30 --batch-size 256
```

### Generation

```bash
# Generate keystroke trace (V1)
python scripts/generate.py models/run_XXX/final_model.keras "The quick brown fox"

# Generate keystroke trace (V2)
python scripts/generate.py --version v2 models/v2/run_XXX/best_model.keras "Hello!"
```

### Live Preview

```bash
# Interactive mode (V1)
python scripts/live_preview.py

# Specific text with WPM and speed control
python scripts/live_preview.py --wpm 80 --speed 0.5 "Some text here"

# V2 model
python scripts/live_preview.py --version v2 "Hello, world!"
```

### Evaluation

```bash
# V1 model evaluation with plots
python scripts/evaluate.py models/run_XXX/final_model.keras

# V2 model evaluation
python scripts/evaluate.py --version v2 models/v2/run_XXX/best_model.keras
```

### Export to TF.js

```bash
# Export with uint8 quantization (default)
python scripts/export.py models/run_XXX/best_model.keras

# Export V2 with float16
python scripts/export.py --version v2 models/v2/run_XXX/best_model.keras --quantize float16
```

### Makefile Shortcuts

```bash
make install          # pip install -e .
make dev              # pip install -e ".[dev]"
make test             # Run test suite
make lint             # Ruff linting
make pipeline         # Full V1 pipeline: download → preprocess → features → train
make train-v1         # Train V1 model
make train-v2         # Train V2 model
```

## Dense Features (V1)

The V1 model uses 14 engineered features per keystroke:

| #   | Feature               | Description                                    |
| --- | --------------------- | ---------------------------------------------- |
| 0   | `iki_log`             | log(IKI ms) / log(5000), clipped [10, 5000]    |
| 1   | `hold_time_log`       | log(hold ms) / log(2000), clipped [5, 2000]    |
| 2   | `key_distance`        | QWERTY Euclidean distance / 10, capped at 1.0  |
| 3   | `pos_in_sentence`     | target_pos / sentence_length                   |
| 4   | `pos_in_word`         | position within current word / word_length     |
| 5   | `is_word_start`       | 1.0 if first character of a word               |
| 6   | `is_word_end`         | 1.0 if last character before space/punctuation |
| 7   | `is_sentence_end`     | 1.0 if punctuation (.!?;:)                     |
| 8   | `chars_since_error`   | Capped at 50, normalized                       |
| 9   | `consecutive_correct` | Capped at 50, normalized                       |
| 10  | `rolling_iki_mean`    | Mean of last 10 IKIs / MAX_IKI                 |
| 11  | `rolling_iki_std`     | Std of last 10 IKIs / MAX_IKI                  |
| 12  | `rolling_error_rate`  | Error rate in last 20 keystrokes               |
| 13  | `elapsed_fraction`    | Keystrokes typed / total session length        |

## Character Encoding

| Range | Meaning                                             |
| ----- | --------------------------------------------------- |
| 0     | PAD                                                 |
| 1–95  | Printable ASCII (space `0x20` through tilde `0x7E`) |
| 96    | BACKSPACE                                           |
| 97    | END (V2 only)                                       |
| 98    | START (V2 only)                                     |

## WPM Buckets

10 persona buckets: `[0, 30, 45, 60, 75, 90, 105, 120, 140, 170, 999]`

The model is conditioned on WPM bucket so it can generate typing patterns appropriate for different speed levels — from hunt-and-peck (~25 WPM) to professional typists (~150+ WPM).

## Dataset

This project uses the [Aalto 136M Keystrokes](https://userinterfaces.aalto.fi/136Mkeystrokes/) dataset:

> Dhakal, V., Feit, A. M., Kristensson, P. O., & Oulasvirta, A. (2018). Observations on Typing from 136 Million Keystrokes. _CHI 2018_.

The dataset contains 136 million keystrokes from 168,000 participants completing typing tests, including timestamps, key identifiers, and target sentences.

## Contributing

Contributions are welcome. Please follow these guidelines:

### Getting Started

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Install dev dependencies: `pip install -e ".[dev]"`
4. Make your changes

### Code Standards

-   **Style**: Follow existing patterns. Use [Ruff](https://docs.astral.sh/ruff/) for linting (`make lint`)
-   **Testing**: Add tests for new functionality (`make test`)
-   **Commits**: Use clear, descriptive commit messages
-   **Type hints**: Use type annotations for function signatures

### Pull Request Process

1. Ensure tests pass: `make test`
2. Ensure linting passes: `make lint`
3. Update documentation if adding new features
4. Open a PR against `main` with a clear description

### Reporting Issues

-   Use [GitHub Issues](https://github.com/owengregson/Lilly/issues) to report bugs or request features
-   Include reproduction steps, expected vs. actual behavior, and environment details
-   For model quality issues, include the WPM setting, input text, and observed output

### Areas for Contribution

-   **Layout support**: Dvorak, Colemak, etc. (keyboard.py variants)
-   **Language support**: Unicode/multilingual vocabulary expansion
-   **Mobile typing**: Touchscreen error patterns and timing
-   **V2 improvements**: Attention visualization, learned segmentation boundaries
-   **Multi-modal timing**: Mixture of LogNormals for bimodal IKI distributions
-   **Personalization**: Fine-tuning on individual user data

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">
  <sub>Built with TensorFlow • Trained on 136M keystrokes • Deployable via TF.js</sub>
</div>
