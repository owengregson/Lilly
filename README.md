<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="images/LillyLogoDesignV4SE2Dark.svg">
  <source media="(prefers-color-scheme: dark)" srcset="images/LillyLogoDesignV4SE2Light.svg">
  <img alt="Lilly" src="images/LillyLogoDesignV4SE2Light.svg" width="320">
</picture>

<br><br>

**A State-of-the-Art (SoTA) Generative Model for Human Typing Behavior**

<p>
Lilly is a novel deep learning model that generates indistinguishable-from-human keystroke sequences<br>
with realistic timing, natural typos, and organic corrections trained on
<a href="https://userinterfaces.aalto.fi/136Mkeystrokes/">136,000,000+ real keystrokes</a>.
</p>

<p>
  <a href="#setup"><img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="#setup"><img src="https://img.shields.io/badge/tensorflow-2.15%2B-orange?logo=tensorflow&logoColor=white" alt="TensorFlow 2.15+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"></a>
</p>

<p>
  <a href="#what-makes-lilly-different">Why Lilly</a> •
  <a href="#features">Features</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#usage">Usage</a> •
  <a href="#contributing">Contributing</a>
</p>

</div>

---

## What Makes Lilly Different

Most approaches to typing simulation treat timing, errors, and corrections as separate problems — or ignore errors entirely. Lilly takes a fundamentally different approach with a **novel action-gated architecture** that unifies all three into a single autoregressive generation process:

1. **The model decides everything.** At each step, Lilly predicts whether the next keystroke is correct, an error, or a backspace — then routes to specialized heads for that action.

2. **Multimodal timing.** Human typing isn't a single bell curve. Lilly uses 24-component mixture density networks (8 per action type) to capture the full multimodal distribution of inter-key intervals — the fast bursts within words, the pauses between them, the hesitation before corrections.

3. **Learned error physics.** Typing errors are not random. Lilly's error head learns QWERTY keyboard geometry through a differentiable distance bias matrix, producing typos that respect finger mechanics and key adjacency.

4. **Style-controllable generation.** A 16-dimensional style vector conditioned via FiLM (Feature-wise Linear Modulation) at every transformer layer means the same model can generate typing patterns ranging from 30 WPM hunt-and-peck to 170+ WPM professional at inference time.

## Features

-   **Action-gated transformer** — Novel encoder-decoder architecture that predicts action (correct / error / backspace) first, then routes to specialized heads for timing and character selection
-   **Mixture density timing** — 8-component LogNormal MDN per action type (24 total components) captures the full multimodal distribution of real human inter-key intervals
-   **QWERTY-aware error generation** — Learnable keyboard distance bias matrix produces physically plausible typos based on actual finger mechanics
-   **FiLM style conditioning** — 16-dimensional style vector modulates every encoder and decoder layer, enabling continuous control over typing speed, error rate, and rhythm
-   **3-tier evaluation** — Point metrics, distributional analysis (Wasserstein, KS), and discriminator-based realism scoring to verify human indistinguishability
-   **Live terminal preview** — Real-time visualization of generated typing with ANSI color coding
-   **TF.js export** — Quantized models ready for web automation deployment via WASM backend

## Getting Started

This walks you from zero to a trained model. Every step includes the exact commands.

### Step 1: Prerequisites

-   **Python 3.10+** — [python.org/downloads](https://www.python.org/downloads/)
-   **Git** — [git-scm.com](https://git-scm.com/)
-   ~2 GB disk for dependencies, ~15 GB for the full training dataset
-   **GPU (recommended)** — NVIDIA GPU with CUDA support for fast training. Training automatically uses GPU when available via TensorFlow's CUDA backend — no code changes needed.

### Step 2: Clone & Install

```bash
git clone https://github.com/owengregson/Lilly.git
cd Lilly
python setup_project.py
```

The setup script works on **macOS, Windows, and Linux**. It creates a virtual environment, installs all dependencies, and verifies the installation.

<details>
<summary><strong>Setup script options</strong></summary>

```bash
python setup_project.py --core-only   # Skip dev/eval/export extras
python setup_project.py --no-venv     # Use current Python instead of creating .venv
```

</details>

<details>
<summary><strong>Manual setup (without the script)</strong></summary>

```bash
git clone https://github.com/owengregson/Lilly.git
cd Lilly

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows

# Install with all extras
pip install -e ".[all]"
```

</details>

<details>
<summary><strong>GPU setup (Lambda, cloud instances, or local NVIDIA GPU)</strong></summary>

TensorFlow automatically detects and uses CUDA GPUs — no code changes needed. On a Lambda instance or any machine with NVIDIA drivers and CUDA already installed, the default setup just works.

If TensorFlow doesn't detect your GPU after install, you may need the CUDA-enabled package:

```bash
pip install tensorflow[and-cuda]
```

Verify GPU access:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

You should see your GPU(s) listed. If the list is empty, check that your NVIDIA drivers and CUDA toolkit are installed.

</details>

### Step 3: Activate the Virtual Environment

Every time you open a new terminal to work on Lilly, activate the venv first:

```bash
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows
```

Verify everything is working:

```bash
python -c "from lilly.core.config import V3ModelConfig; print('OK')"
```

### Step 4: Download the Dataset

Lilly trains on the [Aalto 136M Keystrokes](https://userinterfaces.aalto.fi/136Mkeystrokes/) dataset. This downloads and extracts a ~15 GB zip file to `data/raw/`:

```bash
python scripts/download.py
```

### Step 5: Preprocess Raw Keystrokes

Replays each typing session to classify every keystroke as correct, error, or backspace. Outputs Parquet files to `data/processed/`. Adjust `--workers` to match your CPU core count:

```bash
python scripts/preprocess.py --workers 8
```

### Step 6: Extract Training Segments

Extracts V3 training segments with style vectors, context windows, and teacher-forced decoder I/O. Outputs `.npz` files to `data/v3_segments/`:

```bash
python scripts/segment_v3.py --workers 8
```

### Step 7: Train

```bash
python scripts/train.py --epochs 50
```

Training uses AdamW with WarmupCosineDecay, gradient clipping, and early stopping. Checkpoints are saved to `models/v3/`. Default config: batch size 128, learning rate 3e-4, 2000 warmup steps. **GPU is used automatically** when available — no flags needed.

### One-Liner Alternative

If you want to run steps 4–7 all at once:

```bash
make pipeline
```

### Step 8 (Optional): Evaluate, Generate, Export

```bash
# Evaluate the trained model
python scripts/evaluate.py models/v3/run_XXX/best_model.keras --tier 1

# Generate keystroke sequences
python scripts/generate.py models/v3/run_XXX/best_model.keras "Hello, world!"

# Export to TF.js for browser deployment
python scripts/export.py models/v3/run_XXX/best_model.keras --quantize uint8
```

See [Usage](#usage) for the full command reference.

## Project Structure

```
Lilly/
├── lilly/                          # Core model package
│   ├── core/                       # Shared utilities
│   │   ├── config.py               #   Paths, constants, V3ModelConfig, V3TrainConfig
│   │   ├── encoding.py             #   char_to_id, id_to_char, wpm_to_bucket
│   │   └── keyboard.py             #   QWERTY layout, key distances, finger map
│   ├── data/                       # Data processing
│   │   ├── download.py             #   Aalto dataset downloader
│   │   ├── preprocess.py           #   Raw keystroke parsing & alignment
│   │   ├── segment.py              #   Inference-time text segmentation
│   │   ├── segment_v3.py           #   Training segment extraction
│   │   ├── style.py                #   Style vector computation & normalization
│   │   └── pipeline.py             #   tf.data builder (build_v3_datasets)
│   ├── models/                     # Model architecture
│   │   ├── components.py           #   FiLMModulation, MDNHead, ActionGate, ErrorCharHead
│   │   └── typing_model.py         #   TypingTransformerV3, EncoderLayer, DecoderLayer
│   ├── training/                   # Training logic
│   │   ├── losses.py               #   FocalLoss, mdn_mixture_nll, compute_v3_loss
│   │   ├── schedule.py             #   WarmupCosineDecay LR schedule
│   │   └── trainer.py              #   Custom GradientTape training loop
│   ├── inference/                  # Generation
│   │   ├── sampling.py             #   sample_mdn, weighted_sample, weighted_sample_logits
│   │   └── generator.py            #   generate_v3_segment, generate_v3_full
│   ├── evaluation/                 # Metrics & visualization
│   │   ├── metrics.py              #   Tier 1 point metrics (accuracy, F1, MAE, NLL)
│   │   ├── distributional.py       #   Tier 2 distributional (Wasserstein, KS)
│   │   ├── realism.py              #   Tier 3 realism (discriminator, style)
│   │   └── visualization.py        #   Plotting (IKI, bursts, confusion, MDN, style)
│   └── export/                     # Model export
│       └── converter.py            #   export_model (Keras → TF.js)
├── scripts/                        # CLI entry points
│   ├── download.py                 #   Download Aalto dataset
│   ├── preprocess.py               #   Parse & align raw keystrokes
│   ├── segment_v3.py               #   Extract training segments
│   ├── train.py                    #   Train the model
│   ├── evaluate.py                 #   Evaluate model (--tier 1|2|3)
│   ├── generate.py                 #   Generate keystroke sequences
│   ├── live_preview.py             #   Live terminal preview
│   └── export.py                   #   Export to TF.js
├── tests/                          # Test suite
├── pyproject.toml                  # Package definition & dependencies
├── Makefile                        # Common commands
├── setup_project.py                # Cross-platform setup script
└── CLAUDE.md                       # AI assistant context
```

## Architecture

### Overview

Lilly is built on a novel **action-gated encoder-decoder transformer**. Unlike standard seq2seq models that predict characters directly, Lilly first predicts the _type_ of keystroke action at each step, then routes to specialized output heads — mirroring how real typing involves distinct cognitive processes for correct input, error commission, and error correction.

```
           ┌─────────────────────────────────────────────────────────────────┐
           │                             ENCODER                             │
           │                                                                 │
           │       target_text ──→ Char Embed (48) ──→ + Sinusoidal PE       │
           │                                ↓                                │
           │     style_vector ──→ FiLM ──→ 4× Encoder Layers (d=128, 8h)     │
           │                                ↓                                │
           │                          encoder_output                         │
           └─────────────────────────────────────────────────────────────────┘
                                          │ cross-attention
           ┌──────────────────────────────┴──────────────────────────────────┐
           │                             DECODER                             │
           │                                                                 │
           │     [START, c₁, c₂, ...] → Char Embed ─┐                        │
           │     [0, d₁, d₂, ...]     → Delay Proj ─┤→ Concat → + Sin. PE    │
           │     [0, a₁, a₂, ...]     → Action Emb ─┘    ↓                   │
           │     prev_context ─────────────────────────→ Prepend             │
           │                                 ↓                               │
           │     style_vector ──→ FiLM ──→ 4× Decoder Layers (causal mask)   │
           │                                 ↓                               │
           └─────────────────────────────────┴───────────────────────────────┘
                                             │
                        ┌────────────────────┼────────────────────┐
                        ↓                    ↓                    ↓
                  ┌────────────┐    ┌──────────────────┐   ┌────────────┐
                  │ ActionGate │    │  3× MDN Timing   │   │ ErrorChar  │
                  │ Dense → 3  │    │  Heads (8-comp)  │   │ Head (97)  │
                  │ [softmax]  │    │  [π, μ, log σ]   │   │ + QWERTY   │
                  └─────┬──────┘    └───────┬──────────┘   │ dist bias  │
                        │                   │              └─────┬──────┘
                        ↓                   ↓                    ↓
                  correct/error/       delay_ms per        error character
                    backspace          action type           prediction
```

### Encoder

-   Target text encoded via character embedding (dim 48) + sinusoidal positional encoding
-   FiLM conditioning applied at every layer: `output = γ(style) * x + β(style)`
-   4 encoder layers with multi-head attention (d_model=128, 8 heads, ff=256)

### Decoder

-   Autoregressive input: character ID + log-delay + action type embeddings concatenated and projected
-   Previous segment context (tail of last segment) prepended for cross-segment continuity
-   4 decoder layers with causal self-attention, cross-attention to encoder, and FiLM conditioning

### Output Heads

| Head                | Output                          | Description                                                    |
| ------------------- | ------------------------------- | -------------------------------------------------------------- |
| **ActionGate**      | 3 classes (softmax)             | Predicts correct / error / backspace at each step              |
| **MDN Timing** (×3) | π, μ, log σ (8 components each) | Per-action LogNormal mixture for inter-key intervals           |
| **ErrorCharHead**   | 97 classes                      | Error character prediction with learnable QWERTY distance bias |
| **PositionHead**    | max_encoder_len classes         | Predicted position in target text                              |

### Style Conditioning

The 16-dimensional style vector captures per-session typing characteristics:

| Dimension | Feature              | Description                                  |
| --------- | -------------------- | -------------------------------------------- |
| 0         | `mean_iki_log`       | Log mean inter-key interval                  |
| 1         | `std_iki_log`        | Log std of inter-key intervals               |
| 2         | `error_rate`         | Fraction of error keystrokes                 |
| 3         | `burst_length`       | Mean length of fast-typing bursts            |
| 4         | `correction_latency` | Mean delay before backspace corrections      |
| 5–15      | Various              | Autocorrelation, word-boundary effects, etc. |

Style conditioning uses **FiLM (Feature-wise Linear Modulation)** — learned linear projections from the style vector produce per-layer scale (γ) and shift (β) parameters that modulate every transformer hidden state. At inference time, this enables continuous control over typing persona via simple sliders (WPM, error rate) mapped through a precomputed style lookup grid.

### Loss Functions

| Loss       | Function                                       | Weight |
| ---------- | ---------------------------------------------- | ------ |
| Action     | Focal loss (γ=2.0, α=[0.25, 0.5, 0.5])         | 3.0    |
| Timing     | MDN mixture NLL (LogNormal, masked per action) | 1.0    |
| Error char | Sparse CE (masked to action=error steps only)  | 1.0    |
| Position   | Sparse CE for target position                  | 0.1    |

### Character Encoding

| Range | Meaning                                             |
| ----- | --------------------------------------------------- |
| 0     | PAD                                                 |
| 1–95  | Printable ASCII (space `0x20` through tilde `0x7E`) |
| 96    | BACKSPACE                                           |
| 97    | END                                                 |
| 98    | START                                               |

## Usage

### Training

```bash
# Full end-to-end (download → preprocess → segment → train)
make pipeline

# Or train individually with custom settings
python scripts/train.py --epochs 100 --batch-size 64
```

See [Getting Started](#getting-started) for the full step-by-step walkthrough.

### Generation

```bash
# Generate keystroke sequence
python scripts/generate.py models/v3/run_XXX/best_model.keras "The quick brown fox"

# With WPM and error rate control
python scripts/generate.py models/v3/run_XXX/best_model.keras "Hello" --wpm 80 --error-rate 0.05
```

### Live Preview

```bash
# Watch the model type in real time
python scripts/live_preview.py --wpm 80 "Hello, world!"
```

### Evaluation

Three tiers of progressively deeper quality assessment:

```bash
# Tier 1: Point metrics (accuracy, F1, MAE, NLL)
python scripts/evaluate.py models/v3/run_XXX/best_model.keras --tier 1

# Tier 2: Distributional metrics (Wasserstein distance, KS test)
python scripts/evaluate.py models/v3/run_XXX/best_model.keras --tier 2 --n-samples 500

# Tier 3: Realism scoring (discriminator-based human indistinguishability)
python scripts/evaluate.py models/v3/run_XXX/best_model.keras --tier 3
```

### Export to TF.js

```bash
# Export with uint8 quantization (default, smallest size)
python scripts/export.py models/v3/run_XXX/best_model.keras --quantize uint8

# Export with float16 (higher precision)
python scripts/export.py models/v3/run_XXX/best_model.keras --quantize float16
```

### Makefile Shortcuts

```bash
make install      # pip install -e .
make dev          # pip install -e ".[all]"
make test         # Run test suite
make lint         # Ruff linting
make pipeline     # Full: download → preprocess → segment → train
make evaluate     # Evaluate (set MODEL=path/to/model.keras)
make export       # Export to TF.js (set MODEL=path/to/model.keras)
```

## Model Configuration

Key hyperparameters (defined in `V3ModelConfig` and `V3TrainConfig`):

| Parameter            | Value | Description                       |
| -------------------- | ----- | --------------------------------- |
| `d_model`            | 128   | Transformer hidden dimension      |
| `nhead`              | 8     | Attention heads                   |
| `num_encoder_layers` | 4     | Encoder depth                     |
| `num_decoder_layers` | 4     | Decoder depth                     |
| `dim_feedforward`    | 256   | FFN intermediate dimension        |
| `char_embed_dim`     | 48    | Character embedding dimension     |
| `style_dim`          | 16    | Style vector dimension            |
| `mdn_components`     | 8     | MDN mixture components per head   |
| `max_decoder_len`    | 80    | Max keystrokes per segment        |
| `max_encoder_len`    | 32    | Max target characters per segment |
| `batch_size`         | 128   | Training batch size               |
| `learning_rate`      | 3e-4  | Peak learning rate (AdamW)        |
| `warmup_steps`       | 2000  | Linear warmup steps               |
| `focal_gamma`        | 2.0   | Focal loss focusing parameter     |

## Dataset

Lilly is trained on the [Aalto 136M Keystrokes](https://userinterfaces.aalto.fi/136Mkeystrokes/) dataset — one of the largest public keystroke datasets ever collected:

> Dhakal, V., Feit, A. M., Kristensson, P. O., & Oulasvirta, A. (2018). Observations on Typing from 136 Million Keystrokes. _CHI 2018_.

The dataset contains **136 million keystrokes** from **168,000 participants** completing typing tests, including precise timestamps, key identifiers, and target sentences — providing the scale and diversity needed to learn the full spectrum of human typing behavior.

## Contributing

Contributions are welcome! Please follow these guidelines.

### Getting Started

1. Fork the repository
2. Clone your fork and run setup:
    ```bash
    git clone https://github.com/<your-username>/Lilly.git
    cd Lilly
    python setup_project.py
    ```
3. Create a feature branch:
    ```bash
    git checkout -b feature/my-feature
    ```

### Code Standards

-   **Style** — Follow existing patterns. Line length 100. Lint with [Ruff](https://docs.astral.sh/ruff/): `make lint`
-   **Testing** — Add tests for new functionality: `make test`
-   **Type hints** — Use type annotations for all function signatures
-   **Commits** — Use clear, descriptive commit messages in imperative mood

### Running Tests

```bash
# Run full test suite
make test

# Run a specific test file
python -m pytest tests/test_components.py -v

# Run with coverage
python -m pytest tests/ --cov=lilly --cov-report=term-missing
```

### Linting

```bash
# Check for issues
make lint

# Auto-fix
ruff check --fix lilly/ scripts/ tests/
```

### Pull Request Process

1. Ensure all tests pass: `make test`
2. Ensure linting passes: `make lint`
3. Update documentation if adding new features
4. Open a PR against `main` with a clear description of what and why

### Reporting Issues

-   Use [GitHub Issues](https://github.com/owengregson/Lilly/issues)
-   Include reproduction steps, expected vs actual behavior, and environment details
-   For model quality issues, include the style settings, input text, and observed output

### Areas for Contribution

-   **Keyboard layouts** — Dvorak, Colemak, AZERTY (keyboard.py variants + distance matrices)
-   **Language support** — Unicode/multilingual vocabulary expansion
-   **Mobile typing** — Touchscreen error patterns and timing
-   **Evaluation** — Additional realism metrics, human evaluation framework
-   **Style transfer** — Fine-tuning on individual user typing data
-   **Deployment** — Chrome extension integration, WebSocket streaming

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">
  <sub>Built with TensorFlow • Trained on 136M keystrokes • Deployable via TF.js</sub>
</div>
