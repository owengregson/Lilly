# Lilly V3 Design Document

**Date:** 2026-03-01
**Status:** Approved
**Author:** Owen Gregson + Claude

## Goal

Rebuild the Lilly typing behavior model from the ground up to address five fundamental flaws identified in the V1/V2 evaluation:

1. Unimodal LogNormal timing distribution
2. Split architecture (V1 has error heads but no seq2seq; V2 has seq2seq but no error heads)
3. Hardcoded error correction in V1 generator
4. Evaluation metrics that don't measure distributional realism
5. Coarse WPM bucketing discards typing persona signal

V3 replaces V1 and V2 in-place. Quality first, TF.js optimization later.

## Decisions

- **V1/V2 code:** Replace in-place (delete model, trainer, generator code; keep shared infra)
- **Model size:** Quality first (~2-3M params), optimize/distill later for browser deployment
- **Data pipeline:** Reuse existing Parquet preprocessing, build new V3 segmentation on top
- **User conditioning:** Style Vector + FiLM conditioning (16-dim style vector from session stats)
- **Timing model:** Full 8-component LogNormal MDN from day one, separate per action path
- **Decoder:** Action-Gated Decoder with correct/error/backspace routing

---

## 1. Model Architecture

### Overview

Unified encoder-decoder transformer with an action-gated decoder. The encoder reads target text + style conditioning. The decoder autoregressively generates keystroke events, predicting action (correct/error/backspace) first, then routing to specialized heads for character and timing.

### Encoder

```
Target text chars -> Char Embedding (48d) -> Linear Proj (d_model=128)
                                                    |
                                                    + Sinusoidal PE
                                                    |
                                                    + FiLM(style_vector)
                                                    |
                                                    v
                                            4x Encoder Layers
                                         (d=128, 8 heads, ff=256)
                                                    |
                                                    v
                                           encoder_output (B, S, 128)
```

**Inputs:**
- `target_chars`: (B, max_enc_len) int32
- `target_lengths`: (B,) int32
- `style_vector`: (B, 16) float32

### Decoder

```
Previous keystroke (char_id, delay, action)
       |
       v
  Char Embed (48d) + Delay Proj (16d) + Action Embed (16d) = 80d
       |
       v
  Linear Proj (d_model=128) + Sinusoidal PE + FiLM(style_vector)
       |
       v
  4x Decoder Layers (self-attn + cross-attn, d=128, 8 heads)
       |
       v
  hidden_state (B, T, 128)
       |
       +---> ACTION GATE: Dense(64,relu) -> Dense(3, softmax)
       |
       +---> CORRECT HEAD:  timing MDN (8 components)
       |                     char = copy from encoder at current position
       |
       +---> ERROR HEAD:    timing MDN (8 components)
       |                     char = Dense(97, softmax) + QWERTY distance bias
       |
       +---> BACKSPACE HEAD: timing MDN (8 components)
                              char = always BACKSPACE token
```

### Action Gate

At each decoder step:
1. Predict `p = softmax(W_gate @ hidden + b_gate)` -> (correct, error, backspace)
2. Training: teacher forced with ground-truth action labels
3. Inference: sample from action distribution, route to selected head

### Timing MDN (per action path)

Each action path has its own 8-component LogNormal MDN:

```
hidden (128d) -> Dense(64, relu) -> Dense(8*3) -> reshape to (8, 3)
                                                    [pi_k, mu_k, log_sigma_k]
```

- `pi = softmax(pi_raw)` (mixture weights)
- `sigma = exp(clip(log_sigma, -5, 5))`
- Loss: `NLL = -log(sum_k pi_k * LogNormal(t | mu_k, sigma_k))`

Separate MDNs per action means: correct-MDN learns within-word/between-word timing; error-MDN learns slip vs cognitive error timing; backspace-MDN learns immediate vs delayed correction timing.

### Error Character Head

Uses QWERTY-distance bias:

```
logits = W_error @ hidden + b_error
distance_bias = -alpha * qwerty_distance_matrix[target_char]
biased_logits = logits + distance_bias
error_char = softmax(biased_logits)
```

`alpha` is a learned scalar. `qwerty_distance_matrix` is precomputed (97, 97).

### Model Dimensions

| Parameter | Value |
|-----------|-------|
| d_model | 128 |
| nhead | 8 |
| num_encoder_layers | 4 |
| num_decoder_layers | 4 |
| dim_feedforward | 256 |
| char_embed_dim | 48 |
| action_embed_dim | 16 |
| delay_embed_dim | 16 |
| style_dim | 16 |
| mdn_components | 8 |
| dropout | 0.1 |
| Estimated params | ~2-3M |

---

## 2. Style Vector & FiLM Conditioning

### Style Vector (16 dimensions)

Computed per session from observable typing statistics:

| Dim | Feature | Description |
|-----|---------|-------------|
| 0 | mean_iki_log | Mean of log(IKI) |
| 1 | std_iki_log | Rhythm regularity |
| 2 | median_iki_log | Robust central tendency |
| 3 | iki_skewness | Timing distribution asymmetry |
| 4 | mean_burst_length | Avg keystrokes between pauses |
| 5 | std_burst_length | Burst length variance |
| 6 | error_rate | Fraction of error keystrokes |
| 7 | correction_rate | Fraction of errors corrected within 3 keystrokes |
| 8 | mean_correction_latency_log | Mean log-time error to correction |
| 9 | bigram_speed_variance | Consistency across bigrams |
| 10 | pause_frequency | Fraction of IKIs > 300ms |
| 11 | mean_hold_time_log | Mean key hold duration (log) |
| 12 | iki_autocorrelation | Lag-1 IKI autocorrelation |
| 13 | word_boundary_slowdown | space-IKI / within-word-IKI ratio |
| 14 | error_burst_rate | Fraction of clustered errors |
| 15 | session_speedup_trend | Linear slope of rolling IKI |

Z-score normalized using training set statistics stored in `data/v3_segments/style_norm.json`.

### FiLM Layer

```python
class FiLMModulation(Layer):
    def __init__(self, d_model, style_dim=16):
        self.gamma_proj = Dense(d_model)
        self.beta_proj = Dense(d_model)

    def call(self, x, style_vector):
        gamma = 1.0 + self.gamma_proj(style_vector)
        beta = self.beta_proj(style_vector)
        return gamma[:, None, :] * x + beta[:, None, :]
```

Applied after LayerNorm in each encoder and decoder layer.

### Inference Mapping (Sliders -> Style Vector)

1. During training, collect all (style_vector, WPM, error_rate) tuples
2. Build lookup grid: for each (WPM_bin, error_bin), compute mean style vector
3. At inference: bilinear interpolation from 4 nearest grid points
4. Ship as ~13KB JSON lookup table in the Chrome extension

---

## 3. Data Pipeline

### Flow

```
Existing Parquet -> V3 Segmentation -> .npz chunks -> tf.data.Dataset
```

Reuses `preprocess.py` output. New V3-specific segmentation adds action labels, style vectors, and richer context.

### Segment Format

```python
{
    "encoder_chars":        (max_enc_len,) int32,
    "encoder_lengths":      () int32,
    "decoder_chars":        (max_dec_len,) int32,
    "decoder_delays":       (max_dec_len,) float32,
    "decoder_actions":      (max_dec_len,) int32,
    "decoder_lengths":      () int32,
    "style_vector":         (16,) float32,
    "prev_context_chars":   (4,) int32,
    "prev_context_actions": (4,) int32,
    "prev_context_delays":  (4,) float32,
}
```

### Key differences from V2

- Action labels preserved per keystroke (critical for action gate)
- Style vector per segment (computed once per session)
- Richer context: (char, action, delay) tuples instead of just char IDs
- max_decoder_len increased from 64 to 80

---

## 4. Training

### Loss Function

```
L_total = 3.0 * L_action + 1.0 * L_timing + 1.0 * L_error_char + 0.1 * L_position
```

**L_action (Focal Loss):**
```
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
alpha = [0.25, 0.5, 0.5], gamma = 2.0
```

**L_timing (MDN Mixture NLL):**
```
L = -log(sum_k pi_k * LogNormal(t | mu_k, sigma_k))
```
Per-action masking: each MDN only trained on samples matching its action.

**L_error_char:** Sparse CE on QWERTY-biased logits, masked to action=ERROR.

**L_position:** MSE auxiliary loss predicting normalized position. Weight 0.1.

### Optimizer

```python
schedule = WarmupCosineDecay(peak_lr=3e-4, warmup_steps=2000, decay_steps=total, min_lr=1e-6)
optimizer = AdamW(lr=schedule, weight_decay=1e-4, beta_2=0.98, epsilon=1e-9)
gradient_clip = 1.0 (global norm)
```

### Training Config

| Parameter | Value |
|-----------|-------|
| batch_size | 128 |
| epochs | 50 |
| learning_rate | 3e-4 |
| weight_decay | 1e-4 |
| warmup_steps | 2000 |
| early_stop_patience | 10 |
| gradient_clip | 1.0 |

---

## 5. Evaluation

### Tier 1: Point Metrics (every epoch)

- action_accuracy, action_f1_per_class
- timing_mae_ms, timing_nll
- error_char_accuracy, error_char_top3
- reconstruction_accuracy (100 samples)

### Tier 2: Distributional Metrics (end of training)

- iki_wasserstein, iki_ks_statistic
- burst_length_wasserstein, pause_duration_wasserstein
- error_position_chi2, correction_latency_wasserstein
- iki_autocorrelation_mae

All computed over 1000 generated vs 1000 real sequences.

### Tier 3: Realism Metrics (final validation)

- discriminator_accuracy: small LSTM classifier on real vs generated
- style_consistency: does WPM/error rate match target settings?
- qualitative_replay: terminal playback at various settings

### Visualization

- IKI distribution overlays (per-action breakdown)
- Burst pattern comparison (real vs generated)
- Action confusion matrix
- MDN component visualization (8 LogNormals per action)
- Style interpolation (smooth transition between personas)

---

## 6. Inference

### Generation Loop

Fully model-driven, no hardcoded correction:

```
for each step:
    action = sample(action_gate(hidden))
    if CORRECT:  char = target[position]; timing from correct-MDN; position++
    if ERROR:    char = sample(error_head); timing from error-MDN
    if BACKSPACE: char = BKSP; timing from backspace-MDN; adjust position
```

### Temperature Controls

- `action_temperature`: error/correction frequency
- `timing_temperature`: timing variability
- `char_temperature`: error character selection

### Segment-Level Generation

Split text at word boundaries, propagate (char, delay, action) context between segments.

---

## 7. File Structure

### Delete (V1/V2)

```
lilly/models/lstm.py, lilly/models/transformer.py
lilly/data/features.py
lilly/training/trainer_v1.py, lilly/training/trainer_v2.py
lilly/inference/context.py, lilly/inference/generator.py
lilly/evaluation/evaluator.py
scripts/extract_features.py
```

### Keep (shared)

```
lilly/core/{config,encoding,keyboard}.py
lilly/data/{download,preprocess,pipeline}.py
lilly/inference/sampling.py
lilly/export/converter.py
lilly/training/callbacks.py
```

### New

```
lilly/models/typing_model.py      # TypingTransformerV3
lilly/models/components.py        # FiLM, MDN, ActionGate, ErrorCharHead
lilly/data/segment_v3.py          # V3 segmentation + style vectors
lilly/data/style.py               # Style vector computation
lilly/training/trainer.py         # V3 training loop
lilly/training/losses.py          # Focal, MDN, combined loss
lilly/training/schedule.py        # WarmupCosineDecay
lilly/inference/generator.py      # V3 generation (replaces old)
lilly/evaluation/metrics.py       # Tier 1
lilly/evaluation/distributional.py # Tier 2
lilly/evaluation/realism.py       # Tier 3
lilly/evaluation/visualization.py  # Plots
scripts/segment_v3.py             # V3 data prep
```

### Config

```python
@dataclass
class V3ModelConfig:
    max_encoder_len: int = 32
    max_decoder_len: int = 80
    num_char_classes: int = 99
    char_embed_dim: int = 48
    action_embed_dim: int = 16
    delay_embed_dim: int = 16
    d_model: int = 128
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1
    style_dim: int = 16
    mdn_components: int = 8
    num_actions: int = 3
    qwerty_bias_learnable: bool = True
    context_tail_len: int = 4

@dataclass
class V3TrainConfig:
    batch_size: int = 128
    epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 2000
    early_stop_patience: int = 10
    action_loss_weight: float = 3.0
    timing_loss_weight: float = 1.0
    error_char_loss_weight: float = 1.0
    position_loss_weight: float = 0.1
    focal_gamma: float = 2.0
    focal_alpha: tuple = (0.25, 0.5, 0.5)
    val_split: float = 0.1
    test_split: float = 0.05
    shuffle_buffer: int = 100_000
    seed: int = 42
    max_samples: int = 0
```
