# Phrase-Level Typing Model — Design Document

**Date:** 2025-02-25
**Status:** Approved
**Replaces:** LSTM 3-head next-keystroke model (`models/run_20260224_173441/`)

## Problem Statement

The current model is a 3-head LSTM that predicts the next keystroke (timing, action, error character) from a 32-keystroke sliding window. It has fundamental flaws:

1. **Correction strategy is unlearned.** Backspace prediction is 66.7% accurate, so generation code hardcodes immediate correction after every error. Real humans use diverse strategies (immediate, delayed, word-end, never correct) — the richest part of typing personality is thrown away.

2. **Character-by-character framing mismatches how humans type.** People execute motor plans in word/phrase-level bursts, not character-by-character decisions. Errors emerge from biomechanics of executing those plans, not from per-character coin flips.

3. **Timing and action are predicted independently.** The model outputs timing, action, and error char from separate heads sampled independently. In reality, error keystrokes have different timing distributions than correct ones — they're coupled.

4. **No cross-word error support.** The model has no concept of word boundaries. Errors that span words (missing a space, backspacing into the previous word) can't be represented.

## Design

### Core Idea: Sequence-to-Sequence at the Phrase Level

Replace next-keystroke prediction with **phrase-level sequence generation**. The model takes a target text segment (a few words) and generates the complete keystroke sequence to type it — including all errors, corrections, and timing as one coherent unit.

```
Input:  "the quick"  + context
Output: t(82ms) h(91ms) e(78ms) _(65ms) q(110ms) u(88ms) o(60ms) ⌫(195ms) i(85ms) c(92ms) k(105ms) <END>
```

Errors and corrections are generated together as part of the same sequence, so the model learns complete error-correction narratives from real human data rather than stitching them together from independent predictions.

### Segmentation Strategy

Training examples are extracted from the Aalto data by splitting at **natural pauses** — gaps of 300ms+ between keystrokes. This is how typing actually works: people type in bursts of 2-5 words, then pause.

Segments typically span 15-40 keystrokes. The pause threshold (300ms) is tunable. At inference time, we split input text into segments using a simple heuristic (every 2-4 words) or by learning segment boundaries from pause distributions in the data.

Edge cases:
- Cross-word errors are naturally captured within segments
- Space characters are part of the segment (not a boundary)
- Punctuation attached to words stays in the same segment
- If a segment exceeds max length (e.g., 64 tokens), truncate at the nearest word boundary

### Model Architecture: Small Transformer Encoder-Decoder

#### Encoder (target text + context)

Inputs:
- **Target characters** — the text segment to type, character-level (e.g., "the quick" → `[t, h, e, ' ', q, u, i, c, k]`)
- **WPM bucket** — 10-bucket persona conditioning (same buckets as current model)
- **Previous segment tail** — last 4 keystrokes from the prior segment (transition context)
- **Position features** — normalized position in sentence, position in session (fatigue proxy)

Architecture:
- Character embedding (dim 32) + positional encoding
- WPM embedding (dim 16), broadcast/concatenated
- 2-3 Transformer encoder layers (d_model=64, nhead=4, ff=128)
- Output: sequence of encoder hidden states the decoder cross-attends to

#### Decoder (generates keystroke sequence)

Each decoder token represents one keystroke: a `(char_id, delay)` pair.

Inputs at each step:
- **Previous keystroke char_id** — what was just typed (embedding, dim 32)
- **Previous delay** — LogNormal-sampled delay from previous step (scalar, projected)

Architecture:
- 2-3 Transformer decoder layers (d_model=64, nhead=4, ff=128)
- Causal self-attention (sees own previous outputs)
- Cross-attention to encoder output (sees which target characters remain)
- Two output projections per step:
  - **char_id** — softmax over 98 classes (PAD=0, ASCII 1-95, BACKSPACE=96, END=97)
  - **delay** — 2 values: mu and log_sigma for LogNormal distribution

The `<END>` token (id=97) terminates generation for this segment.

#### Size Budget

| Component | Parameters (est.) |
|---|---|
| Character embedding (98 × 32) | 3.1K |
| WPM embedding (10 × 16) | 160 |
| Encoder (2 layers, d=64, ff=128) | ~100K |
| Decoder (2 layers, d=64, ff=128) | ~130K |
| Output heads | ~10K |
| **Total** | **~250K – 500K** |

Comparable to or smaller than the current model. Can scale up to ~1-2M if needed.

### Training

#### Data Reprocessing Pipeline

1. **Load** Aalto raw keystroke files (same source data)
2. **Replay** each session against its target sentence, classifying keystrokes
3. **Segment** by natural pauses (300ms+ IKI threshold)
4. **For each segment, extract:**
   - Target text (the substring of the target sentence this segment covers)
   - Full keystroke sequence: `[(char_id, delay_ms), ...]` including errors and corrections
   - Context: WPM of session, segment index, session length, previous segment's last 4 keystrokes
5. **Filter:** discard segments shorter than 3 keystrokes or longer than 64
6. **Save** as compressed NumPy archives or TFRecords

Expected yield: ~5-10M segment-level examples from 136M keystrokes.

#### Loss Function

Combined loss per decoder step:

```
L = L_char + λ_timing * L_timing
```

- **L_char:** Sparse categorical cross-entropy on char_id prediction
- **L_timing:** LogNormal negative log-likelihood on delay prediction (same as current model, proven to work)
- **λ_timing:** 1.0 (tunable)

No separate action loss or error_char loss — the model learns correct/error/backspace behavior implicitly through the character prediction. A backspace is just char_id=96 in the output sequence.

No sample weighting or masking needed. Every token in every sequence contributes equally.

#### Training Configuration

- **Teacher forcing** during training (standard seq2seq)
- **Batch size:** 256-512 segments
- **Optimizer:** AdamW, lr=3e-4 with cosine decay
- **Epochs:** 30 with early stopping (patience 7)
- **Max decoder length:** 64 tokens (covers segments up to ~40 keystrokes + errors)
- **Max encoder length:** 32 characters (target text segment)

### Inference

#### Segment Splitting

Given input text "The quick brown fox jumps over the lazy dog.", split into segments:
- Heuristic: every 2-4 words (randomized for variety)
- Or: sample segment lengths from the pause distribution learned from training data

#### Autoregressive Generation

For each segment:
1. Encode target text + context features
2. Start decoder with `<START>` token
3. At each step:
   - Predict char_id distribution → sample with temperature
   - Predict delay distribution → sample from LogNormal
   - Feed sampled char_id + delay as next decoder input
4. Stop when `<END>` token is generated or max length reached
5. Pass last 4 keystrokes as context to next segment

#### Inter-Segment Pauses

The delay on the first keystroke of each segment naturally captures the pause between segments — the model learns these from the 300ms+ gaps in training data.

#### Live Preview

Same real-time playback as `live_preview.py`: iterate through the generated sequence, `time.sleep(delay_ms / 1000)`, render each character to terminal. Errors appear and get corrected as the model decided — no hardcoded correction logic.

### What This Solves

| Current Model Problem | Phrase-Level Solution |
|---|---|
| Backspace prediction 66.7%, unusable | Corrections are part of the generated sequence, learned end-to-end |
| Correction strategy hardcoded | Model learns real correction patterns (immediate, delayed, word-end, never) |
| Timing/action predicted independently | Delay is per-token, naturally conditioned on whether it's an error or correction |
| No cross-word errors | Segments span multiple words |
| 117ms timing MAE | Timing conditioned on richer context (full target text + self-attention over generated sequence) |
| WPM bucket is crude persona signal | Same WPM conditioning, but applied to coherent phrase-level generation |

### Risks and Mitigations

**Risk: Variable-length generation is harder to train than fixed-output classification.**
Mitigation: Teacher forcing is well-understood for seq2seq. The sequences are short (15-40 tokens). Start with a small model and scale up if needed.

**Risk: The model might generate sequences that don't reconstruct to the target text.**
Mitigation: Post-generation verification (same as current `keystroke-stream.ts` verify). If a segment doesn't reconstruct correctly, regenerate it. Track verification failure rate as a quality metric.

**Risk: 300ms pause threshold might not be the right segmentation boundary.**
Mitigation: Make it configurable. Try 200ms, 300ms, 500ms and compare. Also experiment with word-boundary-aligned segmentation as a fallback.

**Risk: Small Transformer might not have enough capacity.**
Mitigation: Start at ~250K params. The task is simpler than language modeling — small vocabulary (98 tokens), short sequences, highly structured output. Scale to 1-2M if validation loss plateaus.

### File Structure

```
ml/
├── v2/                          # New model version
│   ├── config.py                # V2 configuration
│   ├── segment.py               # Segmentation logic (pause-based splitting)
│   ├── preprocess.py            # Reprocess Aalto data into segment examples
│   ├── dataset.py               # TF dataset pipeline for segments
│   ├── model.py                 # Transformer encoder-decoder architecture
│   ├── train.py                 # Training loop
│   ├── generate.py              # Inference / sequence generation
│   ├── evaluate.py              # Evaluation metrics
│   ├── live_preview.py          # Real-time terminal demo
│   └── export.py                # Export to TF.js for Chrome extension
├── docs/plans/
│   └── 2025-02-25-phrase-level-typing-model-design.md  # This document
```

All V2 code lives in `ml/v2/` to avoid disturbing the working V1 model.
