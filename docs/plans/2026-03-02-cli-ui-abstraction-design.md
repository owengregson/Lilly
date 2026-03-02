# CLI UI Abstraction Design

**Date:** 2026-03-02
**Status:** Approved

## Goal

Extract the Lilly CLI UI (rainbow banner, animated progress, Term abstraction) from `setup_project.py` into a shared module at `lilly/cli/ui.py`. All pipeline scripts adopt the same visual identity: animated rainbow banner, step-based progress bars, and colored output.

## Architecture

### Shared module: `lilly/cli/ui.py`

Single source of truth for all UI code. Contains:

- `Term` — ANSI escape code encapsulation (colors, cursor, styles)
- `t` — short alias for `Term`
- `_rainbow_text(text, phase)` — stdlib `colorsys` rainbow rendering
- `BANNER`, `_BANNER_LINES`, `_BANNER_TEXT_COUNT` — ASCII art constants
- `BannerAnimator` — daemon thread for flowing rainbow animation
- `ProgressUI` — 2-line live display with step tracking, lock coordination
- `_DummyLock` — no-op context manager fallback
- `print_banner()` — renders the initial rainbow banner

### Import strategy

- **Pipeline scripts** (`scripts/*.py`): import via `from lilly.cli.ui import ...` (lilly is installed)
- **`setup_project.py`**: uses `sys.path.insert(0, project_root)` to import before lilly is installed

### Script pattern

Every script follows:

```python
from lilly.cli.ui import BannerAnimator, ProgressUI, print_banner

def main():
    args = parse_args()
    print_banner()
    animator = BannerAnimator()
    ui = ProgressUI(N, animator)
    animator.start()

    # step 1...
    # step 2...

    animator.stop()
    ui.finish()
```

## Scripts and their steps

### download.py (4 steps)
1. Checking dataset status
2. Downloading dataset
3. Extracting archive
4. Verifying files

### preprocess.py (3 steps)
1. Scanning keystroke files
2. Processing chunks
3. Summary

### segment_v3.py (3 steps)
1. Scanning Parquet files
2. Extracting V3 segments
3. Summary

### train.py (4 steps)
1. Loading datasets
2. Building model
3. Training epochs
4. Saving model

### evaluate.py (dynamic steps based on --tier)
1. Loading model
2. Loading test data
3-5. Tier 1/2/3 evaluation
Optional: Visualization

### generate.py (3 steps)
1. Loading model
2. Generating sequence
3. Results

### export.py (3 steps)
1. Loading model
2. Converting to TF.js
3. Done

### live_preview.py — unchanged (interactive replay, different UX)

## Key decisions

1. Library modules (`lilly.data.*`, `lilly.training.*`) become pure logic — no print statements. Scripts own the CLI layer.
2. For subprocess operations, `ProgressUI.run_cmd()` streams output to the detail line.
3. For in-process loops, step functions call `ui.update()` manually.
4. `setup_project.py` keeps its unique resumability/state-detection logic but imports UI from the shared module.
