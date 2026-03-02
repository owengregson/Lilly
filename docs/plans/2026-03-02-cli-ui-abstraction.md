# CLI UI Abstraction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract the Lilly CLI UI (rainbow banner, animated progress, Term abstraction) into `lilly/cli/ui.py` and rewrite all pipeline scripts to use it.

**Architecture:** Shared `lilly/cli/ui.py` module contains Term, rainbow, BannerAnimator, ProgressUI. Pipeline scripts import from it normally. `setup_project.py` uses `sys.path.insert` to import before lilly is installed. Library modules (`lilly.data.*`, `lilly.training.*`) become pure logic — scripts own the CLI/UI layer.

**Tech Stack:** Python stdlib only (colorsys, threading, subprocess). No external dependencies for the UI.

---

### Task 1: Create `lilly/cli/ui.py` — extract shared UI module

**Files:**
- Create: `lilly/cli/__init__.py`
- Create: `lilly/cli/ui.py`

**Step 1: Create the package init**

```python
# lilly/cli/__init__.py
```

Empty file, just makes `lilly.cli` a package.

**Step 2: Create `lilly/cli/ui.py`**

Extract from `setup_project.py` (lines 29-300) into `lilly/cli/ui.py`:
- `_ANSI` detection (with Windows `os.system("")` trick)
- `Term` class with all ANSI codes
- `t` alias
- `_rainbow_text(text, phase)` using `colorsys`
- `BANNER`, `_BANNER_LINES`, `_BANNER_TEXT_COUNT`, `_ROWS_AFTER_BANNER_TEXT`
- `BannerAnimator` class
- `_DummyLock` class + `_DUMMY_LOCK` singleton
- `ProgressUI` class (full implementation with `run_cmd`, row tracking, lock)
- `print_banner()` function

The module should be a pure extraction — identical logic, just in a new file. All imports at the top: `colorsys, os, platform, subprocess, sys, threading, time`.

**Step 3: Verify import works**

Run: `python -c "from lilly.cli.ui import Term, t, BannerAnimator, ProgressUI, print_banner; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add lilly/cli/__init__.py lilly/cli/ui.py
git commit -m "refactor: extract shared CLI UI module to lilly/cli/ui"
```

---

### Task 2: Rewire `setup_project.py` to import from `lilly/cli/ui.py`

**Files:**
- Modify: `setup_project.py`

**Step 1: Replace embedded UI code with imports**

Remove from `setup_project.py`:
- The `_ANSI` detection block (lines 29-35)
- `Term` class (lines 38-68)
- `t` alias (line 71)
- `_rainbow_text` function (lines 76-97)
- `BANNER` constant and derived constants (lines 102-113)
- `BannerAnimator` class (lines 116-176)
- `_DummyLock` class and singleton (lines 229-239)
- `ProgressUI` class (lines 244-349)

Replace with:

```python
# Import shared UI before lilly is installed — direct file import via sys.path
sys.path.insert(0, str(Path(__file__).parent))
from lilly.cli.ui import (  # noqa: E402
    BannerAnimator,
    ProgressUI,
    Term,
    _ANSI,
    print_banner,
    t,
)
```

Keep everything else in `setup_project.py`: state detection functions, setup step functions, `get_venv_python()`, `get_venv_activate_hint()`, `print_next_steps()`, `main()`.

**Step 2: Verify setup script still works**

Run: `python3 setup_project.py --help`
Expected: Shows help text without errors.

Run: `python3 setup_project.py`
Expected: Runs to completion with rainbow banner, skips already-done steps.

**Step 3: Commit**

```bash
git add setup_project.py
git commit -m "refactor: setup_project imports UI from lilly.cli.ui"
```

---

### Task 3: Rewrite `scripts/download.py`

**Files:**
- Modify: `scripts/download.py`
- Modify: `lilly/data/download.py` (remove `main()`, remove prints from `download()`/`extract()`/`verify()`)

**Step 1: Make `lilly/data/download.py` a pure library**

Remove the `main()` function. Remove all `print()` calls from `download()`, `extract()`, and `verify()`. Instead:

- `download(url, dest, chunk_size, progress_callback=None)` — calls `progress_callback(bytes_downloaded, total_bytes)` if provided. Returns `{"status": "downloaded"|"skipped", "size_bytes": int}`.
- `extract(zip_path, dest_dir, progress_callback=None)` — calls `progress_callback(files_extracted, total_files)` if provided.
- `verify(data_dir)` — returns `int` (count of files found), no printing.

**Step 2: Rewrite `scripts/download.py`**

Full CLI script with 4 steps:
1. Checking dataset status — check if zip exists, compare size with HEAD request
2. Downloading dataset — stream with progress via `ui.update()`
3. Extracting archive — extract with progress via `ui.update()`
4. Verifying files — count keystroke files

Uses `print_banner()`, `BannerAnimator`, `ProgressUI` from `lilly.cli.ui`.

**Step 3: Verify**

Run: `python scripts/download.py --help`
Expected: Shows argparse help.

**Step 4: Commit**

```bash
git add scripts/download.py lilly/data/download.py
git commit -m "feat: download script uses Lilly UI"
```

---

### Task 4: Rewrite `scripts/preprocess.py`

**Files:**
- Modify: `scripts/preprocess.py`
- Modify: `lilly/data/preprocess.py` (remove `main()`, keep all processing functions)

**Step 1: Make `lilly/data/preprocess.py` a pure library**

Remove `main()`. Keep all functions (`parse_keystroke_file`, `replay_session_arrays`, `compute_wpm_vectorized`, `process_file`, `write_chunk_parquet`, `find_keystroke_files`, `_auto_detect_workers`). Remove all `print()` calls.

**Step 2: Rewrite `scripts/preprocess.py`**

Full CLI script with 3 steps:
1. Scanning for keystroke files — finds files, reports count
2. Processing chunks — uses ProcessPoolExecutor, updates detail line with chunk/file progress
3. Summary — reports total sessions/keystrokes/chunks

argparse flags: `--data-dir`, `--output-dir`, `--max-files`, `--workers`, `--chunk-size`.

**Step 3: Verify**

Run: `python scripts/preprocess.py --help`
Expected: Shows argparse help.

**Step 4: Commit**

```bash
git add scripts/preprocess.py lilly/data/preprocess.py
git commit -m "feat: preprocess script uses Lilly UI"
```

---

### Task 5: Rewrite `scripts/segment_v3.py`

**Files:**
- Modify: `scripts/segment_v3.py`
- Modify: `lilly/data/segment_v3.py` (remove `main()`)

**Step 1: Make `lilly/data/segment_v3.py` a pure library**

Remove `main()`. Keep `extract_v3_segments`, `process_chunk`, and all helper functions. Remove `print()` calls.

**Step 2: Rewrite `scripts/segment_v3.py`**

Full CLI script with 3 steps:
1. Scanning Parquet files — glob for `*.parquet`, report count
2. Extracting V3 segments — loop over parquet files, update detail line with progress
3. Summary — total segments, output path

argparse flags: `--input-dir`, `--output-dir`, `--max-files`.

**Step 3: Verify**

Run: `python scripts/segment_v3.py --help`
Expected: Shows argparse help.

**Step 4: Commit**

```bash
git add scripts/segment_v3.py lilly/data/segment_v3.py
git commit -m "feat: segment_v3 script uses Lilly UI"
```

---

### Task 6: Rewrite `scripts/train.py`

**Files:**
- Modify: `scripts/train.py`
- Modify: `lilly/training/trainer.py` (remove prints, return data instead)

**Step 1: Make `lilly/training/trainer.py` UI-free**

Modify `train()` to accept an optional `progress_callback(epoch, train_loss, val_loss, acc, lr, time_s)` parameter. Remove all `print()` calls. The function still handles the training loop, checkpointing, CSV logging, and metadata saving internally. It returns `(run_dir, metadata_dict)`.

**Step 2: Rewrite `scripts/train.py`**

Full CLI script with 4 steps:
1. Loading datasets — call `build_v3_datasets`, report sample count
2. Building model — call `build_model`, report param count
3. Training — call `train()` with callback, update detail line with epoch/loss/accuracy per epoch
4. Saving model — report paths

argparse flags: `--data-dir`, `--model-dir`, `--epochs`, `--batch-size`, `--learning-rate`, `--max-files`, `--run-name`.

**Step 3: Verify**

Run: `python scripts/train.py --help`
Expected: Shows argparse help.

**Step 4: Commit**

```bash
git add scripts/train.py lilly/training/trainer.py
git commit -m "feat: train script uses Lilly UI"
```

---

### Task 7: Rewrite `scripts/evaluate.py`

**Files:**
- Modify: `scripts/evaluate.py`

**Step 1: Rewrite with Lilly UI**

The evaluation script already has all logic inline. Wrap it with the Lilly UI:
- Step 1: Loading model
- Step 2: Loading test data
- Step 3: Tier 1 evaluation (if --tier >= 1)
- Step 4: Tier 2 evaluation (if --tier >= 2)
- Step 5: Tier 3 evaluation (if --tier >= 3)
- Optional step: Visualization (if --visualize)
- Final step: Saving metrics

Dynamic step count based on `--tier` and `--visualize` flags.

**Step 2: Verify**

Run: `python scripts/evaluate.py --help`
Expected: Shows argparse help.

**Step 3: Commit**

```bash
git add scripts/evaluate.py
git commit -m "feat: evaluate script uses Lilly UI"
```

---

### Task 8: Rewrite `scripts/generate.py`

**Files:**
- Modify: `scripts/generate.py`

**Step 1: Rewrite with Lilly UI**

3 steps:
1. Loading model
2. Generating sequence — update detail with "Generating N keystrokes..."
3. Results — print the keystroke sequence and summary stats after `ui.finish()`

Keep all existing logic (argparse, style vector construction, temperature dict). Wrap model loading and generation in UI steps.

**Step 2: Verify**

Run: `python scripts/generate.py --help`
Expected: Shows argparse help.

**Step 3: Commit**

```bash
git add scripts/generate.py
git commit -m "feat: generate script uses Lilly UI"
```

---

### Task 9: Rewrite `scripts/export.py`

**Files:**
- Modify: `scripts/export.py`

**Step 1: Rewrite with Lilly UI**

3 steps:
1. Loading model
2. Converting to TF.js (with quantization info in detail line)
3. Done — report output path

Keep existing argparse and `export_model()` call.

**Step 2: Verify**

Run: `python scripts/export.py --help`
Expected: Shows argparse help.

**Step 3: Commit**

```bash
git add scripts/export.py
git commit -m "feat: export script uses Lilly UI"
```

---

### Task 10: Replace raw ANSI codes in `scripts/live_preview.py`

**Files:**
- Modify: `scripts/live_preview.py`

**Step 1: Replace ANSI constants with Term imports**

`live_preview.py` is not getting the full Lilly UI treatment (per design decision — it's an interactive replay). But it still has raw `\033[...]` escape codes that should use `Term` from the shared module:

Replace the raw constants block:
```python
BOLD = "\033[1m"
DIM = "\033[2m"
# etc.
```

With:
```python
from lilly.cli.ui import Term as t
```

And replace `CURSOR_HIDE`/`CURSOR_SHOW` with `Term` additions (add these to `lilly/cli/ui.py` if not already present).

**Step 2: Verify**

Run: `python scripts/live_preview.py --help`
Expected: Shows argparse help without errors.

**Step 3: Commit**

```bash
git add scripts/live_preview.py lilly/cli/ui.py
git commit -m "refactor: live_preview uses Term from lilly.cli.ui"
```

---

### Task 11: Lint and final verification

**Files:**
- All modified files

**Step 1: Run ruff on all changed files**

Run: `ruff check lilly/cli/ scripts/ setup_project.py`
Expected: All checks passed.

**Step 2: Run test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests pass.

**Step 3: Run setup script to verify end-to-end**

Run: `python3 setup_project.py`
Expected: Rainbow banner animates, steps skip (already installed), finishes cleanly.

**Step 4: Verify each script's --help works**

```bash
for script in download preprocess segment_v3 train evaluate generate export live_preview; do
  python scripts/$script.py --help > /dev/null 2>&1 && echo "OK: $script" || echo "FAIL: $script"
done
```

**Step 5: Final commit**

```bash
git add -A
git commit -m "chore: lint and verify CLI UI abstraction"
```
