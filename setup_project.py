#!/usr/bin/env python3
"""Lilly — Cross-Platform Project Setup

One-command setup that works on macOS, Windows, and Linux.
Resumable: re-running skips steps whose outcomes already exist.

Usage:
    python3 setup_project.py              # Full install (core + dev + eval + export)
    python3 setup_project.py --core-only  # Core dependencies only
    python3 setup_project.py --no-venv    # Skip virtual environment creation
    python3 setup_project.py --force      # Force re-run all steps
    python3 setup_project.py --help       # Show help
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path

# Import shared UI before lilly is installed — direct file import via sys.path
sys.path.insert(0, str(Path(__file__).parent))
from lilly.cli.ui import (  # noqa: E402
    BannerAnimator,
    ProgressUI,
    print_banner,
    t,
)

# ── Paths & Constants ──────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.resolve()
VENV_DIR = PROJECT_ROOT / ".venv"

MIN_PYTHON = (3, 10)
MAX_PYTHON = (3, 13)  # exclusive — TensorFlow does not support 3.13+

TOTAL_STEPS = 5


# ── State Detection ────────────────────────────────────────────────────────

def _venv_python_works(py: Path) -> bool:
    """Check if a Python executable exists and can run."""
    if not py.exists():
        return False
    try:
        result = subprocess.run(
            [str(py), "-c", "import sys; print(sys.version_info[:2])"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def _pip_version(python: Path) -> str:
    """Return the pip version string, or '' if pip is unavailable."""
    try:
        result = subprocess.run(
            [str(python), "-m", "pip", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            # "pip 24.0 from /path/to/pip (python 3.10)"
            parts = result.stdout.strip().split()
            if len(parts) >= 2:
                return parts[1]
    except (subprocess.TimeoutExpired, OSError):
        pass
    return ""


def _pip_has_package(python: Path, package: str) -> bool:
    """Check if a package is installed (via pip show — fast, no imports)."""
    try:
        result = subprocess.run(
            [str(python), "-m", "pip", "show", "--quiet", package],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def _lilly_installed(python: Path, core_only: bool) -> bool:
    """Check if lilly is installed with the expected extras."""
    if not _pip_has_package(python, "lilly"):
        return False
    if core_only:
        return True
    for pkg in ("tensorflow", "matplotlib", "pytest"):
        if not _pip_has_package(python, pkg):
            return False
    return True


# ── Helpers ─────────────────────────────────────────────────────────────────

def get_venv_python() -> Path:
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def get_venv_activate_hint() -> str:
    if platform.system() == "Windows":
        return r".venv\Scripts\activate"
    return "source .venv/bin/activate"


# ── Setup Steps ─────────────────────────────────────────────────────────────

def step_check_python(ui: ProgressUI) -> None:
    label = "Checking Python version"
    ui.begin(label)

    v = sys.version_info
    if v < MIN_PYTHON or v >= MAX_PYTHON:
        ui.fail(
            label,
            f"Python {v.major}.{v.minor}.{v.micro} is not supported.\n"
            f"     TensorFlow requires Python >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]} "
            f"and < {MAX_PYTHON[0]}.{MAX_PYTHON[1]}.\n"
            f"     Install a compatible version from https://www.python.org/downloads/\n"
            f"     or use pyenv:  pyenv install 3.10",
        )
        sys.exit(1)

    ui.done(label, f"Python {v.major}.{v.minor}.{v.micro}")


def step_create_venv(ui: ProgressUI, skip: bool, force: bool) -> Path:
    label = "Creating virtual environment"

    if skip:
        ui.begin(label)
        ui.warn("Skipping virtual environment", "--no-venv")
        return Path(sys.executable)

    py = get_venv_python()

    if not force and VENV_DIR.exists() and _venv_python_works(py):
        ui.skip(label, "exists (.venv/)")
        return py

    ui.begin(label)

    if VENV_DIR.exists():
        ui.update("Removing broken .venv...")
        shutil.rmtree(VENV_DIR)

    ui.update("Building .venv...")
    venv.create(str(VENV_DIR), with_pip=True)

    py = get_venv_python()
    if not py.exists():
        ui.fail(label, f"Expected Python at {py} but it does not exist.")
        sys.exit(1)

    ui.done(label, ".venv/")
    return py


def step_upgrade_pip(ui: ProgressUI, python: Path, force: bool) -> None:
    label = "Upgrading pip"

    if not force:
        ver = _pip_version(python)
        if ver:
            ui.skip(label, f"pip {ver}")
            return

    ui.begin(label)

    rc, last = ui.run_cmd(
        [str(python), "-m", "pip", "install", "--upgrade", "pip"],
        label,
    )

    if rc != 0:
        ui.warn(label, "pip upgrade had issues (continuing)")
    else:
        version = ""
        if "Successfully installed" in last:
            for part in last.split():
                if part.startswith("pip-"):
                    version = part.replace("pip-", "")
        ui.done(label, f"pip {version}" if version else "")


def step_install_package(
    ui: ProgressUI, python: Path, core_only: bool, force: bool,
) -> None:
    if core_only:
        label = "Installing Lilly (core only)"
        spec = "."
    else:
        label = "Installing Lilly [all]"
        spec = ".[all]"

    if not force and _lilly_installed(python, core_only):
        ui.skip(label, "already installed")
        return

    ui.begin(label)

    rc, _last = ui.run_cmd(
        [str(python), "-m", "pip", "install", "-e", spec],
        label,
    )

    if rc != 0:
        error_lines = getattr(ui, "run_cmd_output", [])
        tail = error_lines[-30:] if len(error_lines) > 30 else error_lines
        error_detail = "\n".join(tail) if tail else f"exit code {rc}"
        ui.fail(label, f"Installation failed (exit code {rc}):\n\n{error_detail}")
        sys.exit(1)

    ui.done(label)


def step_verify(ui: ProgressUI, python: Path) -> None:
    label = "Verifying installation"
    ui.begin(label)

    checks = [
        ("Core config", (
            "from lilly.core.config import V3ModelConfig; "
            "c = V3ModelConfig(); "
            "print(f'V3ModelConfig(d_model={c.d_model})')"
        )),
        ("Encoding", (
            "from lilly.core.encoding import char_to_id, id_to_char; "
            "assert char_to_id('a') == ord('a') - 31; "
            "print('char_to_id OK')"
        )),
        ("TensorFlow", (
            "import tensorflow as tf; "
            "print(f'TensorFlow {tf.__version__}')"
        )),
    ]

    passed = 0
    issues = []
    for name, code in checks:
        ui.update(f"Checking {name}...")
        try:
            result = subprocess.run(
                [str(python), "-c", code],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                passed += 1
            else:
                issues.append(name)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            issues.append(name)

    if issues:
        ui.warn(label, f"{passed}/{len(checks)} OK  ({', '.join(issues)} had issues)")
    else:
        ui.done(label, f"{passed}/{len(checks)} checks passed")


# ── Post-Setup ──────────────────────────────────────────────────────────────

def print_next_steps(used_venv: bool) -> None:
    print()
    print(f"  {t.GREEN}{t.BOLD}Setup complete!{t.RESET}")
    print()
    print(f"  {t.BOLD}Next steps:{t.RESET}")
    print()

    step = 1
    if used_venv:
        print(f"    {step}. Activate the virtual environment:")
        print(f"       {t.YELLOW}{get_venv_activate_hint()}{t.RESET}")
        print()
        step += 1

    steps = [
        ("Download the dataset (~15 GB)", "python scripts/download.py"),
        ("Preprocess raw keystroke data", "python scripts/preprocess.py"),
        ("Extract V3 training segments", "python scripts/segment_v3.py"),
        ("Train the model", "python scripts/train.py --epochs 50"),
    ]

    for desc, cmd in steps:
        print(f"    {step}. {desc}:")
        print(f"       {t.YELLOW}{cmd}{t.RESET}")
        print()
        step += 1

    print("    Or run the full pipeline:")
    print(f"       {t.YELLOW}make pipeline{t.RESET}")
    print()


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lilly \u2014 Cross-Platform Project Setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--no-venv", action="store_true",
        help="Skip virtual environment creation (use current Python)",
    )
    parser.add_argument(
        "--core-only", action="store_true",
        help="Install only core dependencies (skip dev/eval/export)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-run all steps (ignore existing state)",
    )
    args = parser.parse_args()

    print_banner()

    if not (PROJECT_ROOT / "pyproject.toml").exists():
        print(f"  {t.RED}\u2717{t.RESET}  pyproject.toml not found. "
              f"Run this script from the Lilly project root.")
        sys.exit(1)

    os.chdir(PROJECT_ROOT)

    animator = BannerAnimator()
    ui = ProgressUI(TOTAL_STEPS, animator)

    animator.start()

    step_check_python(ui)
    python = step_create_venv(ui, skip=args.no_venv, force=args.force)
    step_upgrade_pip(ui, python, force=args.force)
    step_install_package(ui, python, core_only=args.core_only, force=args.force)
    step_verify(ui, python)

    animator.stop()

    ui.finish()
    print_next_steps(used_venv=not args.no_venv)


if __name__ == "__main__":
    main()
