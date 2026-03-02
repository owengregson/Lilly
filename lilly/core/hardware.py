"""Hardware detection and resource allocation for pipeline scripts.

Auto-detects optimal worker/thread counts based on CPU cores and available
memory.  All scripts import from here instead of rolling their own detection.

Pure library module — no prints, no CLI.
"""

from __future__ import annotations

import os
import platform


def cpu_count() -> int:
    """Return the number of logical CPU cores (fallback: 4)."""
    return os.cpu_count() or 4


def available_ram_gb() -> float:
    """Return total physical RAM in GB, or 0.0 on failure.

    Uses OS-level APIs — no external dependencies.
    """
    s = platform.system()
    try:
        if s == "Darwin":
            import subprocess

            raw = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if raw.returncode == 0 and raw.stdout.strip():
                return int(raw.stdout.strip()) / (1024**3)
        elif s == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        return int(line.split()[1]) / (1024**2)
        elif s == "Windows":
            import subprocess

            raw = subprocess.run(
                ["wmic", "ComputerSystem", "get", "TotalPhysicalMemory", "/value"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if raw.returncode == 0 and raw.stdout.strip():
                for line in raw.stdout.strip().split("\n"):
                    if "=" in line:
                        return int(line.split("=", 1)[1].strip()) / (1024**3)
    except (OSError, ValueError, FileNotFoundError):
        pass
    return 0.0


def detect_workers(*, reserve: int = 1) -> int:
    """Auto-detect optimal worker count for CPU-bound parallel tasks.

    Reserves ``reserve`` cores for the main process, UI animation, and
    system overhead.  Clamped to [1, 128].

    Parameters
    ----------
    reserve:
        Number of cores to leave idle.  Default 1 (main process + UI).
    """
    cores = cpu_count()
    workers = max(1, cores - reserve)
    return min(workers, 128)
