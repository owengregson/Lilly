"""Lilly CLI UI — shared terminal abstraction, rainbow banner, and progress display.

All ANSI escape codes and terminal rendering logic lives here.
Pipeline scripts import from this module; library modules stay UI-free.
"""

from __future__ import annotations

import colorsys
import functools
import os
import platform
import subprocess
import sys
import multiprocessing as mp
from collections import Counter

# ── Terminal Abstraction ──────────────────────────────────────────────────
# All ANSI escape codes are encapsulated here — no raw codes elsewhere.

if platform.system() == "Windows":
    os.system("")  # Enable ANSI escape codes on Windows 10+

_ANSI = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


class Term:
    """Stdlib terminal colors and cursor control."""

    BOLD = "\033[1m" if _ANSI else ""
    DIM = "\033[2m" if _ANSI else ""
    RESET = "\033[0m" if _ANSI else ""

    RED = "\033[0;31m" if _ANSI else ""
    GREEN = "\033[0;32m" if _ANSI else ""
    YELLOW = "\033[1;33m" if _ANSI else ""
    BLUE = "\033[0;34m" if _ANSI else ""

    CURSOR_HIDE = "\033[?25l" if _ANSI else ""
    CURSOR_SHOW = "\033[?25h" if _ANSI else ""

    @staticmethod
    def rgb(r: int, g: int, b: int) -> str:
        """24-bit foreground color."""
        return f"\033[38;2;{r};{g};{b}m" if _ANSI else ""

    @staticmethod
    def up(n: int = 1) -> str:
        """Move cursor up n lines."""
        return f"\033[{n}A" if _ANSI else ""

    @staticmethod
    def down(n: int = 1) -> str:
        """Move cursor down n lines."""
        return f"\033[{n}B" if _ANSI else ""

    @staticmethod
    def erase_line() -> str:
        """Erase the entire current line."""
        return "\033[2K" if _ANSI else ""


t = Term  # short alias used throughout


# ── Rainbow Rendering (stdlib colorsys) ───────────────────────────────────

def _rainbow_text(text: str, phase: float = 0.0) -> str:
    """Render text with a per-character rainbow gradient.

    Increment ``phase`` over time to animate the flow.
    """
    if not _ANSI:
        return text
    visible = sum(1 for c in text if c not in " \t")
    if visible == 0:
        return text
    parts: list[str] = []
    vi = 0
    for ch in text:
        if ch in (" ", "\t"):
            parts.append(ch)
        else:
            hue = (phase + vi / visible) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 1.0)
            parts.append(f"{t.rgb(int(r * 255), int(g * 255), int(b * 255))}{ch}")
            vi += 1
    parts.append(t.RESET)
    return "".join(parts)


# ── Banner ────────────────────────────────────────────────────────────────

BANNER = r"""
,__,      ,__, ,__,      ,__,  ____    ____
|  |      |  | |  |      |  |  \   \  /   /
|  |      |  | |  |      |  |   \   \/   /
|  |      |  | |  |      |  |    \_    _/
|  `----. |  | |  `----. |  `----. |  |
|_______| |__| |_______| |_______| |__|
"""

_BANNER_LINES = BANNER.strip().split("\n")
_BANNER_TEXT_COUNT = len(_BANNER_LINES)  # 6

# Lines printed by print_banner() below the banner text.
# Updated dynamically by print_banner() based on actual info lines.
_rows_after_banner = 4


def _animation_loop(
    stop_event,
    lock,
    content_rows,
    live_rows,
    rows_after: int,
    banner_lines: list[str],
    banner_count: int,
) -> None:
    """Redraw the banner with flowing rainbow — runs in a dedicated process.

    Fully self-contained: uses raw ANSI codes and inline rainbow logic
    so it works correctly regardless of multiprocessing start method.
    """
    import colorsys
    import sys

    if not (hasattr(sys.stdout, "isatty") and sys.stdout.isatty()):
        return

    phase = 0.0
    while not stop_event.is_set():
        with lock:
            up = rows_after + banner_count + content_rows.value + live_rows.value
            buf: list[str] = [f"\033[{up}A"]
            for i, line in enumerate(banner_lines):
                # Inline rainbow text
                visible = sum(1 for c in line if c not in " \t")
                if visible == 0:
                    colored = line
                else:
                    parts: list[str] = []
                    vi = 0
                    for ch in line:
                        if ch in (" ", "\t"):
                            parts.append(ch)
                        else:
                            hue = (phase + vi / visible) % 1.0
                            r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 1.0)
                            parts.append(
                                f"\033[38;2;{int(r*255)};{int(g*255)};{int(b*255)}m{ch}"
                            )
                            vi += 1
                    parts.append("\033[0m")
                    colored = "".join(parts)

                buf.append(f"\r\033[2K  \033[1m{colored}\033[0m")
                if i < banner_count - 1:
                    buf.append("\033[1B")

            remaining = up - (banner_count - 1)
            buf.append(f"\r\033[{remaining}B")
            sys.stdout.write("".join(buf))
            sys.stdout.flush()

        phase += 0.03
        stop_event.wait(0.08)


class BannerAnimator:
    """Background process that redraws the banner with a flowing rainbow.

    Uses ``multiprocessing.Process`` instead of a thread so the animation
    stays smooth even when the main process is CPU-bound (no GIL contention).
    """

    def __init__(self) -> None:
        self._stop_event = mp.Event()
        self._content_rows = mp.Value("i", 0)
        self._live_rows = mp.Value("i", 0)
        self.lock = mp.Lock()
        self._process: mp.Process | None = None

    def start(self) -> None:
        if not _ANSI:
            return
        self._process = mp.Process(
            target=_animation_loop,
            args=(
                self._stop_event,
                self.lock,
                self._content_rows,
                self._live_rows,
                _rows_after_banner,
                list(_BANNER_LINES),
                _BANNER_TEXT_COUNT,
            ),
            daemon=True,
        )
        self._process.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._process:
            self._process.join(timeout=1.0)
            if self._process.is_alive():
                self._process.terminate()

    def add_rows(self, n: int) -> None:
        with self._content_rows.get_lock():
            self._content_rows.value += n

    def set_live(self, n: int) -> None:
        self._live_rows.value = n


# ── Dummy Lock ─────────────────────────────────────────────────────────────

class _DummyLock:
    """No-op context manager when no animator is active."""
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


_DUMMY_LOCK = _DummyLock()


# ── Progress UI ────────────────────────────────────────────────────────────

class ProgressUI:
    """Manages a 2-line live display: progress bar + latest subprocess output.

    Completed steps print as permanent checkmark lines above the live area.
    Coordinates with BannerAnimator via a shared lock and row tracking.
    """

    def __init__(self, total: int, animator: BannerAnimator | None = None):
        self.total = total
        self.step = 0
        self._live = False
        self._anim = animator
        self._sub_progress = 0.0   # 0.0–1.0 within the current step
        self._current_label = ""   # label of the active step (for bar redraws)
        self.run_cmd_output: list[str] = []

    @property
    def _lock(self):
        return self._anim.lock if self._anim else _DUMMY_LOCK

    @staticmethod
    def _width() -> int:
        try:
            return os.get_terminal_size().columns
        except OSError:
            return 80

    def _write(self, text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    def _track(self, n: int) -> None:
        """Inform the animator about permanent lines added."""
        if self._anim:
            self._anim.add_rows(n)

    def _set_live(self, active: bool) -> None:
        if self._anim:
            self._anim.set_live(2 if active else 0)

    def _make_bar(self) -> str:
        w = self._width()
        bar_w = max(10, min(40, w - 30))
        completed = max(0, self.step - 1)
        frac = (completed + self._sub_progress) / self.total
        filled = int(bar_w * frac)
        rest = bar_w - filled
        if rest > 0:
            seg = "━" * filled + "╺" + "─" * (rest - 1)
        else:
            seg = "━" * bar_w
        bar = f"{t.GREEN}{seg}{t.RESET}" if _ANSI else seg
        pct = int(frac * 100)
        return f"{bar} {pct:>3}%"

    def _erase(self) -> None:
        if not self._live or not _ANSI:
            return
        self._write(f"{t.up(1)}{t.erase_line()}{t.up(1)}{t.erase_line()}\r")
        self._live = False
        self._set_live(False)

    def _draw(self, label: str, detail: str = "") -> None:
        if not _ANSI:
            return
        self._erase()
        w = self._width()
        bar = self._make_bar()
        line1 = f"  {bar}  {t.BLUE}\u25cf{t.RESET} {label}"
        trunc = detail[:w - 10].rstrip() if detail else ""
        line2 = f"     {t.DIM}\u21b3 {trunc}{t.RESET}" if trunc else ""
        self._write(f"{line1}\n{line2}\n")
        self._live = True
        self._set_live(True)

    def _update_detail(self, detail: str) -> None:
        if not _ANSI or not self._live:
            return
        w = self._width()
        trunc = detail[:w - 10].rstrip() if detail else ""
        text = f"     {t.DIM}\u21b3 {trunc}{t.RESET}" if trunc else ""
        self._write(f"{t.up(1)}\r{t.erase_line()}{text}\n")

    # ── Public API ──────────────────────────────────────────────────────────

    def begin(self, label: str) -> None:
        self.step += 1
        self._sub_progress = 0.0
        self._current_label = label
        with self._lock:
            if _ANSI:
                self._draw(label)
            else:
                self._write(f"  [{self.step}/{self.total}] {label}...")

    def skip(self, label: str, detail: str = "") -> None:
        self.step += 1
        with self._lock:
            suffix = f"  {t.DIM}{detail}{t.RESET}" if detail else ""
            icon = f"{t.DIM}\u2713{t.RESET}" if _ANSI else "-"
            self._write(f"  {icon}  {t.DIM}{label}{t.RESET}{suffix}\n")
            self._track(1)

    def update(self, detail: str, progress: float | None = None) -> None:
        with self._lock:
            if progress is not None:
                self._sub_progress = max(0.0, min(1.0, progress))
                # Redraw both bar and detail so the bar advances
                self._erase()
                self._draw(self._current_label, detail)
            else:
                self._update_detail(detail)

    def done(self, label: str, detail: str = "") -> None:
        with self._lock:
            self._erase()
            suffix = f"  {t.DIM}{detail}{t.RESET}" if detail else ""
            self._write(f"  {t.GREEN}\u2713{t.RESET}  {label}{suffix}\n")
            self._track(1)
            if not _ANSI:
                self._write("\n")
                self._track(1)

    def warn(self, label: str, detail: str = "") -> None:
        with self._lock:
            self._erase()
            suffix = f"  {t.DIM}{detail}{t.RESET}" if detail else ""
            self._write(f"  {t.YELLOW}\u26a0{t.RESET}  {label}{suffix}\n")
            self._track(1)
            if not _ANSI:
                self._write("\n")
                self._track(1)

    def fail(self, label: str, detail: str = "") -> None:
        with self._lock:
            self._erase()
            self._write(f"  {t.RED}\u2717{t.RESET}  {label}\n")
            n = 1
            if detail:
                for line in detail.split("\n"):
                    self._write(f"     {line}\n")
                    n += 1
            self._track(n)

    def finish(self) -> None:
        with self._lock:
            self._erase()
            w = self._width()
            bar_w = max(10, min(40, w - 30))
            seg = "━" * bar_w
            bar = f"{t.GREEN}{seg}{t.RESET}" if _ANSI else seg
            self._write(f"  {bar} {t.GREEN}100%{t.RESET}  Done\n")
            self._track(1)

    def run_cmd(self, cmd: list, label: str) -> tuple:
        """Run a subprocess, streaming output to the detail line.

        Returns (exit_code, last_output_line).
        Full output available via self.run_cmd_output after the call.
        """
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except FileNotFoundError:
            self.run_cmd_output = [f"Command not found: {cmd[0]}"]
            return 127, f"Command not found: {cmd[0]}"

        last = ""
        all_lines: list[str] = []
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            stripped = raw_line.rstrip("\n\r")
            parts = stripped.split("\r")
            line = parts[-1].strip()
            if line:
                all_lines.append(line)
                last = line
                self.update(line)

        proc.wait()
        self.run_cmd_output = all_lines
        return proc.returncode, last


# ── System Info (cached) ─────────────────────────────────────────────────


def _run_cmd(*args: str, timeout: int = 3) -> str | None:
    """Run a command, return stripped stdout or None on failure."""
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None
    except Exception:
        return None


def _clean_cpu_name(name: str) -> str:
    """Remove trademark symbols, frequency, and redundant tokens."""
    for tok in ("(R)", "(TM)", "(tm)", " CPU"):
        name = name.replace(tok, "")
    if "@" in name:
        name = name.split("@")[0]
    return " ".join(name.split())


def _get_cpu_name() -> str:
    s = platform.system()
    if s == "Darwin":
        name = _run_cmd("sysctl", "-n", "machdep.cpu.brand_string")
        if name:
            return _clean_cpu_name(name)
    elif s == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return _clean_cpu_name(line.split(":", 1)[1].strip())
        except OSError:
            pass
    elif s == "Windows":
        name = _run_cmd("wmic", "cpu", "get", "name", "/value")
        if name and "=" in name:
            return _clean_cpu_name(name.split("=", 1)[1].strip())
    return platform.processor() or "Unknown"


def _get_cpu_freq(cpu_name: str = "") -> str | None:
    """Return max CPU frequency as a GHz string like '3.6', or None."""
    s = platform.system()
    if s == "Darwin":
        raw = _run_cmd("sysctl", "-n", "hw.cpufrequency_max")
        if raw:
            try:
                return f"{int(raw) / 1e9:.1f}"
            except ValueError:
                pass
    elif s == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "cpu MHz" in line:
                        mhz = float(line.split(":", 1)[1].strip())
                        return f"{mhz / 1000:.1f}"
        except (OSError, ValueError):
            pass
    elif s == "Windows":
        raw = _run_cmd("wmic", "cpu", "get", "MaxClockSpeed", "/value")
        if raw and "=" in raw:
            try:
                return f"{int(raw.split('=', 1)[1].strip()) / 1000:.1f}"
            except ValueError:
                pass
    # Fallback: parse "@ X.XXGHz" from the original brand string
    if "@" in cpu_name:
        for token in cpu_name.split("@")[-1].split():
            try:
                return f"{float(token.replace('GHz', '')):.1f}"
            except ValueError:
                continue
    return None


def _get_gpu_names() -> list[str]:
    """Return a list of detected GPU names."""
    gpus: list[str] = []
    s = platform.system()
    if s == "Darwin":
        raw = _run_cmd("system_profiler", "SPDisplaysDataType", timeout=5)
        if raw:
            for line in raw.split("\n"):
                stripped = line.strip()
                if stripped.startswith("Chipset Model:"):
                    gpus.append(stripped.split(":", 1)[1].strip())
    elif s == "Linux":
        raw = _run_cmd("nvidia-smi", "--query-gpu=name", "--format=csv,noheader", timeout=5)
        if raw:
            for line in raw.split("\n"):
                if line.strip():
                    gpus.append(line.strip())
        if not gpus:
            raw = _run_cmd("lspci", timeout=5)
            if raw:
                for line in raw.split("\n"):
                    if "VGA" in line or "3D controller" in line:
                        parts = line.split(": ", 1)
                        if len(parts) > 1:
                            gpus.append(parts[1].strip())
    elif s == "Windows":
        raw = _run_cmd(
            "wmic", "path", "win32_videocontroller", "get", "name", "/value", timeout=5,
        )
        if raw:
            for line in raw.split("\n"):
                line = line.strip()
                if line.startswith("Name="):
                    gpus.append(line.split("=", 1)[1].strip())
    return gpus


def _get_ram_gb() -> float:
    """Return total physical RAM in GB, or 0.0 on failure."""
    s = platform.system()
    try:
        if s == "Darwin":
            raw = _run_cmd("sysctl", "-n", "hw.memsize")
            if raw:
                return int(raw) / (1024**3)
        elif s == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        return int(line.split()[1]) / (1024**2)
        elif s == "Windows":
            raw = _run_cmd("wmic", "ComputerSystem", "get", "TotalPhysicalMemory", "/value")
            if raw and "=" in raw:
                return int(raw.split("=", 1)[1].strip()) / (1024**3)
    except (OSError, ValueError):
        pass
    return 0.0


@functools.lru_cache(maxsize=1)
def _system_info() -> dict[str, object]:
    """Gather and cache all system diagnostic information."""
    from importlib.metadata import version as _pkg_version

    info: dict[str, object] = {}

    # Package version
    try:
        info["version"] = _pkg_version("lilly")
    except Exception:
        info["version"] = "dev"

    # Python
    v = sys.version_info
    info["python"] = f"{v.major}.{v.minor}.{v.micro}"

    # TensorFlow version (lightweight — no full import)
    try:
        info["tf_version"] = _pkg_version("tensorflow")
    except Exception:
        info["tf_version"] = None

    # TF acceleration (only if already loaded — avoids heavy import)
    tf_accel = None
    if "tensorflow" in sys.modules:
        try:
            _tf = sys.modules["tensorflow"]
            gpu_devs = _tf.config.list_physical_devices("GPU")
            tf_accel = f"{len(gpu_devs)} GPU" if gpu_devs else "CPU"
        except Exception:
            pass
    info["tf_accel"] = tf_accel

    # CPU
    cpu_name = _get_cpu_name()
    info["cpu_name"] = cpu_name
    info["cpu_cores"] = os.cpu_count() or 0
    info["cpu_ghz"] = _get_cpu_freq(cpu_name)

    # GPU
    info["gpus"] = _get_gpu_names()

    # RAM
    info["ram_gb"] = _get_ram_gb()

    return info


def _build_info_lines() -> list[str]:
    """Format cached system info into display lines for the banner."""
    info = _system_info()
    lines: list[str] = []

    # Line 1 — Version
    lines.append(f"Version: {info['version']} (arch v3)")

    # Line 2 — Python | TensorFlow | Platform
    parts = [f"Python {info['python']}"]
    if info["tf_version"]:
        tf_str = f"TensorFlow {info['tf_version']}"
        if info["tf_accel"]:
            tf_str += f" ({info['tf_accel']})"
        parts.append(tf_str)
    else:
        parts.append("TensorFlow not installed")
    parts.append(f"{platform.system()} {platform.machine()}")
    lines.append(" | ".join(parts))

    # Line 3 — CPU
    cpu_parts = [str(info["cpu_name"])]
    cores = info["cpu_cores"]
    ghz = info["cpu_ghz"]
    if cores and ghz:
        cpu_parts.append(f"{cores} cores @ {ghz} GHz")
    elif cores:
        cpu_parts.append(f"{cores} cores")
    lines.append(f"CPU: {' | '.join(cpu_parts)}")

    # Line 4 — GPU
    gpus: list[str] = info["gpus"]  # type: ignore[assignment]
    if gpus:
        counts = Counter(gpus)
        gpu_strs = [
            f"{name} x{count}" if count > 1 else name
            for name, count in counts.items()
        ]
        lines.append(f"GPU: {', '.join(gpu_strs)}")
    else:
        lines.append("GPU: None detected")

    # Line 5 — RAM
    ram: float = info["ram_gb"]  # type: ignore[assignment]
    lines.append(f"RAM: {ram:.0f} GB" if ram > 0 else "RAM: Unknown")

    return lines


# ── Banner Printing ──────────────────────────────────────────────────────


def print_banner() -> None:
    """Print the LILLY ASCII art watermark with an initial rainbow and system info."""
    global _rows_after_banner
    print()
    for line in _BANNER_LINES:
        colored = _rainbow_text(line) if _ANSI else line
        print(f"  {t.BOLD}{colored}{t.RESET}")
    print()
    info_lines = _build_info_lines()
    for iline in info_lines:
        print(f"  {t.DIM}{iline}{t.RESET}")
    print()
    # blank + info lines + blank
    _rows_after_banner = 2 + len(info_lines)
