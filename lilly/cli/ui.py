"""Lilly CLI UI — shared terminal abstraction, rainbow banner, and progress display.

All ANSI escape codes and terminal rendering logic lives here.
Pipeline scripts import from this module; library modules stay UI-free.
"""

from __future__ import annotations

import colorsys
import os
import platform
import subprocess
import sys
import threading
import time

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

# Lines printed by print_banner() below the banner text:
#   blank + subtitle + platform + blank = 4
_ROWS_AFTER_BANNER_TEXT = 4


class BannerAnimator:
    """Background thread that redraws the banner with a flowing rainbow.

    Tracks how many terminal rows lie between the banner and the cursor,
    then uses cursor-up/down to navigate, redraw, and return.
    """

    def __init__(self) -> None:
        self._phase = 0.0
        self._running = False
        self._thread: threading.Thread | None = None
        self.lock = threading.Lock()
        self._content_rows = 0  # permanent lines printed after banner area
        self._live_rows = 0     # 0 or 2 (ProgressUI live lines)

    def start(self) -> None:
        if not _ANSI:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.5)

    def add_rows(self, n: int) -> None:
        self._content_rows += n

    def set_live(self, n: int) -> None:
        self._live_rows = n

    def _loop(self) -> None:
        while self._running:
            with self.lock:
                self._redraw()
            time.sleep(0.08)
            self._phase += 0.03

    def _redraw(self) -> None:
        if not _ANSI:
            return
        up = (
            _ROWS_AFTER_BANNER_TEXT
            + _BANNER_TEXT_COUNT
            + self._content_rows
            + self._live_rows
        )

        buf: list[str] = [t.up(up)]
        for i, line in enumerate(_BANNER_LINES):
            colored = _rainbow_text(line, self._phase)
            buf.append(f"\r{t.erase_line()}  {t.BOLD}{colored}{t.RESET}")
            if i < _BANNER_TEXT_COUNT - 1:
                buf.append(t.down(1))

        remaining = up - (_BANNER_TEXT_COUNT - 1)
        buf.append(f"\r{t.down(remaining)}")

        sys.stdout.write("".join(buf))
        sys.stdout.flush()


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
        frac = completed / self.total
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

    def update(self, detail: str) -> None:
        with self._lock:
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


# ── Banner Printing ──────────────────────────────────────────────────────

def print_banner() -> None:
    """Print the LILLY ASCII art watermark with an initial rainbow."""
    print()
    for line in _BANNER_LINES:
        colored = _rainbow_text(line) if _ANSI else line
        print(f"  {t.BOLD}{colored}{t.RESET}")
    print()
    print(f"  {t.DIM}ML Typing Behavior Model (V3){t.RESET}")
    print(f"  {t.DIM}Platform: {platform.system()} {platform.machine()}{t.RESET}")
    print()
