"""QWERTY keyboard layout data for distance and neighbor calculations."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Key positions on a standard QWERTY keyboard.
# Coordinates: (row, col) where row 0 = number row, row 3 = bottom row.
# Column spacing is normalized to 1.0 per key width.
# Row offsets account for the physical stagger of keyboard rows.
# ---------------------------------------------------------------------------

_ROW_OFFSETS = {0: 0.0, 1: 0.25, 2: 0.5, 3: 0.75}

# fmt: off
KEY_POSITIONS: Dict[str, Tuple[float, float]] = {
    # Number row (row 0)
    '`': (0, 0.0), '1': (0, 1.0), '2': (0, 2.0), '3': (0, 3.0),
    '4': (0, 4.0), '5': (0, 5.0), '6': (0, 6.0), '7': (0, 7.0),
    '8': (0, 8.0), '9': (0, 9.0), '0': (0, 10.0), '-': (0, 11.0),
    '=': (0, 12.0),
    # Row 1 (QWERTY row)
    'q': (1, 0.25), 'w': (1, 1.25), 'e': (1, 2.25), 'r': (1, 3.25),
    't': (1, 4.25), 'y': (1, 5.25), 'u': (1, 6.25), 'i': (1, 7.25),
    'o': (1, 8.25), 'p': (1, 9.25), '[': (1, 10.25), ']': (1, 11.25),
    '\\': (1, 12.25),
    # Row 2 (home row)
    'a': (2, 0.5), 's': (2, 1.5), 'd': (2, 2.5), 'f': (2, 3.5),
    'g': (2, 4.5), 'h': (2, 5.5), 'j': (2, 6.5), 'k': (2, 7.5),
    'l': (2, 8.5), ';': (2, 9.5), "'": (2, 10.5),
    # Row 3 (bottom row)
    'z': (3, 0.75), 'x': (3, 1.75), 'c': (3, 2.75), 'v': (3, 3.75),
    'b': (3, 4.75), 'n': (3, 5.75), 'm': (3, 6.75), ',': (3, 7.75),
    '.': (3, 8.75), '/': (3, 9.75),
    # Space bar (row 4, centered)
    ' ': (4, 5.0),
}
# fmt: on

# Shifted character -> base key mapping
SHIFT_MAP: Dict[str, str] = {
    '~': '`', '!': '1', '@': '2', '#': '3', '$': '4', '%': '5',
    '^': '6', '&': '7', '*': '8', '(': '9', ')': '0', '_': '-',
    '+': '=', '{': '[', '}': ']', '|': '\\', ':': ';', '"': "'",
    '<': ',', '>': '.', '?': '/',
}

# Uppercase -> lowercase
for _c in range(ord('A'), ord('Z') + 1):
    SHIFT_MAP[chr(_c)] = chr(_c + 32)

# Finger assignments (0=left pinky .. 4=left thumb, 5=right thumb .. 9=right pinky)
FINGER_MAP: Dict[str, int] = {}
_FINGER_COLUMNS = {
    0: [0.0, 0.25, 0.5, 0.75],
    1: [1.0, 1.25, 1.5, 1.75],
    2: [2.0, 2.25, 2.5, 2.75],
    3: [3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75],
    6: [5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75],
    7: [7.0, 7.25, 7.5, 7.75],
    8: [8.0, 8.25, 8.5, 8.75],
    9: [9.0, 9.25, 9.5, 9.75, 10.0, 10.25, 10.5, 11.0, 11.25, 12.0, 12.25],
}
for _finger, _cols in _FINGER_COLUMNS.items():
    for _key, (_r, _c) in KEY_POSITIONS.items():
        if _c in _cols:
            FINGER_MAP[_key] = _finger
FINGER_MAP[' '] = 4  # thumb


@lru_cache(maxsize=4096)
def key_distance(a: str, b: str) -> float:
    """Euclidean distance between two keys on the QWERTY layout.

    Returns 0.0 if either key is unknown, and handles shifted characters
    by mapping to their base key.
    """
    a_base = SHIFT_MAP.get(a, a).lower()
    b_base = SHIFT_MAP.get(b, b).lower()

    pos_a = KEY_POSITIONS.get(a_base)
    pos_b = KEY_POSITIONS.get(b_base)

    if pos_a is None or pos_b is None:
        return 0.0

    ra, ca = pos_a
    rb, cb = pos_b

    ca += _ROW_OFFSETS.get(int(ra), 0.0)
    cb += _ROW_OFFSETS.get(int(rb), 0.0)

    return math.sqrt((ra - rb) ** 2 + (ca - cb) ** 2)


@lru_cache(maxsize=256)
def get_neighbors(key: str, radius: float = 1.6) -> List[str]:
    """Return keys within `radius` of the given key on the QWERTY layout."""
    base = SHIFT_MAP.get(key, key).lower()
    pos = KEY_POSITIONS.get(base)
    if pos is None:
        return []
    return [k for k in KEY_POSITIONS if k != base and key_distance(base, k) <= radius]


def same_finger(a: str, b: str) -> bool:
    """Check if two keys are typed with the same finger."""
    fa = FINGER_MAP.get(SHIFT_MAP.get(a, a).lower())
    fb = FINGER_MAP.get(SHIFT_MAP.get(b, b).lower())
    if fa is None or fb is None:
        return False
    return fa == fb


def same_hand(a: str, b: str) -> bool:
    """Check if two keys are typed with the same hand."""
    fa = FINGER_MAP.get(SHIFT_MAP.get(a, a).lower())
    fb = FINGER_MAP.get(SHIFT_MAP.get(b, b).lower())
    if fa is None or fb is None:
        return False
    return (fa <= 4) == (fb <= 4)
