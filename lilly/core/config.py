"""Shared configuration for the Lilly typing model pipeline.

Contains path definitions, encoding constants, preprocessing parameters,
WPM buckets, and V3 model/training configuration dataclasses.
"""

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TFRECORD_DIR = DATA_DIR / "tfrecords"
V3_SEGMENT_DIR = DATA_DIR / "v3_segments"
MODEL_DIR = PROJECT_ROOT / "models"
V3_MODEL_DIR = MODEL_DIR / "v3"
EXPORT_DIR = PROJECT_ROOT / "export"
V3_EXPORT_DIR = EXPORT_DIR / "v3"

DATASET_URL = (
    "https://userinterfaces.aalto.fi/136Mkeystrokes/data/Keystrokes.zip"
)
DATASET_ZIP = RAW_DIR / "Keystrokes.zip"

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
MIN_KEYSTROKES_PER_SESSION = 20
MAX_KEYSTROKES_PER_SESSION = 5000
MODIFIER_KEYS = frozenset({"SHIFT", "CAPS_LOCK", "CTRL", "ALT", "TAB"})
BACKSPACE_KEY = "BKSP"

# ---------------------------------------------------------------------------
# Character encoding
# ---------------------------------------------------------------------------
# 0 = PAD, 1..95 = printable ASCII (space 0x20 .. tilde 0x7E), 96 = BACKSPACE
PAD_TOKEN = 0
BACKSPACE_TOKEN = 96
ASCII_OFFSET = 31  # char_id = ord(c) - 31 for printable chars (gives 1..95)

# V2/V3: adds END (97) and START (98) tokens = 99 classes
END_TOKEN = 97
START_TOKEN = 98
NUM_CHAR_CLASSES_V2 = 99

# Legacy V1 constant (kept for encoding module compatibility)
NUM_CHAR_CLASSES_V1 = 97

# ---------------------------------------------------------------------------
# Feature extraction (legacy, used by preprocessing)
# ---------------------------------------------------------------------------
SEQ_LEN = 32  # context window size
NUM_DENSE_FEATURES = 14

MAX_IKI_MS = 5000.0
MIN_IKI_MS = 10.0
MAX_HOLD_MS = 2000.0
MIN_HOLD_MS = 5.0
ROLLING_SPEED_WINDOW = 10
ROLLING_ERROR_WINDOW = 20
NORMALIZATION_CAP = 50

# ---------------------------------------------------------------------------
# WPM persona buckets
# ---------------------------------------------------------------------------
WPM_BUCKET_EDGES = [0, 30, 45, 60, 75, 90, 105, 120, 140, 170, 999]
NUM_WPM_BUCKETS = len(WPM_BUCKET_EDGES) - 1  # 10

# ---------------------------------------------------------------------------
# Action labels
# ---------------------------------------------------------------------------
ACTION_CORRECT = 0
ACTION_ERROR = 1
ACTION_BACKSPACE = 2
NUM_ACTIONS = 3

# ---------------------------------------------------------------------------
# V3 Segmentation
# ---------------------------------------------------------------------------
PAUSE_THRESHOLD_MS = 300.0
MIN_SEGMENT_KEYSTROKES = 3
MAX_SEGMENT_KEYSTROKES = 80  # increased from 64
MAX_TARGET_CHARS = 32
STYLE_DIM = 16

# ---------------------------------------------------------------------------
# V3 Model & Training Config
# ---------------------------------------------------------------------------
@dataclass
class V3ModelConfig:
    max_encoder_len: int = MAX_TARGET_CHARS
    max_decoder_len: int = MAX_SEGMENT_KEYSTROKES
    num_char_classes: int = NUM_CHAR_CLASSES_V2  # 99
    char_embed_dim: int = 48
    action_embed_dim: int = 16
    delay_embed_dim: int = 16
    d_model: int = 128
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1
    style_dim: int = STYLE_DIM
    mdn_components: int = 8
    num_actions: int = NUM_ACTIONS  # 3
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
