"""Shared configuration for the Lilly typing model pipeline.

Contains path definitions, encoding constants, preprocessing parameters,
WPM buckets, and version-specific model/training configuration dataclasses.
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
V2_SEGMENT_DIR = DATA_DIR / "v2_segments"
MODEL_DIR = PROJECT_ROOT / "models"
V2_MODEL_DIR = MODEL_DIR / "v2"
EXPORT_DIR = PROJECT_ROOT / "export"
V2_EXPORT_DIR = EXPORT_DIR / "v2"

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

# V1: 97 classes (0-96)
NUM_CHAR_CLASSES_V1 = 97

# V2: adds END (97) and START (98) tokens = 99 classes
END_TOKEN = 97
START_TOKEN = 98
NUM_CHAR_CLASSES_V2 = 99

# ---------------------------------------------------------------------------
# Feature extraction (V1)
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
# Segmentation (V2)
# ---------------------------------------------------------------------------
PAUSE_THRESHOLD_MS = 300.0
MIN_SEGMENT_KEYSTROKES = 3
MAX_SEGMENT_KEYSTROKES = 64
MAX_TARGET_CHARS = 32

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
# V1 Model & Training Config
# ---------------------------------------------------------------------------
@dataclass
class V1ModelConfig:
    seq_len: int = SEQ_LEN
    num_char_classes: int = NUM_CHAR_CLASSES_V1
    num_wpm_buckets: int = NUM_WPM_BUCKETS
    num_dense_features: int = NUM_DENSE_FEATURES
    num_actions: int = NUM_ACTIONS
    char_embed_dim: int = 16
    wpm_embed_dim: int = 16
    action_embed_dim: int = 8
    lstm_units_1: int = 128
    lstm_units_2: int = 64
    dense_units: int = 64
    head_units: int = 32
    dropout: float = 0.2


@dataclass
class V1TrainConfig:
    batch_size: int = 512
    epochs: int = 30
    learning_rate: float = 1e-3
    lr_decay_patience: int = 3
    lr_decay_factor: float = 0.5
    early_stop_patience: int = 7
    val_split: float = 0.1
    test_split: float = 0.05
    timing_loss_weight: float = 1.0
    action_loss_weight: float = 2.0
    error_char_loss_weight: float = 0.5
    shuffle_buffer: int = 100_000
    seed: int = 42
    max_samples: int = 0  # 0 = all


# ---------------------------------------------------------------------------
# V2 Model & Training Config
# ---------------------------------------------------------------------------
@dataclass
class V2ModelConfig:
    num_char_classes: int = NUM_CHAR_CLASSES_V2
    num_wpm_buckets: int = NUM_WPM_BUCKETS
    max_encoder_len: int = MAX_TARGET_CHARS
    max_decoder_len: int = MAX_SEGMENT_KEYSTROKES
    char_embed_dim: int = 32
    wpm_embed_dim: int = 16
    d_model: int = 64
    nhead: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    context_tail_len: int = 4


@dataclass
class V2TrainConfig:
    batch_size: int = 256
    epochs: int = 30
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    lr_warmup_steps: int = 1000
    lr_decay_patience: int = 3
    lr_decay_factor: float = 0.5
    early_stop_patience: int = 7
    val_split: float = 0.1
    test_split: float = 0.05
    timing_loss_weight: float = 1.0
    char_loss_weight: float = 1.0
    shuffle_buffer: int = 100_000
    seed: int = 42
    max_samples: int = 0
