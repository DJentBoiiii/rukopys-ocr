"""Central configuration for training/evaluation paths and hyperparameters."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "experiments" / "logs"
DATA_CACHE_DIR = PROJECT_ROOT / "data_cache" / "rukopys"
VOCAB_PATH = CHECKPOINT_DIR / "vocab.json"

PRETRAIN_CHECKPOINT = CHECKPOINT_DIR / "pretrain.pt"
FINETUNE_CHECKPOINT = CHECKPOINT_DIR / "finetune.pt"
FINETUNE_CHECKPOINT_V2 = CHECKPOINT_DIR / "finetune_v2.pt"

# CPU-only, one overnight run (~5h) budget: kept conservative so the full
# pipeline is guaranteed to finish rather than risk an overnight timeout.
PRETRAIN_N_EXAMPLES = 15000
PRETRAIN_EPOCHS = 2
PRETRAIN_BATCH_SIZE = 16
PRETRAIN_LR = 1e-3

FINETUNE_MAX_EXAMPLES = 18000
FINETUNE_EPOCHS = 20
FINETUNE_BATCH_SIZE = 16
FINETUNE_LR = 3e-4

CHECKPOINT_EVERY_STEPS = 200

MODEL_IMG_HEIGHT = 32
MODEL_CNN_CHANNELS = 64
MODEL_RNN_HIDDEN = 256
MODEL_RNN_LAYERS = 2

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
