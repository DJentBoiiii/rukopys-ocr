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
# GPU finetune run (see experiments/gpu_benchmark.md): separate path so this
# never overwrites the confirmed CPU baseline (finetune.pt) or the stale,
# unconfirmed CPU v2 attempt (finetune_v2.pt).
FINETUNE_CHECKPOINT_GPU = CHECKPOINT_DIR / "finetune_gpu.pt"

# CPU-only, one overnight run (~5h) budget: kept conservative so the full
# pipeline is guaranteed to finish rather than risk an overnight timeout.
PRETRAIN_N_EXAMPLES = 15000
PRETRAIN_EPOCHS = 2
PRETRAIN_BATCH_SIZE = 16
PRETRAIN_LR = 1e-3

# GPU finetune budget (RTX 3050 Mobile), derived from measured throughput in
# experiments/gpu_benchmark.md: full real-data pool (21227 non-val handwritten
# train regions), epoch count sized to the remaining time budget with margin.
FINETUNE_MAX_EXAMPLES = 21227
FINETUNE_EPOCHS = 150
FINETUNE_BATCH_SIZE = 16
FINETUNE_LR = 3e-4

CHECKPOINT_EVERY_STEPS = 200

MODEL_IMG_HEIGHT = 32
MODEL_CNN_CHANNELS = 64
MODEL_RNN_HIDDEN = 256
MODEL_RNN_LAYERS = 2

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
