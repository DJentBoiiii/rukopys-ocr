#!/usr/bin/env bash
# Runs the full overnight training pipeline (pretrain -> finetune -> evaluate)
# under a hard OS-level time budget, so it cannot run past the available
# overnight window even if a stage misbehaves.
#
# Usage:
#   timeout 5h bash scripts/run_overnight.sh
#
# Each stage also checkpoints its own progress (see training/config.py),
# so if the overall timeout fires mid-stage, the last checkpoint written
# so far can still be used/resumed from.
set -euo pipefail
cd "$(dirname "$0")/.."

source .venv/bin/activate

LOG_DIR=experiments/logs
mkdir -p "$LOG_DIR"

{
    echo "=== overnight run started at $(date -Iseconds) ==="
    python -m training.train --stage pretrain
    python -m training.train --stage finetune
    python -m training.evaluate
    echo "=== overnight run finished at $(date -Iseconds) ==="
} 2>&1 | tee -a "$LOG_DIR/overnight.log"
