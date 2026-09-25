"""Streaming loader for the pumb-ai/synthetic-cyrillic-large dataset.

The full dataset is ~123GB; we never download it in full. Instead we use
`datasets` streaming mode and materialize only the first N examples into
memory (as small grayscale numpy arrays), which is enough for a CPU-only
overnight pretraining budget.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from datasets import load_dataset

from training.datasets_common import to_grayscale_array

DATASET_NAME = "pumb-ai/synthetic-cyrillic-large"


def load_synthetic_subset(n_examples: int, seed: int = 42) -> List[Tuple[np.ndarray, str]]:
    """Stream the first `n_examples` (image, text) pairs from the synthetic corpus.

    Returns a list of (grayscale numpy image, transcription) tuples.
    """
    ds = load_dataset(DATASET_NAME, split="train", streaming=True)

    samples: List[Tuple[np.ndarray, str]] = []
    for example in ds:
        text = example["txt"].strip()
        if not text:
            continue
        image = to_grayscale_array(example["png"])
        samples.append((image, text))
        if len(samples) >= n_examples:
            break

    return samples
