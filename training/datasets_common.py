"""Shared image preprocessing and torch Dataset/collate utilities for line-image OCR training."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from model.crnn import compute_output_seq_length
from model.tokenizer import CharTokenizer

IMG_HEIGHT = 32
# Global safety cap against pathological outliers (measured p99=864 on the
# full train split); actual per-batch padding width is computed dynamically
# in `collate_batch` from the batch's real line widths, not this constant.
HARD_CAP = 800


def to_grayscale_array(image) -> np.ndarray:
    """Convert a PIL image or numpy array to a 2D grayscale uint8 array."""
    arr = np.array(image)
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
        else:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return arr.astype(np.uint8)


def resize_to_fixed_height(image: np.ndarray, target_height: int = IMG_HEIGHT, hard_cap: int = HARD_CAP) -> np.ndarray:
    """Resize a grayscale line image to a fixed height, preserving aspect ratio.

    Width is NOT padded or cropped to a fixed value here -- only capped at
    `hard_cap` as a safeguard against pathological outliers. Per-batch
    padding to a common width happens in `collate_batch`, so different
    batches can use different widths instead of a single global constant.
    """
    h, w = image.shape
    if h == 0 or w == 0:
        return np.full((target_height, 1), 255, dtype=np.uint8)

    new_w = max(1, round(w * target_height / h))
    resized = cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_AREA)

    if new_w > hard_cap:
        resized = resized[:, :hard_cap]
    return resized


def normalize_image(image: np.ndarray) -> torch.Tensor:
    """Convert a (H, W) uint8 grayscale array into a normalized (1, H, W) float tensor."""
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
    tensor = (tensor - 0.5) / 0.5
    return tensor.unsqueeze(0)


class LineImageDataset(Dataset):
    """Dataset of (grayscale line image, transcription) pairs for CTC training.

    `samples` is a sequence of (image_array, text) where image_array is
    already a 2D grayscale numpy array (preprocessing applied upstream).
    """

    def __init__(self, samples: Sequence[Tuple[np.ndarray, str]], tokenizer: CharTokenizer):
        self.samples = samples
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image, text = self.samples[idx]
        resized = resize_to_fixed_height(image)
        tensor = normalize_image(resized)  # (1, H, W) -- W varies per example
        target = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        return tensor, target, resized.shape[1]


# Normalized value of a white pixel (255 / 255 -> 1.0, then (1.0 - 0.5) / 0.5 == 1.0),
# used to pad images to a common per-batch width without reintroducing a fixed constant.
_PAD_VALUE = 1.0


def collate_batch(batch: List[Tuple[torch.Tensor, torch.Tensor, int]]):
    widths = [b[2] for b in batch]
    batch_max_width = min(max(widths), HARD_CAP)

    padded_images = []
    for tensor, _target, width in batch:
        if width < batch_max_width:
            pad = torch.full(
                (tensor.shape[0], tensor.shape[1], batch_max_width - width),
                _PAD_VALUE, dtype=tensor.dtype,
            )
            tensor = torch.cat([tensor, pad], dim=-1)
        elif width > batch_max_width:
            tensor = tensor[:, :, :batch_max_width]
        padded_images.append(tensor)
    images = torch.stack(padded_images)  # (B, 1, H, batch_max_width)

    targets = torch.cat([b[1] for b in batch])
    target_lengths = torch.tensor([len(b[1]) for b in batch], dtype=torch.long)

    max_t = compute_output_seq_length(batch_max_width)
    input_lengths = torch.tensor(
        [max(1, min(compute_output_seq_length(w), max_t)) for w in widths],
        dtype=torch.long,
    )
    return images, targets, input_lengths, target_lengths
