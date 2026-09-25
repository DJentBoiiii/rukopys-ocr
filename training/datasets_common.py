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
MAX_WIDTH = 384


def to_grayscale_array(image) -> np.ndarray:
    """Convert a PIL image or numpy array to a 2D grayscale uint8 array."""
    arr = np.array(image)
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
        else:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return arr.astype(np.uint8)


def resize_to_fixed_height(image: np.ndarray, target_height: int = IMG_HEIGHT, max_width: int = MAX_WIDTH) -> np.ndarray:
    """Resize a grayscale line image to a fixed height, preserving aspect ratio,
    then pad (white) or crop the width to `max_width`.
    """
    h, w = image.shape
    if h == 0 or w == 0:
        return np.full((target_height, max_width), 255, dtype=np.uint8)

    new_w = max(1, round(w * target_height / h))
    resized = cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_AREA)

    if new_w >= max_width:
        return resized[:, :max_width]

    padded = np.full((target_height, max_width), 255, dtype=np.uint8)
    padded[:, :new_w] = resized
    return padded


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
        original_width = min(image.shape[1] * IMG_HEIGHT // max(image.shape[0], 1), MAX_WIDTH)
        tensor = normalize_image(resized)
        target = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        return tensor, target, original_width


def collate_batch(batch: List[Tuple[torch.Tensor, torch.Tensor, int]]):
    images = torch.stack([b[0] for b in batch])  # (B, 1, H, MAX_WIDTH)
    targets = torch.cat([b[1] for b in batch])
    target_lengths = torch.tensor([len(b[1]) for b in batch], dtype=torch.long)

    max_t = compute_output_seq_length(MAX_WIDTH)
    input_lengths = torch.tensor(
        [max(1, min(compute_output_seq_length(b[2]), max_t)) for b in batch],
        dtype=torch.long,
    )
    return images, targets, input_lengths, target_lengths
