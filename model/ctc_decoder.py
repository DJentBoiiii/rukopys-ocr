"""Greedy (best-path) CTC decoder with per-character confidence scores."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from model.tokenizer import CharTokenizer


@dataclass
class Decoded:
    text: str
    confidences: List[float]  # per-output-character confidence, aligned with `text`


class CTCDecoder:
    """Greedy CTC decoder: takes the argmax at each timestep, collapses
    repeated symbols, and removes blanks. Confidence per decoded character
    is the softmax probability of the chosen class, averaged over the
    (possibly repeated) timesteps that collapsed into that character.
    """

    def __init__(self, tokenizer: CharTokenizer):
        self.tokenizer = tokenizer

    def decode(self, log_probs: torch.Tensor) -> Decoded:
        """Args: log_probs of shape (T, vocab_size) for a single sample."""
        probs = log_probs.exp()
        best_ids = probs.argmax(dim=-1)  # (T,)
        best_probs = probs.gather(-1, best_ids.unsqueeze(-1)).squeeze(-1)  # (T,)

        chars: List[str] = []
        confidences: List[float] = []

        prev_id = None
        run_probs: List[float] = []

        def flush():
            if prev_id is not None and prev_id != self.tokenizer.blank_id and run_probs:
                chars.append(self.tokenizer.itos[prev_id])
                confidences.append(sum(run_probs) / len(run_probs))

        for t in range(best_ids.shape[0]):
            cur_id = int(best_ids[t].item())
            cur_prob = float(best_probs[t].item())

            if cur_id == prev_id:
                run_probs.append(cur_prob)
            else:
                flush()
                prev_id = cur_id
                run_probs = [cur_prob]

        flush()

        return Decoded(text="".join(chars), confidences=confidences)

    def decode_batch(self, log_probs: torch.Tensor) -> List[Decoded]:
        """Args: log_probs of shape (T, B, vocab_size)."""
        return [self.decode(log_probs[:, b, :]) for b in range(log_probs.shape[1])]
