"""CRNN model: CNN feature extractor + BiLSTM + linear layer -> CTC logits."""
from __future__ import annotations

import torch
import torch.nn as nn


class CRNNModel(nn.Module):
    """CNN + BiLSTM CTC model for handwritten line recognition.

    Expects grayscale line images of shape (B, 1, H, W) with a fixed
    height (e.g. 32). The CNN reduces the height to 1 and downsamples
    the width, producing a sequence of feature vectors over width that
    are fed into a bidirectional LSTM and a final linear classifier
    over the character vocabulary (including the CTC blank at index 0).
    """

    def __init__(
        self,
        vocab_size: int,
        img_height: int = 32,
        cnn_channels: int = 64,
        rnn_hidden: int = 256,
        rnn_layers: int = 2,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # H/2, W/2

            nn.Conv2d(cnn_channels, cnn_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # H/4, W/4

            nn.Conv2d(cnn_channels * 2, cnn_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # H/8, W/4

            nn.Conv2d(cnn_channels * 4, cnn_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # H/16, W/4
            nn.Conv2d(cnn_channels * 4, cnn_channels * 4, kernel_size=2, padding=0),
            nn.ReLU(inplace=True),
        )

        cnn_out_height = img_height // 16 - 1
        if cnn_out_height < 1:
            raise ValueError("img_height too small for this CNN architecture")
        feature_size = cnn_channels * 4 * cnn_out_height

        self.rnn = nn.LSTM(
            input_size=feature_size,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.classifier = nn.Linear(rnn_hidden * 2, vocab_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Args: images (B, 1, H, W). Returns log-softmax logits (T, B, vocab_size)."""
        features = self.cnn(images)  # (B, C, H', W')
        b, c, h, w = features.shape
        features = features.permute(0, 3, 1, 2).reshape(b, w, c * h)  # (B, W, C*H)

        rnn_out, _ = self.rnn(features)  # (B, W, 2*hidden)
        logits = self.classifier(rnn_out)  # (B, W, vocab_size)

        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs.permute(1, 0, 2)  # (T=W, B, vocab_size), required by CTCLoss
