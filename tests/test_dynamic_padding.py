import numpy as np
import torch

from model.crnn import CRNNModel, compute_output_seq_length
from model.tokenizer import CharTokenizer
from training.datasets_common import (
    HARD_CAP,
    IMG_HEIGHT,
    LineImageDataset,
    collate_batch,
    resize_to_fixed_height,
)


def _line_image(width: int, height: int = 100) -> np.ndarray:
    return np.full((height, width), 255, dtype=np.uint8)


def test_resize_to_fixed_height_does_not_crop_widths_up_to_hard_cap():
    # a line whose resized width lands well under HARD_CAP must survive intact
    image = _line_image(width=2000, height=1000)  # resized width = 2000 * 32 / 1000 = 64
    resized = resize_to_fixed_height(image)
    assert resized.shape == (IMG_HEIGHT, 64)

    # a line whose resized width is exactly at HARD_CAP must not be truncated
    image_at_cap = _line_image(width=HARD_CAP * 10, height=320)  # resized width = HARD_CAP
    resized_at_cap = resize_to_fixed_height(image_at_cap)
    assert resized_at_cap.shape[1] == HARD_CAP

    # only a pathological outlier past HARD_CAP should be capped
    image_over_cap = _line_image(width=(HARD_CAP + 500) * 10, height=320)
    resized_over_cap = resize_to_fixed_height(image_over_cap)
    assert resized_over_cap.shape[1] == HARD_CAP


def test_collate_batch_uses_per_batch_max_width_not_a_global_constant():
    tokenizer = CharTokenizer.from_texts(["ab"])

    small_samples = [(_line_image(width=200, height=100), "ab") for _ in range(4)]
    large_samples = [(_line_image(width=2000, height=100), "ab") for _ in range(4)]

    small_dataset = LineImageDataset(small_samples, tokenizer)
    large_dataset = LineImageDataset(large_samples, tokenizer)

    small_batch = [small_dataset[i] for i in range(len(small_dataset))]
    large_batch = [large_dataset[i] for i in range(len(large_dataset))]

    small_images, _, _, _ = collate_batch(small_batch)
    large_images, _, _, _ = collate_batch(large_batch)

    # the two batches must be padded to different widths, driven by their
    # own actual content, not a single fixed constant shared across batches
    assert small_images.shape[-1] != large_images.shape[-1]
    assert small_images.shape[-1] == max(b[2] for b in small_batch)
    assert large_images.shape[-1] == max(b[2] for b in large_batch)


def test_collate_batch_pads_to_batch_max_width_without_cropping_shorter_lines():
    tokenizer = CharTokenizer.from_texts(["ab"])
    # mixed widths within a single batch
    widths = [64, 128, 300]
    samples = [(_line_image(width=w * 10, height=320), "ab") for w in widths]  # resized widths == `widths`
    dataset = LineImageDataset(samples, tokenizer)
    batch = [dataset[i] for i in range(len(dataset))]

    images, _, input_lengths, _ = collate_batch(batch)

    assert images.shape[-1] == max(widths)
    # every real line width (<=800) must be fully preserved (no cropping),
    # i.e. its computed input_length must match its own true width, not the
    # padded batch width
    max_t = compute_output_seq_length(max(widths))
    for i, w in enumerate(widths):
        expected = max(1, min(compute_output_seq_length(w), max_t))
        assert input_lengths[i].item() == expected


def test_model_forward_accepts_variable_batch_widths():
    """`compute_output_seq_length` / the CNN must work correctly for whatever
    width a given batch happens to use, not just a hardcoded constant.
    """
    tokenizer = CharTokenizer.from_texts(["ab"])
    model = CRNNModel(vocab_size=tokenizer.vocab_size, img_height=IMG_HEIGHT)
    model.eval()

    for batch_width in (64, 384, 800):
        images = torch.zeros(2, 1, IMG_HEIGHT, batch_width)
        with torch.no_grad():
            log_probs = model(images)  # (T, B, V)
        expected_t = compute_output_seq_length(batch_width)
        assert log_probs.shape[0] == expected_t
        assert log_probs.shape[1] == 2
        assert log_probs.shape[2] == tokenizer.vocab_size
