import numpy as np

from preprocessing.binarization import binarize


def test_binarize_separates_text_from_background():
    rng = np.random.default_rng(42)

    image = np.full((100, 100), 220, dtype=np.uint8)
    image[40:60, 20:80] = 30  # dark "text" stripe on a light background

    noise = rng.integers(-10, 10, size=image.shape)
    noisy = np.clip(image.astype(int) + noise, 0, 255).astype(np.uint8)

    binary = binarize(noisy)

    assert binary.shape == image.shape
    assert set(np.unique(binary)).issubset({0, 255})

    # the text stripe should be predominantly foreground (255)
    text_region = binary[40:60, 20:80]
    assert (text_region == 255).mean() > 0.9

    # a clean background patch away from the stripe should be mostly background (0)
    bg_region = binary[0:20, 0:20]
    assert (bg_region == 0).mean() > 0.9
