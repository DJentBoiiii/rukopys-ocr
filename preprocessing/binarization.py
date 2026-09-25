"""Image preprocessing utilities: binarization and deskewing for line images."""
from __future__ import annotations

import numpy as np
from skimage.filters import threshold_sauvola


def binarize(image: np.ndarray, window_size: int = 25, k: float = 0.2) -> np.ndarray:
    """Binarize a grayscale image using Sauvola local thresholding.

    Args:
        image: 2D grayscale image array (uint8 or float), shape (H, W).
        window_size: odd size of the local neighbourhood used for the
            threshold estimation.
        k: Sauvola sensitivity parameter.

    Returns:
        A binary image of the same shape as `image`, dtype uint8, with
        values 0 (background) and 255 (foreground/text), using the
        convention that text is darker than the background.
    """
    if image.ndim != 2:
        raise ValueError("binarize expects a 2D grayscale image")
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")

    # keep the original dtype for threshold_sauvola: it infers the dynamic
    # range from the dtype, so casting to float64 beforehand would break
    # the default range estimation and produce a wrong threshold.
    thresh = threshold_sauvola(image, window_size=window_size, k=k)
    binary = np.where(image.astype(np.float64) < thresh, 255, 0).astype(np.uint8)
    return binary
