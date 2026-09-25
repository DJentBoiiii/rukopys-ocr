"""Deskewing based on Hough line detection."""
from __future__ import annotations

import cv2
import numpy as np

from preprocessing.binarization import binarize


def _estimate_skew_angle(binary_image: np.ndarray) -> float:
    """Estimate the dominant skew angle (in degrees) of text lines.

    Uses probabilistic Hough transform on a binary image where the
    foreground (text) is 255. Returns 0.0 if no lines are detected.
    """
    lines = cv2.HoughLinesP(
        binary_image,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=binary_image.shape[1] // 4,
        maxLineGap=10,
    )
    if lines is None:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = np.asarray(line).reshape(-1)[:4]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        # keep only near-horizontal candidates (text baselines)
        if -45 < angle < 45:
            angles.append(angle)

    if not angles:
        return 0.0

    return float(np.median(angles))


def deskew(image: np.ndarray) -> np.ndarray:
    """Correct the skew (rotation) of a grayscale line/document image.

    Estimates the dominant line angle via a Hough transform on a
    Sauvola-binarized version of the image, then rotates the original
    image to correct it.

    Args:
        image: 2D grayscale image array, shape (H, W).

    Returns:
        The rotation-corrected grayscale image, same shape as input.
    """
    if image.ndim != 2:
        raise ValueError("deskew expects a 2D grayscale image")

    binary = binarize(image)
    angle = _estimate_skew_angle(binary)

    if abs(angle) < 0.1:
        return image.copy()

    h, w = image.shape
    center = (w / 2, h / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    border_value = int(np.median(image))
    corrected = cv2.warpAffine(
        image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    return corrected
