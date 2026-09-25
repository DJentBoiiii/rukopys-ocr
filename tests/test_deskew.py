import cv2
import numpy as np

from preprocessing.deskew import deskew, _estimate_skew_angle
from preprocessing.binarization import binarize


def _make_lined_image(angle_deg: float, size: int = 200) -> np.ndarray:
    image = np.full((size, size), 255, dtype=np.uint8)
    for y in range(20, size - 20, 20):
        cv2.line(image, (10, y), (size - 10, y), color=0, thickness=3)

    center = (size / 2, size / 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        image, rot_matrix, (size, size),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )
    return rotated


def test_deskew_corrects_known_rotation_angle():
    known_angle = 8.0
    skewed = _make_lined_image(known_angle)

    corrected = deskew(skewed)

    remaining_angle = _estimate_skew_angle(binarize(corrected))

    assert abs(remaining_angle) < 2.0


def test_deskew_returns_same_shape():
    skewed = _make_lined_image(5.0)
    corrected = deskew(skewed)
    assert corrected.shape == skewed.shape
