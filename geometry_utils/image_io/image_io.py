import os
import numpy as np
import cv2


def write_rgb(fn: str, rgb: np.ndarray):
    assert rgb.dtype == np.uint8
    assert len(rgb.shape) == 3
    assert rgb.shape[2] == 3
    assert os.path.splitext(fn)[-1] in ['.png', '.jpg', '.jpeg']

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(fn, bgr)


def write_grayscale(fn: str, grayscale: np.ndarray):
    assert grayscale.dtype == np.uint8
    assert len(grayscale.shape) == 2
    assert os.path.splitext(fn)[-1] in ['.png', '.jpg', '.jpeg']

    cv2.imwrite(fn, grayscale)
