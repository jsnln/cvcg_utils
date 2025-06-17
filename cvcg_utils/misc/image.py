import os
import numpy as np
import cv2

def read_rgb(fn):
    img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    assert len(img.shape) == 3
    assert img.shape[-1] == 3
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def read_rgba(fn):
    img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    assert len(img.shape) == 3
    assert img.shape[-1] == 4
    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

def read_grayscale(fn: str):
    grayscale = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    assert len(grayscale.shape) == 2
    return grayscale

def read_rgb_exr(fn):
    assert os.path.splitext(fn)[-1] in ['.exr']
    img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    assert len(img.shape) == 3
    assert img.shape[-1] == 3
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def write_rgb(fn: str, rgb: np.ndarray):
    assert rgb.dtype == np.uint8
    assert len(rgb.shape) == 3
    assert rgb.shape[2] == 3
    assert os.path.splitext(fn)[-1] in ['.png', '.jpg', '.jpeg']

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(fn, bgr)

def write_rgb_uint16(fn: str, rgb: np.ndarray):
    assert rgb.dtype == np.uint16
    assert len(rgb.shape) == 3
    assert rgb.shape[2] == 3
    assert os.path.splitext(fn)[-1] in ['.png', '.jpg', '.jpeg']

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(fn, bgr)


def write_grayscale_uint16(fn: str, grayscale: np.ndarray):
    assert grayscale.dtype == np.float16
    assert len(grayscale.shape) == 2
    assert os.path.splitext(fn)[-1] in ['.png']

    cv2.imwrite(fn, grayscale)


def write_rgba_uint16(fn: str, rgba: np.ndarray):
    assert rgba.dtype == np.uint16
    assert len(rgba.shape) == 3
    assert rgba.shape[2] == 4
    assert os.path.splitext(fn)[-1] in ['.png', '.jpg', '.jpeg']

    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(fn, bgra)

def write_rgb_exr(fn: str, rgb: np.ndarray):
    assert rgb.dtype == np.float32
    assert len(rgb.shape) == 3
    assert rgb.shape[2] == 3
    assert os.path.splitext(fn)[-1] in ['.exr']

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(fn, bgr)

def write_bgr(fn: str, bgr: np.ndarray):
    assert bgr.dtype == np.uint8
    assert len(bgr.shape) == 3
    assert bgr.shape[2] == 3
    assert os.path.splitext(fn)[-1] in ['.png', '.jpg', '.jpeg']

    cv2.imwrite(fn, bgr)

def write_grayscale_exr(fn: str, grayscale: np.ndarray):
    assert grayscale.dtype == np.float32
    assert len(grayscale.shape) == 2
    assert os.path.splitext(fn)[-1] in ['.exr']

    cv2.imwrite(fn, grayscale)

def write_grayscale(fn: str, grayscale: np.ndarray):
    assert grayscale.dtype == np.uint8
    assert len(grayscale.shape) == 2
    assert os.path.splitext(fn)[-1] in ['.png', '.jpg', '.jpeg']

    cv2.imwrite(fn, grayscale)

def to_u8_s255(src: np.ndarray):
    """
    change type to uint8 and scale by 255
    """
    return (src * 255).astype(np.uint8)
