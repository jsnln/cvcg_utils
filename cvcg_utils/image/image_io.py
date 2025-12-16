from typing import Union, Tuple, IO
from pathlib import Path
import os
import io
import numpy as np
import cv2
import PIL.Image
import PIL.PngImagePlugin

def read_rgb(fn: str) -> np.ndarray:
    """
    Reads and returns a 3-channel RGB image (``cv2`` backend).

    Raises ``AssertionError`` if image read fails or if the image is not 3-channel.
    """
    img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    assert img is not None, f"image read failed for {fn}"
    assert len(img.shape) == 3, f"image {fn} must not be grayscale"
    assert img.shape[-1] == 3, f"image {fn} must have 3 channels"
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def read_rgba(fn: str) -> np.ndarray:
    """
    Reads and returns a 4-channel RGBA image (``cv2`` backend).
    
    Raises ``AssertionError`` if image read fails or if the image is not 4-channel.
    """
    img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    assert img is not None, f"image read failed for {fn}"
    assert len(img.shape) == 3, f"image {fn} must not be grayscale"
    assert img.shape[-1] == 4, f"image {fn} must have 4 channels"
    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

def read_grayscale(fn: str) -> np.ndarray:
    """
    Reads and returns a grayscale image (``cv2`` backend).
    
    Raises ``AssertionError`` if image read fails or if the image is not 1-channel.
    """
    grayscale = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    assert grayscale is not None, f"image read failed for {fn}"
    assert len(grayscale.shape) == 2, f"image {fn} must be grayscale"
    return grayscale

def read_rgb_exr(fn) -> np.ndarray:
    """
    Reads and returns an exr format RGB image (``cv2`` backend).
    
    Raises ``AssertionError`` if the image is not in exr format, if image read fails or if the image is not 3-channel.
    """
    assert os.path.splitext(fn)[-1] in ['.exr'], f"{fn} does not have .exr format"
    img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    assert img is not None, f"image read failed for {fn}"
    assert len(img.shape) == 3, f"image {fn} must not be grayscale"
    assert img.shape[-1] == 3, f"image {fn} must have 3 channels"
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def write_rgb(fn: str, rgb: np.ndarray):
    """
    Write ``np.uint8`` RGB image ``rgb`` to path ``fn`` (``cv2`` backend). Supports only ``.png``, ``.jpg`` and ``.jpeg`` formats.
    
    Raises ``AssertionError`` if the image is not ``np.uint8`` RGB, if ``fn`` has a wrong suffix, or if image write fails.
    """
    assert rgb.dtype == np.uint8, f"the image must have dtype np.uint8"
    assert len(rgb.shape) == 3, f"the image must be 3-channel RGB"
    assert rgb.shape[2] == 3, f"the image must be 3-channel RGB"
    assert os.path.splitext(fn)[-1].lower() in ['.png', '.jpg', '.jpeg'], f"the output format must be png, jpg or jpeg"

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    assert cv2.imwrite(fn, bgr), f"writing to {fn} failed"


def write_rgba(fn: str, rgba: np.ndarray):
    """
    Write ``np.uint8`` RGBA image ``rgb`` to path ``fn`` (``cv2`` backend). Supports only ``.png``, ``.jpg`` and ``.jpeg`` formats.
    
    Raises ``AssertionError`` if the image is not ``np.uint8`` RGBA, if ``fn`` has a wrong suffix, or if image write fails.
    """
    assert rgba.dtype == np.uint8, f"the image must have dtype np.uint8"
    assert len(rgba.shape) == 3, f"the image must be 4-channel RGBA"
    assert rgba.shape[2] == 4, f"the image must be 4-channel RGBA"
    assert os.path.splitext(fn)[-1].lower() in ['.png', '.jpg', '.jpeg'], f"the output format must be png, jpg or jpeg"

    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    assert cv2.imwrite(fn, bgra), f"writing to {fn} failed"

def write_rgb_uint16(fn: str, rgb: np.ndarray):
    """
    Write ``np.uint16`` RGB image ``rgb`` to path ``fn`` (``cv2`` backend). Supports only ``.png``, ``.jpg`` and ``.jpeg`` formats.
    
    Raises ``AssertionError`` if the image is not ``np.uint16`` RGB, if ``fn`` has a wrong suffix, or if image write fails.
    """
    assert rgb.dtype == np.uint16, f"the image must have dtype np.uint16"
    assert len(rgb.shape) == 3, f"the image must be 3-channel RGB"
    assert rgb.shape[2] == 3, f"the image must be 3-channel RGB"
    assert os.path.splitext(fn)[-1].lower() in ['.png', '.jpg', '.jpeg'], f"the output format must be png, jpg or jpeg"

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    assert cv2.imwrite(fn, bgr), f"writing to {fn} failed"

def write_grayscale_uint16(fn: str, grayscale: np.ndarray):
    """
    Write ``np.uint16`` grayscale image ``rgb`` to path ``fn`` (``cv2`` backend). Supports only ``.png`` format.
    
    Raises ``AssertionError`` if the image is not ``np.uint16`` grayscale, if ``fn`` has a wrong suffix, or if image write fails.
    """
    assert grayscale.dtype == np.uint16, f"the image must have dtype np.uint16"
    assert len(grayscale.shape) == 2, f"the image must be grayscale"
    assert os.path.splitext(fn)[-1].lower() in ['.png'], f"the output format must be png"

    assert cv2.imwrite(fn, grayscale), f"writing to {fn} failed"


def write_rgba_uint16(fn: str, rgba: np.ndarray):
    """
    Write ``np.uint16`` RGBA image ``rgba`` to path ``fn`` (``cv2`` backend). Supports only ``.png``, ``.jpg`` and ``.jpeg`` formats.
    
    Raises ``AssertionError`` if the image is not ``np.uint16`` RGBA, if ``fn`` has a wrong suffix, or if image write fails.
    """
    assert rgba.dtype == np.uint16, f"the image must have dtype np.uint16"
    assert len(rgba.shape) == 3, f"the image must be 4-channel RGBA"
    assert rgba.shape[2] == 4, f"the image must be 4-channel RGBA"
    assert os.path.splitext(fn)[-1].lower() in ['.png', '.jpg', '.jpeg'], f"the output format must be png, jpg or jpeg"

    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    assert cv2.imwrite(fn, bgra), f"writing to {fn} failed"

def write_bgr(fn: str, bgr: np.ndarray):
    """
    Write ``np.uint8`` BGR image ``bgr`` to path ``fn`` (``cv2`` backend). Supports only ``.png``, ``.jpg`` and ``.jpeg`` formats.
    
    Raises ``AssertionError`` if the image is not ``np.uint8`` RGB, if ``fn`` has a wrong suffix, or if image write fails.
    """
    assert bgr.dtype == np.uint8, f"the image must have dtype np.uint8"
    assert len(bgr.shape) == 3, f"the image must be 3-channel RGB"
    assert bgr.shape[2] == 3, f"the image must be 3-channel RGB"
    assert os.path.splitext(fn)[-1].lower() in ['.png', '.jpg', '.jpeg'], f"the output format must be png, jpg or jpeg"

    assert cv2.imwrite(fn, bgr), f"writing to {fn} failed"


def write_rgb_exr(fn: str, rgb: np.ndarray):
    """
    Write ``np.float32`` RGB image ``rgb`` to path ``fn`` (``cv2`` backend). Supports only ``.exr`` format.
    
    Raises ``AssertionError`` if the image is not ``np.float32`` RGB, if ``fn`` has a wrong suffix, or if image write fails.
    """
    assert rgb.dtype == np.float32, f"the image must have dtype np.float32"
    assert len(rgb.shape) == 3, f"the image must be 3-channel RGB"
    assert rgb.shape[2] == 3, f"the image must be 3-channel RGB"
    assert os.path.splitext(fn)[-1] in ['.exr'], f"the output format must be exr"

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    assert cv2.imwrite(fn, bgr), f"writing to {fn} failed"


def write_grayscale_exr(fn: str, grayscale: np.ndarray):
    """
    Write ``np.float32`` grayscale image ``grayscale`` to path ``fn`` (``cv2`` backend). Supports only ``.exr`` format.
    
    Raises ``AssertionError`` if the image is not ``np.float32`` grayscale, if ``fn`` has a wrong suffix, or if image write fails.
    """
    assert grayscale.dtype == np.float32, f"the image must have dtype np.float32"
    assert len(grayscale.shape) == 2, f"the image must be 2-channel grayscale"
    assert os.path.splitext(fn)[-1].lower() in ['.exr'], f"the output format must be exr"

    assert cv2.imwrite(fn, grayscale), f"writing to {fn} failed"

def write_grayscale(fn: str, grayscale: np.ndarray):
    """
    Write ``np.uint8`` grayscale image ``grayscale`` to path ``fn`` (``cv2`` backend). Supports only ``.png``, ``.jpg`` and ``.jpeg`` format.
    
    Raises ``AssertionError`` if the image is not ``np.float32`` grayscale, if ``fn`` has a wrong suffix, or if image write fails.
    """
    assert grayscale.dtype == np.uint8, f"the image must have dtype np.uint8"
    assert len(grayscale.shape) == 2, f"the image must be 2-channel grayscale"
    assert os.path.splitext(fn)[-1].lower() in ['.png', '.jpg', '.jpeg'], f"the output format must be png, jpg or jpeg"

    assert cv2.imwrite(fn, grayscale), f"writing to {fn} failed"

def to_u8_s255(src: np.ndarray):
    """
    change type to uint8 and scale by 255
    """
    return (src * 255).astype(np.uint8)



def read_depth_compressed_dr_png(path: Union[str, os.PathLike, IO], read_nan_inf_as_zero: bool=False) -> Tuple[np.ndarray, float]:
    """
    Take from https://github.com/microsoft/MoGe/blob/main/moge/utils/io.py#L89

    Read a depth image, return float32 depth array of shape (H, W).

    This depth should be in uint16 png format, values are dynamically log-scaled to 1 ~ 65534
    """
    if isinstance(path, (str, os.PathLike)):
        data = Path(path).read_bytes()
    else:
        data = path.read()
    pil_image = PIL.Image.open(io.BytesIO(data))
    near = float(pil_image.info.get('near'))
    far = float(pil_image.info.get('far'))
    unit = float(pil_image.info.get('unit')) if 'unit' in pil_image.info else None
    depth = np.array(pil_image)
    mask_nan, mask_inf = depth == 0, depth == 65535
    depth = (depth.astype(np.float32) - 1) / 65533
    depth = near ** (1 - depth) * far ** depth
    if read_nan_inf_as_zero:
        depth[mask_nan] = 0
        depth[mask_inf] = 0
    else:
        depth[mask_nan] = np.nan
        depth[mask_inf] = np.inf
    return depth, unit


def write_depth_compressed_dr_png(
    path: Union[str, os.PathLike, IO], 
    depth: np.ndarray, 
    unit: float = None,
    max_range: float = 1e5,
    compression_level: int = 7,
):
    """
    This is taken from https://github.com/microsoft/MoGe/blob/main/moge/utils/io.py#L110

    This depth will be dynamically log-scaled to 1 ~ 65534 and converted to uint16 png format.

    Encode and write a depth image as 16-bit PNG format.
    ### Parameters:
    - `path: Union[str, os.PathLike, IO]`
        The file path or file object to write to.
    - `depth: np.ndarray`
        The depth array, float32 array of shape (H, W). 
        May contain `NaN` for invalid values and `Inf` for infinite values.
    - `unit: float = None`
        The unit of the depth values.
    
    Depth values are encoded as follows:
    - 0: unknown
    - 1 ~ 65534: depth values in logarithmic
    - 65535: infinity
    
    metadata is stored in the PNG file as text fields:
    - `near`: the minimum depth value
    - `far`: the maximum depth value
    - `unit`: the unit of the depth values (optional)
    """
    mask_values, mask_nan, mask_inf = np.isfinite(depth) & (depth > 0), np.isnan(depth) | (depth <= 0), np.isinf(depth)

    depth = depth.astype(np.float32)
    mask_finite = depth
    near = max(depth[mask_values].min(), 1e-5)
    far = max(near * 1.1, min(depth[mask_values].max(), near * max_range))
    depth = 1 + np.round((np.log(np.nan_to_num(depth, nan=0).clip(near, far) / near) / np.log(far / near)).clip(0, 1) * 65533).astype(np.uint16) # 1~65534
    depth[mask_nan] = 0
    depth[mask_inf] = 65535

    pil_image = PIL.Image.fromarray(depth)
    pnginfo = PIL.PngImagePlugin.PngInfo()
    pnginfo.add_text('near', str(near))
    pnginfo.add_text('far', str(far))
    if unit is not None:
        pnginfo.add_text('unit', str(unit))
    pil_image.save(path, pnginfo=pnginfo, compress_level=compression_level)

def read_rgb_compressed_flat_dr_png(path: Union[str, os.PathLike, IO], read_nan_as: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    This is used to read images written with write_rgb_compressed_dr_png
    """
    if isinstance(path, (str, os.PathLike)):
        data = Path(path).read_bytes()
    else:
        data = path.read()
    pil_image = PIL.Image.open(io.BytesIO(data))
    R_low = float(pil_image.info.get('R_low'))
    G_low = float(pil_image.info.get('G_low'))
    B_low = float(pil_image.info.get('B_low'))
    R_high = float(pil_image.info.get('R_high'))
    G_high = float(pil_image.info.get('G_high'))
    B_high = float(pil_image.info.get('B_high'))
    
    data = np.array(pil_image)
    
    H, W_flat = data.shape
    assert W_flat % 3 == 0
    W = W_flat // 3
    data = data.reshape(H, 3, W)
    mask = (data[:, 0] > 0) & (data[:, 0] < 65535)

    data = (data.astype(np.float32) - 1) / 65533
    R = data[:, 0] * (R_high - R_low) + R_low
    G = data[:, 1] * (G_high - G_low) + G_low
    B = data[:, 2] * (B_high - B_low) + B_low

    data_metric = np.stack([R, G, B], axis=-1)
    data_metric[~mask] = read_nan_as
    return data_metric, mask

def write_rgb_compressed_flat_dr_png(
    path: Union[str, os.PathLike, IO], 
    data: np.ndarray,
    mask: np.ndarray = None,
    compression_level: int = 7,
):
    """
    This is taken from https://github.com/microsoft/MoGe/blob/main/moge/utils/io.py#L110

    This depth will be dynamically scaled to 1 ~ 65534 and converted to uint16 png format.

    Encode and write a float image as 16-bit PNG format

    Range is computed channel-wise.

    BG pixels will be converted to nan for compression

    ### Parameters:
    - `path: Union[str, os.PathLike, IO]`
        The file path or file object to write to.
    - `depth: np.ndarray`
        The depth array, float32 array of shape (H, W). 
        May contain `NaN` for invalid values and `Inf` for infinite values.
    - `unit: float = None`
        The unit of the depth values.
    
    Depth values are encoded as follows:
    - 0: unknown
    - 1 ~ 65534: depth values in logarithmic
    - 65535: infinity
    
    metadata is stored in the PNG file as text fields:
    - `near`: the minimum depth value
    - `far`: the maximum depth value
    - `unit`: the unit of the depth values (optional)
    """
    assert len(data.shape) == 3 # must be [H, W, 3]
    assert data.shape[2] == 3   # must be RGB

    H, W, _ = data.shape
    data = data.astype(np.float32)

    if mask is None:
        mask = np.ones((H, W), dtype=bool)

    R_low, R_high = data[mask, 0].min().item(), data[mask, 0].max().item()
    G_low, G_high = data[mask, 1].min().item(), data[mask, 1].max().item()
    B_low, B_high = data[mask, 2].min().item(), data[mask, 2].max().item()
    
    R_scaled = 1 + np.round(((data[..., 0] - R_low) / (R_high - R_low)).clip(0, 1) * 65533).astype(np.uint16) # 1~65534
    G_scaled = 1 + np.round(((data[..., 1] - G_low) / (G_high - G_low)).clip(0, 1) * 65533).astype(np.uint16) # 1~65534
    B_scaled = 1 + np.round(((data[..., 2] - B_low) / (B_high - B_low)).clip(0, 1) * 65533).astype(np.uint16) # 1~65534
    data_scaled = np.stack([R_scaled, G_scaled, B_scaled], axis=-1)     # [H, W, 3]

    data_scaled[~mask] = 0
    # reserving 65535 for something else?

    data_scaled_flat = data_scaled.swapaxes(1, 2).reshape(H, 3*W)

    pil_image = PIL.Image.fromarray(data_scaled_flat)
    pnginfo = PIL.PngImagePlugin.PngInfo()
    pnginfo.add_text('R_low', str(R_low))
    pnginfo.add_text('G_low', str(G_low))
    pnginfo.add_text('B_low', str(B_low))
    pnginfo.add_text('R_high', str(R_high))
    pnginfo.add_text('G_high', str(G_high))
    pnginfo.add_text('B_high', str(B_high))
    pil_image.save(path, pnginfo=pnginfo, compress_level=compression_level)

