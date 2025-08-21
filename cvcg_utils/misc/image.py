from warnings import warn

warn("[Deprecation Warning] Imports from `cvcg_utils.misc.image` is deprecated. Kindly replace them with cvcg_utils.image.image_io")

from cvcg_utils.image.image_io import read_rgb, read_rgb_exr, read_grayscale, read_rgba, write_rgb_exr, write_bgr, \
    write_grayscale, write_grayscale_exr, write_grayscale_uint16, write_rgb, write_rgb_uint16, write_rgba, write_rgba_uint16


