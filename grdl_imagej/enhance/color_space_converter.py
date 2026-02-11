# -*- coding: utf-8 -*-
"""
Color Space Converter - RGB to/from HSB, Lab, YCbCr.

Converts between color spaces: RGB, HSB (HSV), CIE L*a*b*, and YCbCr.
Useful for color-based segmentation and analysis where perceptual (Lab)
or luminance-chrominance spaces are advantageous.

Attribution
-----------
ImageJ implementation: ``ij/process/ColorProcessor.java`` (``toHSB()``,
``toFloat()``), ``ij/plugin/ColorConverter.java`` in ImageJ 1.54j.
ImageJ 1.x source is in the public domain.
CIE (1976), Poynton "Digital Video and HDTV", Morgan Kaufmann, 2003.

Dependencies
------------
numpy

Author
------
Steven Siebert

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-11

Modified
--------
2026-02-11
"""

# Standard library
from typing import Annotated, Any

# Third-party
import numpy as np

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.params import Desc, Options
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC

# sRGB → XYZ (D65) matrix
_SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])

# XYZ → sRGB (D65) matrix
_XYZ_TO_SRGB = np.linalg.inv(_SRGB_TO_XYZ)

# D65 reference white point
_D65_WHITE = np.array([0.95047, 1.00000, 1.08883])
# D50 reference white point
_D50_WHITE = np.array([0.96422, 1.00000, 0.82521])


def _linearize_srgb(c: np.ndarray) -> np.ndarray:
    """Convert sRGB [0,1] to linear RGB."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _delinearize_srgb(c: np.ndarray) -> np.ndarray:
    """Convert linear RGB to sRGB [0,1]."""
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(np.maximum(c, 0), 1 / 2.4) - 0.055)


def _rgb_to_hsb(image: np.ndarray) -> np.ndarray:
    """Convert RGB [0,1] to HSB [0,1]."""
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # Hue
    h = np.zeros_like(r)
    mask_r = (delta > 0) & (cmax == r)
    mask_g = (delta > 0) & (cmax == g)
    mask_b = (delta > 0) & (cmax == b)
    h[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6.0
    h[mask_g] = (b[mask_g] - r[mask_g]) / delta[mask_g] + 2.0
    h[mask_b] = (r[mask_b] - g[mask_b]) / delta[mask_b] + 4.0
    h = h / 6.0

    # Saturation
    s = np.where(cmax > 0, delta / cmax, 0.0)

    return np.stack([h, s, cmax], axis=-1)


def _hsb_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert HSB [0,1] to RGB [0,1]."""
    h, s, v = image[..., 0], image[..., 1], image[..., 2]
    h6 = h * 6.0
    i = np.floor(h6).astype(int) % 6
    f = h6 - np.floor(h6)
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))

    result = np.zeros_like(image)
    for idx, (r, g, b) in enumerate([(v, t, p), (q, v, p), (p, v, t),
                                      (p, q, v), (t, p, v), (v, p, q)]):
        mask = i == idx
        result[mask, 0] = r[mask]
        result[mask, 1] = g[mask]
        result[mask, 2] = b[mask]

    return result


def _rgb_to_lab(image: np.ndarray, white: np.ndarray) -> np.ndarray:
    """Convert RGB [0,1] to CIE L*a*b*."""
    linear = _linearize_srgb(image)
    # RGB → XYZ
    xyz = np.einsum('ij,...j->...i', _SRGB_TO_XYZ, linear)
    # Normalize by white point
    xyz = xyz / white

    # f(t) function
    delta = 6.0 / 29.0
    delta_sq = delta ** 2
    f = np.where(xyz > delta_sq * delta,
                 np.cbrt(xyz),
                 xyz / (3.0 * delta_sq) + 4.0 / 29.0)

    l = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])

    return np.stack([l, a, b], axis=-1)


def _lab_to_rgb(image: np.ndarray, white: np.ndarray) -> np.ndarray:
    """Convert CIE L*a*b* to RGB [0,1]."""
    l, a, b = image[..., 0], image[..., 1], image[..., 2]

    fy = (l + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    delta = 6.0 / 29.0
    delta_sq = delta ** 2

    xyz = np.stack([fx, fy, fz], axis=-1)
    xyz = np.where(xyz > delta,
                   xyz ** 3,
                   3.0 * delta_sq * (xyz - 4.0 / 29.0))
    xyz = xyz * white

    # XYZ → linear RGB
    linear = np.einsum('ij,...j->...i', _XYZ_TO_SRGB, xyz)
    return np.clip(_delinearize_srgb(linear), 0.0, 1.0)


def _rgb_to_ycbcr(image: np.ndarray) -> np.ndarray:
    """Convert RGB [0,1] to YCbCr (BT.601)."""
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.5 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return np.stack([y, cb, cr], axis=-1)


def _ycbcr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert YCbCr (BT.601) to RGB [0,1]."""
    y, cb, cr = image[..., 0], image[..., 1], image[..., 2]
    r = y + 1.402 * (cr - 0.5)
    g = y - 0.344136 * (cb - 0.5) - 0.714136 * (cr - 0.5)
    b = y + 1.772 * (cb - 0.5)
    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0)


@processor_tags(modalities=[IM.EO, IM.MSI, IM.PAN],
                category=PC.ENHANCE)
@processor_version('1.54j')
class ColorSpaceConverter(ImageTransform):
    """Color space converter, ported from ImageJ 1.54j.

    Converts 3-channel images between RGB, HSB (HSV), CIE L*a*b*,
    and YCbCr color spaces.

    Parameters
    ----------
    source_space : str
        Input color space. Default is ``'rgb'``.
    target_space : str
        Output color space. Default is ``'lab'``.
    illuminant : str
        White point for Lab conversion. ``'D65'`` (default) or ``'D50'``.

    Notes
    -----
    Port of ``ij/process/ColorProcessor.java`` and
    ``ij/plugin/ColorConverter.java`` from ImageJ 1.54j (public domain).

    Input images should have values in [0, 1] for RGB channels.
    For 8-bit images, divide by 255 first.

    Examples
    --------
    >>> from grdl_imagej import ColorSpaceConverter
    >>> csc = ColorSpaceConverter(source_space='rgb', target_space='lab')
    >>> lab_image = csc.apply(rgb_normalized)
    """

    __imagej_source__ = 'ij/plugin/ColorConverter.java'
    __imagej_version__ = '1.54j'
    __gpu_compatible__ = True

    source_space: Annotated[str, Options('rgb', 'hsb', 'lab', 'ycbcr'),
                             Desc('Input color space')] = 'rgb'
    target_space: Annotated[str, Options('rgb', 'hsb', 'lab', 'ycbcr'),
                             Desc('Output color space')] = 'lab'
    illuminant: Annotated[str, Options('D50', 'D65'),
                           Desc('White point for Lab conversion')] = 'D65'

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Convert image between color spaces.

        Parameters
        ----------
        source : np.ndarray
            3-channel image array. Shape ``(rows, cols, 3)``.

        Returns
        -------
        np.ndarray
            Converted image (float64), same shape as input.

        Raises
        ------
        ValueError
            If source is not 3D with 3 channels.
        """
        if source.ndim != 3 or source.shape[2] != 3:
            raise ValueError(
                f"Expected 3-channel image (rows, cols, 3), got shape {source.shape}"
            )

        p = self._resolve_params(kwargs)
        src = p['source_space']
        tgt = p['target_space']
        white = _D65_WHITE if p['illuminant'] == 'D65' else _D50_WHITE

        image = source.astype(np.float64)

        if src == tgt:
            return image.copy()

        # Convert to RGB first (as intermediate)
        if src == 'rgb':
            rgb = image
        elif src == 'hsb':
            rgb = _hsb_to_rgb(image)
        elif src == 'lab':
            rgb = _lab_to_rgb(image, white)
        else:  # ycbcr
            rgb = _ycbcr_to_rgb(image)

        # Convert from RGB to target
        if tgt == 'rgb':
            return rgb
        elif tgt == 'hsb':
            return _rgb_to_hsb(rgb)
        elif tgt == 'lab':
            return _rgb_to_lab(rgb, white)
        else:  # ycbcr
            return _rgb_to_ycbcr(rgb)
