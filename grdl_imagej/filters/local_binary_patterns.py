# -*- coding: utf-8 -*-
"""
Local Binary Patterns (LBP) - Texture micro-pattern encoding.

Encodes local texture by comparing each pixel to its circular neighborhood,
producing binary codes. Supports default, uniform, and rotation-invariant
variants. Powerful texture descriptor for terrain classification.

Attribution
-----------
Algorithm: Ojala, Pietikainen & Maenpaa, "Multiresolution Gray-Scale and
Rotation Invariant Texture Classification with Local Binary Patterns",
IEEE PAMI, 24(7), 2002.
Related: ``imagej-ops`` — ``LBP2D`` feature ops (BSD-2).

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
from grdl.image_processing.params import Desc, Options, Range
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC


def _bilinear_sample(image: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Bilinear interpolation for sub-pixel sampling."""
    rows, cols = image.shape
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    x0 = np.clip(x0, 0, cols - 1)
    x1 = np.clip(x1, 0, cols - 1)
    y0 = np.clip(y0, 0, rows - 1)
    y1 = np.clip(y1, 0, rows - 1)

    dx = x - np.floor(x)
    dy = y - np.floor(y)

    return (image[y0, x0] * (1 - dx) * (1 - dy) +
            image[y0, x1] * dx * (1 - dy) +
            image[y1, x0] * (1 - dx) * dy +
            image[y1, x1] * dx * dy)


def _count_transitions(code: int, n: int) -> int:
    """Count 0→1 and 1→0 transitions in a circular binary pattern."""
    transitions = 0
    for i in range(n):
        bit_curr = (code >> i) & 1
        bit_next = (code >> ((i + 1) % n)) & 1
        if bit_curr != bit_next:
            transitions += 1
    return transitions


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI, IM.HSI, IM.SWIR, IM.MWIR, IM.LWIR],
                category=PC.FILTERS)
@processor_version('1.0.0')
class LocalBinaryPatterns(ImageTransform):
    """Local Binary Patterns (LBP) texture descriptor.

    For each pixel, samples N points on a circle of radius R using
    bilinear interpolation, thresholds against the center value, and
    encodes the result as a binary number.

    Parameters
    ----------
    radius : int
        Neighborhood radius. Default is 1.
    n_neighbors : int
        Number of sampling points on the circle. Default is 8.
    method : str
        LBP variant:

        - ``'default'``: Standard LBP code (0 to 2^N - 1).
        - ``'uniform'``: Uniform patterns (at most 2 bit transitions)
          mapped to compact codes; non-uniform → single bin.
        - ``'rotation_invariant'``: Minimum rotation of binary code.

    Notes
    -----
    Reference: Ojala, Pietikainen & Maenpaa, IEEE PAMI 24(7), 2002.
    Based on ``imagej-ops`` LBP2D (BSD-2).

    The output image contains integer codes. For ``'default'`` mode,
    codes range from 0 to ``2^n_neighbors - 1``. For ``'uniform'``
    mode, there are ``n_neighbors + 2`` distinct values.

    Examples
    --------
    >>> from grdl_imagej import LocalBinaryPatterns
    >>> lbp = LocalBinaryPatterns(radius=1, n_neighbors=8, method='uniform')
    >>> texture_codes = lbp.apply(pan_image)
    """

    __imagej_source__ = 'imagej-ops/features/lbp2d/LBP2D.java'
    __imagej_version__ = '1.0.0'
    __gpu_compatible__ = False

    radius: Annotated[int, Range(min=1, max=5),
                       Desc('Neighborhood radius')] = 1
    n_neighbors: Annotated[int, Options(8, 16, 24),
                            Desc('Sampling points on circle')] = 8
    method: Annotated[str, Options('default', 'uniform', 'rotation_invariant'),
                       Desc('LBP variant')] = 'default'

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute LBP codes for a 2D image.

        Parameters
        ----------
        source : np.ndarray
            2D image array. Shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            LBP code image (float64), same shape as input.

        Raises
        ------
        ValueError
            If source is not 2D.
        """
        if source.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {source.shape}")

        p = self._resolve_params(kwargs)
        r = p['radius']
        n = p['n_neighbors']
        method = p['method']

        image = source.astype(np.float64)
        rows, cols = image.shape

        # Compute sampling coordinates (vectorized over all pixels)
        angles = 2.0 * np.pi * np.arange(n) / n

        # Create coordinate grids for all pixels
        yy, xx = np.mgrid[0:rows, 0:cols]

        # Compute LBP codes
        codes = np.zeros((rows, cols), dtype=np.int64)
        for k in range(n):
            # Sample point position
            sy = yy + r * np.sin(angles[k])
            sx = xx + r * np.cos(angles[k])

            # Bilinear interpolation
            neighbor_val = _bilinear_sample(image, sy.ravel(), sx.ravel())
            neighbor_val = neighbor_val.reshape(rows, cols)

            # Threshold: 1 if neighbor >= center
            bit = (neighbor_val >= image).astype(np.int64)
            codes += bit << k

        # Apply variant mapping
        if method == 'uniform':
            result = self._to_uniform(codes, n)
        elif method == 'rotation_invariant':
            result = self._to_rotation_invariant(codes, n)
        else:
            result = codes

        return result.astype(np.float64)

    @staticmethod
    def _to_uniform(codes: np.ndarray, n: int) -> np.ndarray:
        """Map LBP codes to uniform pattern indices."""
        # Build lookup table
        max_code = 1 << n
        lut = np.zeros(max_code, dtype=np.int64)
        uniform_idx = 0
        for code in range(max_code):
            if _count_transitions(code, n) <= 2:
                lut[code] = uniform_idx
                uniform_idx += 1
            else:
                lut[code] = uniform_idx  # all non-uniform → last bin

        return lut[codes]

    @staticmethod
    def _to_rotation_invariant(codes: np.ndarray, n: int) -> np.ndarray:
        """Map LBP codes to rotation-invariant minimum."""
        max_code = 1 << n
        lut = np.zeros(max_code, dtype=np.int64)
        mask = max_code - 1
        for code in range(max_code):
            min_code = code
            rotated = code
            for _ in range(1, n):
                rotated = ((rotated >> 1) | ((rotated & 1) << (n - 1))) & mask
                if rotated < min_code:
                    min_code = rotated
            lut[code] = min_code

        return lut[codes]
