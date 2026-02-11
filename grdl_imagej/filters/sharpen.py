# -*- coding: utf-8 -*-
"""
Sharpen (Laplacian) - Fixed 3x3 Laplacian-based sharpening kernel.

ImageJ's "Sharpen" command (Process > Sharpen), distinct from the
configurable UnsharpMask. Uses a fixed 3x3 kernel with center weight 12,
edge weights -2, and corner weights -1, normalized to unit gain.

Attribution
-----------
ImageJ implementation: ``ij/process/ImageProcessor.java`` (``sharpen()``
method) in ImageJ 1.54j. ImageJ 1.x source is in the public domain.

Dependencies
------------
numpy
scipy.ndimage

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
from typing import Any

# Third-party
import numpy as np
from scipy.ndimage import convolve

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC

# ImageJ's sharpen kernel: center=12, all neighbors=-1, sum=4.
# Normalized by /4 for unit DC gain (flat regions are preserved).
_SHARPEN_KERNEL = np.array([
    [-1, -1, -1],
    [-1, 12, -1],
    [-1, -1, -1],
], dtype=np.float64) / 4.0


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI, IM.HSI, IM.SWIR, IM.MWIR, IM.LWIR],
                category=PC.FILTERS)
@processor_version('1.54j')
class Sharpen(ImageTransform):
    """Fixed 3x3 Laplacian sharpening filter, ported from ImageJ 1.54j.

    Applies a fixed sharpening kernel equivalent to adding a scaled
    Laplacian to the original image. No configurable parameters.

    Notes
    -----
    Port of ``ij/process/ImageProcessor.java`` ``sharpen()`` from
    ImageJ 1.54j (public domain). The kernel ``[[-1,-1,-1],[-1,12,-1],
    [-1,-1,-1]] / 4`` has unit DC gain (sum of weights = 1) so that
    flat regions are preserved. High-frequency detail is amplified.

    For configurable sharpening, use ``UnsharpMask``.

    Examples
    --------
    >>> from grdl_imagej import Sharpen
    >>> s = Sharpen()
    >>> sharpened = s.apply(blurry_image)
    """

    __imagej_source__ = 'ij/process/ImageProcessor.java'
    __imagej_version__ = '1.54j'
    __gpu_compatible__ = True

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply 3x3 Laplacian sharpening to a 2D image.

        Parameters
        ----------
        source : np.ndarray
            2D image array. Shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Sharpened image (float64), same shape as input.

        Raises
        ------
        ValueError
            If source is not 2D.
        """
        if source.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {source.shape}")

        image = source.astype(np.float64)
        return convolve(image, _SHARPEN_KERNEL, mode='nearest')
