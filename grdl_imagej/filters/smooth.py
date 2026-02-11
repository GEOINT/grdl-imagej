# -*- coding: utf-8 -*-
"""
Smooth (Mean Filter) - Fixed 3x3 mean smoothing filter.

Applies a fixed 3x3 averaging kernel for simple noise reduction. This is
ImageJ's "Smooth" command (Process > Smooth), distinct from the configurable-
radius mean filter in RankFilters.

Attribution
-----------
ImageJ implementation: ``ij/process/ImageProcessor.java`` (``smooth()``
method) in ImageJ 1.54j. ImageJ 1.x source is in the public domain.

Dependencies
------------
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
from scipy.ndimage import uniform_filter

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI, IM.HSI, IM.SWIR, IM.MWIR, IM.LWIR],
                category=PC.FILTERS)
@processor_version('1.54j')
class Smooth(ImageTransform):
    """Fixed 3x3 mean smoothing filter, ported from ImageJ 1.54j.

    Replaces each pixel with the average of its 3x3 neighborhood.
    No configurable parameters — this is the canonical "Smooth" command.

    Notes
    -----
    Port of ``ij/process/ImageProcessor.java`` ``smooth()`` from
    ImageJ 1.54j (public domain). Uses nearest-neighbor boundary padding
    to match ImageJ's boundary handling.

    For configurable-radius mean filtering, use ``RankFilters`` with
    ``filter_type='mean'``.

    Examples
    --------
    >>> from grdl_imagej import Smooth
    >>> s = Smooth()
    >>> smoothed = s.apply(noisy_image)
    """

    __imagej_source__ = 'ij/process/ImageProcessor.java'
    __imagej_version__ = '1.54j'
    __gpu_compatible__ = True

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply 3x3 mean smoothing to a 2D image.

        Parameters
        ----------
        source : np.ndarray
            2D image array. Shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Smoothed image (float64), same shape as input.

        Raises
        ------
        ValueError
            If source is not 2D.
        """
        if source.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {source.shape}")

        image = source.astype(np.float64)
        return uniform_filter(image, size=3, mode='nearest')
