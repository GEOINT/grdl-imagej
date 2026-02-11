# -*- coding: utf-8 -*-
"""
Binary Outline - Reduces binary objects to 1-pixel-wide outlines.

Removes interior pixels that are fully surrounded by foreground, leaving
only boundary pixels. Equivalent to ``original AND NOT erode(original)``.

Attribution
-----------
ImageJ implementation: ``ij/process/BinaryProcessor.java`` (``outline()``
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
from typing import Annotated, Any

# Third-party
import numpy as np
from scipy.ndimage import binary_erosion

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.params import Desc, Options
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC

# Structuring elements for 4- and 8-connectivity
_STRUCT_4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
_STRUCT_8 = np.ones((3, 3), dtype=bool)


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI],
                category=PC.BINARY)
@processor_version('1.54j')
class BinaryOutline(ImageTransform):
    """Extracts 1-pixel-wide outlines from binary objects.

    Computes ``output = input AND NOT erode(input)`` to produce object
    boundaries.

    Parameters
    ----------
    connectivity : int
        Neighborhood connectivity for erosion. 4 uses a cross-shaped
        structuring element; 8 uses a full 3x3 square. Default is 4.

    Notes
    -----
    Port of ``ij/process/BinaryProcessor.java`` ``outline()`` from
    ImageJ 1.54j (public domain). Input is thresholded at 0.5 to
    ensure binary interpretation.

    Examples
    --------
    >>> from grdl_imagej import BinaryOutline
    >>> outline = BinaryOutline(connectivity=4)
    >>> edges = outline.apply(binary_mask)
    """

    __imagej_source__ = 'ij/process/BinaryProcessor.java'
    __imagej_version__ = '1.54j'
    __gpu_compatible__ = True

    connectivity: Annotated[int, Options(4, 8),
                             Desc('Neighborhood connectivity')] = 4

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Extract outlines from a binary 2D image.

        Parameters
        ----------
        source : np.ndarray
            2D binary image. Shape ``(rows, cols)``. Non-zero pixels
            are treated as foreground.

        Returns
        -------
        np.ndarray
            Binary outline image (float64, values 0.0 or 1.0).

        Raises
        ------
        ValueError
            If source is not 2D.
        """
        if source.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {source.shape}")

        p = self._resolve_params(kwargs)
        struct = _STRUCT_4 if p['connectivity'] == 4 else _STRUCT_8

        binary = source > 0.5
        eroded = binary_erosion(binary, structure=struct, border_value=0)
        outline = binary & ~eroded

        return outline.astype(np.float64)
