# -*- coding: utf-8 -*-
"""
Extended Min/Max and H-Minima/H-Maxima - Morphological extrema detection.

Finds extended (regional) minima/maxima suppressed to dynamic height h.
H-minima suppresses shallow minima (preventing over-segmentation).

Attribution
-----------
Algorithm: Soille, "Morphological Image Analysis", Springer, 2nd ed., 2003.

MorphoLibJ implementation:
``src/main/java/inra/ijpb/morphology/MinimaAndMaxima.java``
Source: https://github.com/ijpb/MorphoLibJ (LGPL-3).
This is an independent NumPy reimplementation.

Dependencies
------------
numpy
Depends on MorphologicalReconstruction (T2-06).

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
from scipy.ndimage import label as scipy_label, generate_binary_structure

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.params import Desc, Options, Range
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC

from grdl_imagej.binary.morphological_reconstruction import (
    reconstruct_by_dilation, reconstruct_by_erosion,
)

EXTREMA_TYPES = (
    'h_minima', 'h_maxima',
    'extended_minima', 'extended_maxima',
    'regional_minima', 'regional_maxima',
)


def _regional_minima(image: np.ndarray, connectivity: int) -> np.ndarray:
    """Find regional minima (flat zones lower than all neighbors)."""
    marker = image + 1
    reconstructed = reconstruct_by_erosion(marker, image, connectivity)
    return (reconstructed > image).astype(np.float64)


def _regional_maxima(image: np.ndarray, connectivity: int) -> np.ndarray:
    """Find regional maxima (flat zones higher than all neighbors)."""
    marker = image - 1
    reconstructed = reconstruct_by_dilation(marker, image, connectivity)
    return (reconstructed < image).astype(np.float64)


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI],
                category=PC.SEGMENTATION)
@processor_version('1.6.0')
class ExtendedMinMax(ImageTransform):
    """Extended minima/maxima and H-minima/maxima, ported from MorphoLibJ.

    Parameters
    ----------
    h : float
        Dynamic height threshold for h-minima/maxima. Default 10.0.
    connectivity : int
        4 or 8 connectivity. Default 4.
    type : str
        Operation type: ``'h_minima'``, ``'h_maxima'``,
        ``'extended_minima'``, ``'extended_maxima'``,
        ``'regional_minima'``, ``'regional_maxima'``.
        Default ``'h_minima'``.

    Notes
    -----
    Independent reimplementation of MorphoLibJ ``MinimaAndMaxima.java``
    (LGPL-3). Follows Soille (Springer, 2003).

    - H-minima: ``reconstruct_by_erosion(image + h, image)``
    - H-maxima: ``reconstruct_by_dilation(image - h, image)``
    - Extended minima: regional minima of h-minima result
    - Extended maxima: regional maxima of h-maxima result
    - Regional minima/maxima: flat zones lower/higher than all neighbors

    Examples
    --------
    >>> from grdl_imagej import ExtendedMinMax
    >>> emm = ExtendedMinMax(type='h_minima', h=5.0)
    >>> result = emm.apply(image)
    """

    __imagej_source__ = 'MorphoLibJ/morphology/MinimaAndMaxima.java'
    __imagej_version__ = '1.6.0'
    __gpu_compatible__ = False

    h: Annotated[float, Range(min=0.1, max=255.0),
                 Desc('Dynamic height threshold')] = 10.0
    connectivity: Annotated[int, Options(4, 8),
                            Desc('Pixel connectivity')] = 4
    type: Annotated[str, Options(*EXTREMA_TYPES),
                    Desc('Operation type')] = 'h_minima'

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply extended minima/maxima detection.

        Parameters
        ----------
        source : np.ndarray
            2D grayscale image. Shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            For h_minima/h_maxima: filtered grayscale image.
            For extended/regional: binary mask (0.0 or 1.0).
        """
        if source.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {source.shape}")

        p = self._resolve_params(kwargs)
        image = source.astype(np.float64)
        h = p['h']
        conn = p['connectivity']
        op_type = p['type']

        if op_type == 'h_minima':
            marker = image + h
            return reconstruct_by_erosion(marker, image, conn)

        elif op_type == 'h_maxima':
            marker = image - h
            return reconstruct_by_dilation(marker, image, conn)

        elif op_type == 'extended_minima':
            h_min = reconstruct_by_erosion(image + h, image, conn)
            return _regional_minima(h_min, conn)

        elif op_type == 'extended_maxima':
            h_max = reconstruct_by_dilation(image - h, image, conn)
            return _regional_maxima(h_max, conn)

        elif op_type == 'regional_minima':
            return _regional_minima(image, conn)

        else:  # regional_maxima
            return _regional_maxima(image, conn)
