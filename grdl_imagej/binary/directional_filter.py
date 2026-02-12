# -*- coding: utf-8 -*-
"""
Directional Filtering - Oriented line structuring element morphology.

Applies morphological operations using oriented line structuring elements
at multiple angles, then combines results. Detects and enhances linear
structures at specific orientations.

Attribution
-----------
Algorithm: Soille, Breen & Jones, "Recursive Implementation of Erosions
and Dilations Along Discrete Lines at Arbitrary Angles", IEEE PAMI, 18(5), 1996.

MorphoLibJ implementation:
``src/main/java/inra/ijpb/morphology/directional/DirectionalFilter.java``
Source: https://github.com/ijpb/MorphoLibJ (LGPL-3).
This is an independent NumPy reimplementation.

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
from scipy.ndimage import (
    minimum_filter, maximum_filter,
    grey_opening, grey_closing,
)

# GRDL internal
from grdl.image_processing.base import ImageTransform, BandwiseTransformMixin
from grdl.image_processing.params import Desc, Options, Range
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC

OPERATIONS = ('opening', 'closing', 'erosion', 'dilation')
COMBINATIONS = ('max', 'mean', 'median')


def _make_line_se(length: int, angle_deg: float) -> np.ndarray:
    """Create a line structuring element at a given angle.

    Parameters
    ----------
    length : int
        Length of the line in pixels (must be odd).
    angle_deg : float
        Angle in degrees (0 = horizontal).

    Returns
    -------
    np.ndarray
        2D bool array containing the line SE.
    """
    half = length // 2
    size = 2 * half + 1
    se = np.zeros((size, size), dtype=bool)

    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    for t in range(-half, half + 1):
        x = int(round(t * cos_a)) + half
        y = int(round(-t * sin_a)) + half
        if 0 <= x < size and 0 <= y < size:
            se[y, x] = True

    return se


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI],
                category=PC.BINARY)
@processor_version('1.6.0')
class DirectionalFilter(BandwiseTransformMixin, ImageTransform):
    """Directional morphological filtering, ported from MorphoLibJ.

    Applies morphological operations with oriented line structuring
    elements at evenly spaced angles, then combines results.

    Parameters
    ----------
    n_directions : int
        Number of evenly spaced angles. Default 12.
    line_length : int
        Length of line structuring element in pixels. Default 15.
    operation : str
        Morphological operation to apply at each angle. Default ``'opening'``.
    combination : str
        How to combine results across directions. Default ``'max'``.

    Notes
    -----
    Independent reimplementation of MorphoLibJ ``DirectionalFilter.java``
    (LGPL-3). Algorithm follows Soille, Breen & Jones (IEEE PAMI, 1996).

    Examples
    --------
    >>> from grdl_imagej import DirectionalFilter
    >>> df = DirectionalFilter(n_directions=12, line_length=15)
    >>> enhanced = df.apply(image)
    """

    __imagej_source__ = 'MorphoLibJ/morphology/directional/DirectionalFilter.java'
    __imagej_version__ = '1.6.0'
    __gpu_compatible__ = False

    n_directions: Annotated[int, Range(min=4, max=64),
                            Desc('Number of angles')] = 12
    line_length: Annotated[int, Range(min=3, max=101),
                           Desc('SE length in pixels')] = 15
    operation: Annotated[str, Options(*OPERATIONS),
                         Desc('Morphological operation')] = 'opening'
    combination: Annotated[str, Options(*COMBINATIONS),
                           Desc('Combination method')] = 'max'

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply directional filtering.

        Parameters
        ----------
        source : np.ndarray
            2D image. Shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Filtered image, dtype float64.
        """
        if source.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {source.shape}")

        p = self._resolve_params(kwargs)
        image = source.astype(np.float64)
        n_dirs = p['n_directions']
        length = p['line_length']
        if length % 2 == 0:
            length += 1  # Ensure odd

        op = p['operation']
        results = []

        for i in range(n_dirs):
            angle = 180.0 * i / n_dirs
            se = _make_line_se(length, angle)

            if op == 'opening':
                r = grey_opening(image, footprint=se)
            elif op == 'closing':
                r = grey_closing(image, footprint=se)
            elif op == 'erosion':
                r = minimum_filter(image, footprint=se)
            else:  # dilation
                r = maximum_filter(image, footprint=se)

            results.append(r)

        stack = np.array(results)
        combo = p['combination']
        if combo == 'max':
            return np.max(stack, axis=0)
        elif combo == 'mean':
            return np.mean(stack, axis=0)
        else:  # median
            return np.median(stack, axis=0)
