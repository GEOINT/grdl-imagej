# -*- coding: utf-8 -*-
"""
Morphological Gradient - Beucher, internal, and external gradients.

Morphological gradient as difference between dilation and erosion.
Three variants: Beucher (dilation - erosion), internal (original -
erosion), external (dilation - original).

Attribution
-----------
Algorithm: Serra, "Image Analysis and Mathematical Morphology",
Academic Press, 1982.

MorphoLibJ implementation:
``src/main/java/inra/ijpb/morphology/Morphology.java`` (gradient methods)
Source: https://github.com/ijpb/MorphoLibJ (LGPL-3).
This is an independent NumPy reimplementation.

Dependencies
------------
numpy
Reuses existing MorphologicalFilter.

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
from grdl.image_processing.base import ImageTransform, BandwiseTransformMixin
from grdl.image_processing.params import Desc, Options, Range
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC

from grdl_imagej.binary.morphology import MorphologicalFilter

GRADIENT_TYPES = ('beucher', 'internal', 'external')
SE_SHAPES = ('disk', 'square', 'diamond')


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI],
                category=PC.BINARY)
@processor_version('1.6.0')
class MorphologicalGradient(BandwiseTransformMixin, ImageTransform):
    """Morphological gradient, ported from MorphoLibJ.

    Computes gradient as difference of dilation and erosion.

    Parameters
    ----------
    se_shape : str
        Structuring element shape. Default ``'disk'``.
    se_radius : int
        Structuring element radius. Default 1.
    type : str
        Gradient type: ``'beucher'`` (dilate - erode), ``'internal'``
        (original - erode), ``'external'`` (dilate - original).
        Default ``'beucher'``.

    Notes
    -----
    Independent reimplementation of MorphoLibJ gradient methods (LGPL-3).
    Wraps the existing ``MorphologicalFilter`` processor.

    Examples
    --------
    >>> from grdl_imagej import MorphologicalGradient
    >>> mg = MorphologicalGradient(type='beucher', se_radius=1)
    >>> edges = mg.apply(image)
    """

    __imagej_source__ = 'MorphoLibJ/morphology/Morphology.java'
    __imagej_version__ = '1.6.0'
    __gpu_compatible__ = False

    se_shape: Annotated[str, Options(*SE_SHAPES),
                        Desc('Structuring element shape')] = 'disk'
    se_radius: Annotated[int, Range(min=1, max=15),
                         Desc('Structuring element radius')] = 1
    type: Annotated[str, Options(*GRADIENT_TYPES),
                    Desc('Gradient type')] = 'beucher'

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute morphological gradient.

        Parameters
        ----------
        source : np.ndarray
            2D image. Shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Gradient image, dtype float64.
        """
        if source.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {source.shape}")

        p = self._resolve_params(kwargs)
        image = source.astype(np.float64)

        # Map diamond -> cross for MorphologicalFilter
        shape = p['se_shape'] if p['se_shape'] != 'diamond' else 'cross'

        dilator = MorphologicalFilter(operation='dilate', radius=p['se_radius'],
                                      kernel_shape=shape)
        eroder = MorphologicalFilter(operation='erode', radius=p['se_radius'],
                                     kernel_shape=shape)

        gtype = p['type']
        if gtype == 'beucher':
            return dilator.apply(image) - eroder.apply(image)
        elif gtype == 'internal':
            return image - eroder.apply(image)
        else:  # external
            return dilator.apply(image) - image
