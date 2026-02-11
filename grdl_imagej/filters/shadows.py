# -*- coding: utf-8 -*-
"""
Shadows (Emboss) - Directional derivative / emboss effects.

Applies directional shadow/emboss effects using 3x3 directional derivative
kernels. Eight directional options (N, NE, E, SE, S, SW, W, NW). Useful
for terrain feature enhancement in DEM-derived imagery.

Attribution
-----------
ImageJ implementation: ``ij/plugin/filter/Shadows.java`` in ImageJ 1.54j.
ImageJ 1.x source is in the public domain.

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
from scipy.ndimage import convolve

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.params import Desc, Options, Range
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC

# ImageJ's 3x3 directional kernels for the Shadows command
_KERNELS = {
    'N':  np.array([[ 1,  2,  1], [ 0,  1,  0], [-1, -2, -1]], dtype=np.float64),
    'NE': np.array([[ 0,  1,  2], [-1,  1,  1], [-2, -1,  0]], dtype=np.float64),
    'E':  np.array([[-1,  0,  1], [-2,  1,  2], [-1,  0,  1]], dtype=np.float64),
    'SE': np.array([[-2, -1,  0], [-1,  1,  1], [ 0,  1,  2]], dtype=np.float64),
    'S':  np.array([[-1, -2, -1], [ 0,  1,  0], [ 1,  2,  1]], dtype=np.float64),
    'SW': np.array([[ 0, -1, -2], [ 1,  1, -1], [ 2,  1,  0]], dtype=np.float64),
    'W':  np.array([[ 1,  0, -1], [ 2,  1, -2], [ 1,  0, -1]], dtype=np.float64),
    'NW': np.array([[ 2,  1,  0], [ 1,  1, -1], [ 0, -1, -2]], dtype=np.float64),
}


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI, IM.SWIR, IM.MWIR, IM.LWIR],
                category=PC.FILTERS)
@processor_version('1.54j')
class Shadows(ImageTransform):
    """Directional shadow/emboss filter, ported from ImageJ 1.54j.

    Applies a 3x3 directional derivative kernel to produce emboss-like
    effects. An offset is added so that flat areas appear as mid-gray.

    Parameters
    ----------
    direction : str
        Shadow direction. One of 'N', 'NE', 'E', 'SE', 'S', 'SW',
        'W', 'NW'. Default is 'SE'.
    offset : float
        Output offset added after convolution. Default is 128.0 for
        8-bit-scale imagery so flat regions appear mid-gray.

    Notes
    -----
    Port of ``ij/plugin/filter/Shadows.java`` from ImageJ 1.54j
    (public domain). Each direction uses a predefined 3x3 kernel
    that computes a directional first derivative with an asymmetric
    weighting.

    Examples
    --------
    >>> from grdl_imagej import Shadows
    >>> emboss = Shadows(direction='SE', offset=128.0)
    >>> result = emboss.apply(dem_hillshade)
    """

    __imagej_source__ = 'ij/plugin/filter/Shadows.java'
    __imagej_version__ = '1.54j'
    __gpu_compatible__ = True

    direction: Annotated[str, Options('N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'),
                          Desc('Shadow direction')] = 'SE'
    offset: Annotated[float, Range(min=0.0, max=255.0),
                       Desc('Output offset (mid-gray for flat areas)')] = 128.0

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply directional shadow/emboss to a 2D image.

        Parameters
        ----------
        source : np.ndarray
            2D image array. Shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Embossed image (float64), same shape as input.

        Raises
        ------
        ValueError
            If source is not 2D.
        """
        if source.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {source.shape}")

        p = self._resolve_params(kwargs)
        kernel = _KERNELS[p['direction']]
        image = source.astype(np.float64)

        result = convolve(image, kernel, mode='nearest')
        result += p['offset']

        return result
