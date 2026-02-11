# -*- coding: utf-8 -*-
"""
Kuwahara Filter - Edge-preserving smoothing with quadrant selection.

Divides each pixel's neighborhood into four overlapping quadrants, computes
the mean and variance of each, and assigns the pixel the mean of the
quadrant with minimum variance. Produces smooth regions with sharp edges.

Attribution
-----------
Algorithm: Kuwahara et al., "Processing of RI-angiocardiographic images",
Digital Processing of Biomedical Images, Plenum Press, 1976.
Related to Fiji plugin implementations.

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
from scipy.ndimage import uniform_filter

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.params import Desc, Range
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI, IM.HSI, IM.SWIR, IM.MWIR, IM.LWIR],
                category=PC.FILTERS)
@processor_version('1.0.0')
class KuwaharaFilter(ImageTransform):
    """Kuwahara edge-preserving smoothing filter.

    For each pixel, divides the ``(2r+1) x (2r+1)`` neighborhood into
    four overlapping ``(r+1) x (r+1)`` quadrant sub-windows. Computes
    the mean and variance of each quadrant. The output pixel is assigned
    the mean of the quadrant with the lowest variance.

    Parameters
    ----------
    radius : int
        Window half-size. The full window is ``(2*radius+1) x (2*radius+1)``
        and each quadrant is ``(radius+1) x (radius+1)``. Default is 3.

    Notes
    -----
    Reference: Kuwahara et al. (1976). Produces characteristic
    "painting-like" smoothing with sharp, well-preserved edges.

    For SAR imagery, use moderate radius (3-5) to reduce speckle
    while maintaining target boundaries.

    Examples
    --------
    >>> from grdl_imagej import KuwaharaFilter
    >>> kf = KuwaharaFilter(radius=3)
    >>> smoothed = kf.apply(noisy_image)
    """

    __imagej_source__ = 'fiji/plugin/filter/Kuwahara_Filter.java'
    __imagej_version__ = '1.0.0'
    __gpu_compatible__ = False

    radius: Annotated[int, Range(min=1, max=15),
                       Desc('Window half-size')] = 3

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply Kuwahara filter to a 2D image.

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

        p = self._resolve_params(kwargs)
        r = p['radius']
        image = source.astype(np.float64)
        rows, cols = image.shape

        # Pad image
        padded = np.pad(image, r, mode='edge')

        # Quadrant size
        q = r + 1

        result = np.empty_like(image)

        for i in range(rows):
            for j in range(cols):
                pi, pj = i + r, j + r  # padded indices

                # Four overlapping quadrants (top-left, top-right,
                # bottom-left, bottom-right) each of size (r+1) x (r+1)
                quads = [
                    padded[pi - r:pi + 1, pj - r:pj + 1],  # top-left
                    padded[pi - r:pi + 1, pj:pj + q],       # top-right
                    padded[pi:pi + q, pj - r:pj + 1],       # bottom-left
                    padded[pi:pi + q, pj:pj + q],            # bottom-right
                ]

                best_mean = 0.0
                best_var = np.inf
                for quad in quads:
                    m = quad.mean()
                    v = quad.var()
                    if v < best_var:
                        best_var = v
                        best_mean = m

                result[i, j] = best_mean

        return result
