# -*- coding: utf-8 -*-
"""
Variance / Std Dev Filter - Local variance and standard deviation maps.

Computes local variance or standard deviation in a sliding window. Produces
texture/variability maps useful for detecting change regions, noise
estimation, and adaptive processing.

Attribution
-----------
Related to ``imagej-ops`` ``DefaultVariance.java``.
Repository: https://github.com/imagej/imagej-ops (BSD-2).
Standard statistical filtering.

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
from grdl.image_processing.params import Desc, Options, Range
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI, IM.HSI, IM.SWIR, IM.MWIR, IM.LWIR],
                category=PC.FILTERS)
@processor_version('1.0.0')
class VarianceFilter(ImageTransform):
    """Local variance / standard deviation filter.

    Computes ``Var = E[X^2] - E[X]^2`` using efficient uniform (box)
    filtering for both terms, then optionally takes the square root
    to produce standard deviation.

    Parameters
    ----------
    radius : int
        Window half-size. The full window is ``(2*radius+1) x (2*radius+1)``.
        Default is 3.
    output : str
        Output statistic: ``'variance'`` or ``'std_dev'``. Default is
        ``'std_dev'``.

    Notes
    -----
    Uses the identity ``Var(X) = E[X^2] - (E[X])^2`` with
    ``scipy.ndimage.uniform_filter`` for O(1)-per-pixel computation
    regardless of window size. Variance values are clamped to >= 0
    to handle floating-point rounding.

    Examples
    --------
    >>> from grdl_imagej import VarianceFilter
    >>> vf = VarianceFilter(radius=5, output='std_dev')
    >>> texture_map = vf.apply(sar_amplitude)
    """

    __imagej_source__ = 'imagej-ops/stats/DefaultVariance.java'
    __imagej_version__ = '1.0.0'
    __gpu_compatible__ = True

    radius: Annotated[int, Range(min=1, max=25),
                       Desc('Window half-size')] = 3
    output: Annotated[str, Options('variance', 'std_dev'),
                       Desc('Output statistic type')] = 'std_dev'

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute local variance or standard deviation.

        Parameters
        ----------
        source : np.ndarray
            2D image array. Shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Variance or standard deviation map (float64).

        Raises
        ------
        ValueError
            If source is not 2D.
        """
        if source.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {source.shape}")

        p = self._resolve_params(kwargs)
        win = 2 * p['radius'] + 1

        image = source.astype(np.float64)
        mean = uniform_filter(image, size=win, mode='nearest')
        mean_sq = uniform_filter(image * image, size=win, mode='nearest')

        variance = np.maximum(mean_sq - mean * mean, 0.0)

        if p['output'] == 'std_dev':
            return np.sqrt(variance)
        return variance
