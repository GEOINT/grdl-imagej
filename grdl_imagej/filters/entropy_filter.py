# -*- coding: utf-8 -*-
"""
Entropy Filter - Local Shannon entropy for texture detection.

Computes local Shannon entropy within a sliding window. Homogeneous
regions have low entropy; textured/edge regions have high entropy.
Useful as a texture feature for land cover classification.

Attribution
-----------
Related to ``imagej-ops`` ``DefaultEntropy.java``.
Repository: https://github.com/imagej/imagej-ops (BSD-2).
Reference: Shannon, "A Mathematical Theory of Communication",
Bell System Technical Journal, 27(3), 1948.

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
from scipy.ndimage import generic_filter

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.params import Desc, Range
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC


def _local_entropy(values: np.ndarray, n_bins: int) -> float:
    """Compute Shannon entropy of a flattened neighborhood."""
    hist, _ = np.histogram(values, bins=n_bins, range=(0.0, 256.0))
    # Normalize to probabilities
    total = hist.sum()
    if total == 0:
        return 0.0
    probs = hist[hist > 0].astype(np.float64) / total
    return -np.sum(probs * np.log2(probs))


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI, IM.HSI, IM.SWIR, IM.MWIR, IM.LWIR],
                category=PC.FILTERS)
@processor_version('1.0.0')
class EntropyFilter(ImageTransform):
    """Local Shannon entropy filter for texture mapping.

    For each pixel, computes the Shannon entropy of the intensity
    histogram within a square sliding window.

    Parameters
    ----------
    radius : int
        Window half-size. Full window is ``(2*radius+1) x (2*radius+1)``.
        Default is 5.
    n_bins : int
        Number of histogram bins for local probability estimation.
        Default is 256.

    Notes
    -----
    Based on ``imagej-ops`` ``DefaultEntropy.java`` (BSD-2).

    Entropy is computed as ``H = -sum(p * log2(p))`` where ``p`` is
    the normalized histogram of pixel values in the local window.

    For float images, values are assumed to be in [0, 255] range for
    histogram binning. Scale your data accordingly.

    Examples
    --------
    >>> from grdl_imagej import EntropyFilter
    >>> ef = EntropyFilter(radius=5, n_bins=256)
    >>> entropy_map = ef.apply(pan_image)
    """

    __imagej_source__ = 'imagej-ops/stats/DefaultEntropy.java'
    __imagej_version__ = '1.0.0'
    __gpu_compatible__ = False

    radius: Annotated[int, Range(min=1, max=25),
                       Desc('Window half-size')] = 5
    n_bins: Annotated[int, Range(min=16, max=256),
                       Desc('Histogram bins for local probability')] = 256

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute local entropy map of a 2D image.

        Parameters
        ----------
        source : np.ndarray
            2D image array. Shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Entropy map (float64), same shape as input. Values are
            in bits (log base 2).

        Raises
        ------
        ValueError
            If source is not 2D.
        """
        if source.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {source.shape}")

        p = self._resolve_params(kwargs)
        win = 2 * p['radius'] + 1
        n_bins = p['n_bins']

        image = source.astype(np.float64)

        def _entropy_func(values):
            return _local_entropy(values, n_bins)

        return generic_filter(image, _entropy_func, size=win, mode='nearest')
