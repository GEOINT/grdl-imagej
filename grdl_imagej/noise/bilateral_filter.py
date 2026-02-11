# -*- coding: utf-8 -*-
"""
Bilateral Filter - Edge-preserving smoothing.

Combines a spatial Gaussian kernel with a range (intensity) Gaussian
kernel so that only nearby pixels with similar intensities contribute
to the average. Smooths homogeneous regions while preserving edges.

Particularly useful for:
- SAR speckle reduction while preserving target boundaries
- Pre-processing before thresholding or segmentation
- Noise reduction in PAN/EO imagery without blurring edges
- Thermal image smoothing for feature extraction

Attribution
-----------
Algorithm: Tomasi & Manduchi, "Bilateral Filtering for Gray and Color
Images", ICCV 1998.
Java source: ``imagej-ops`` — ``DefaultBilateralFilter.java``.
Repository: https://github.com/imagej/imagej-ops (BSD-2).

Dependencies
------------
numpy

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
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.params import Desc, Range
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI, IM.HSI, IM.SWIR, IM.MWIR, IM.LWIR],
                category=PC.NOISE)
@processor_version('1.0.0')
class BilateralFilter(ImageTransform):
    """Bilateral edge-preserving smoothing filter.

    For each pixel, computes a weighted average of neighbors where
    ``weight = G_spatial(distance) * G_range(|intensity_diff|)``.

    Parameters
    ----------
    sigma_spatial : float
        Spatial Gaussian standard deviation (neighborhood size).
        Larger values include more distant pixels. Default is 3.0.
    sigma_range : float
        Intensity Gaussian standard deviation (edge sensitivity).
        Larger values allow more intensity difference before the
        weight drops. Default is 30.0.
    radius : int
        Kernel radius in pixels. The full window is
        ``(2*radius+1) x (2*radius+1)``. Default is 5.

    Notes
    -----
    Based on ``imagej-ops`` ``DefaultBilateralFilter.java`` (BSD-2).
    This implementation uses the direct (non-separable) bilateral
    formulation: O(n * r^2) per pixel where r is the kernel radius.

    References: Tomasi & Manduchi, ICCV 1998.

    Examples
    --------
    >>> from grdl_imagej import BilateralFilter
    >>> bf = BilateralFilter(sigma_spatial=3.0, sigma_range=30.0)
    >>> smoothed = bf.apply(noisy_sar)
    """

    __imagej_source__ = 'imagej-ops/filter/bilateral/DefaultBilateralFilter.java'
    __imagej_version__ = '1.0.0'
    __gpu_compatible__ = False

    sigma_spatial: Annotated[float, Range(min=0.5, max=50.0),
                              Desc('Spatial Gaussian std dev')] = 3.0
    sigma_range: Annotated[float, Range(min=1.0, max=255.0),
                            Desc('Intensity Gaussian std dev')] = 30.0
    radius: Annotated[int, Range(min=1, max=25),
                       Desc('Kernel radius in pixels')] = 5

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply bilateral filter to a 2D image.

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
        sigma_s = p['sigma_spatial']
        sigma_r = p['sigma_range']
        r = p['radius']

        image = source.astype(np.float64)
        rows, cols = image.shape

        # Pre-compute spatial Gaussian weights
        y, x = np.ogrid[-r:r + 1, -r:r + 1]
        spatial_weights = np.exp(-(x * x + y * y) / (2.0 * sigma_s * sigma_s))

        # Pad image for border handling
        padded = np.pad(image, r, mode='edge')

        result = np.empty_like(image)
        inv_2sigma_r_sq = 1.0 / (2.0 * sigma_r * sigma_r)

        for i in range(rows):
            for j in range(cols):
                # Extract neighborhood
                patch = padded[i:i + 2 * r + 1, j:j + 2 * r + 1]
                center_val = image[i, j]

                # Range weights
                diff = patch - center_val
                range_weights = np.exp(-diff * diff * inv_2sigma_r_sq)

                # Combined weights
                weights = spatial_weights * range_weights
                w_sum = weights.sum()

                if w_sum > 0:
                    result[i, j] = (weights * patch).sum() / w_sum
                else:
                    result[i, j] = center_val

        return result
