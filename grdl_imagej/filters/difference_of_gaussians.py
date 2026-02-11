# -*- coding: utf-8 -*-
"""
Difference of Gaussians (DoG) - Band-pass filter via Gaussian subtraction.

Subtracts two Gaussian-blurred images at different scales, approximating
the Laplacian of Gaussian (LoG). Key building block for blob detection
and scale-space analysis in feature extraction pipelines.

Particularly useful for:
- Blob detection in PAN/EO satellite imagery
- Scale-space approximation for multi-scale analysis
- Band-pass filtering to isolate features at a specific spatial scale
- Pre-processing for feature detection in SAR amplitude images

Attribution
-----------
Algorithm: Marr & Hildreth, "Theory of Edge Detection", Proc. Royal
Society London B, 207, 1980.
Java source: ``imagej-ops`` — ``DefaultDoG.java``.
Repository: https://github.com/imagej/imagej-ops (BSD-2).

Dependencies
------------
numpy
scipy (via existing GaussianBlur)

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
from scipy.ndimage import gaussian_filter

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.params import Desc, Range
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI, IM.HSI, IM.SWIR, IM.MWIR, IM.LWIR],
                category=PC.FILTERS)
@processor_version('1.0.0')
class DifferenceOfGaussians(ImageTransform):
    """Difference of Gaussians (DoG) band-pass filter.

    Computes ``DoG = GaussianBlur(image, sigma1) - GaussianBlur(image, sigma2)``
    where ``sigma1 < sigma2``. The result highlights features at scales
    between the two blur radii.

    Parameters
    ----------
    sigma1 : float
        Smaller Gaussian sigma (fine scale). Default is 1.0.
    sigma2 : float
        Larger Gaussian sigma (coarse scale). Must be > sigma1.
        Default is 3.0.

    Notes
    -----
    Based on ``imagej-ops`` ``DefaultDoG.java`` (BSD-2). The DoG is a
    well-known approximation to the Laplacian of Gaussian (LoG) and is
    used extensively in SIFT-style feature detection.

    If ``sigma1 >= sigma2``, the sigmas are automatically swapped.

    Examples
    --------
    >>> from grdl_imagej import DifferenceOfGaussians
    >>> dog = DifferenceOfGaussians(sigma1=1.0, sigma2=3.0)
    >>> blob_response = dog.apply(pan_image)
    """

    __imagej_source__ = 'imagej-ops/filter/dog/DefaultDoG.java'
    __imagej_version__ = '1.0.0'
    __gpu_compatible__ = True

    sigma1: Annotated[float, Range(min=0.1, max=20.0),
                       Desc('Smaller Gaussian sigma (fine scale)')] = 1.0
    sigma2: Annotated[float, Range(min=0.1, max=40.0),
                       Desc('Larger Gaussian sigma (coarse scale)')] = 3.0

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply Difference of Gaussians to a 2D image.

        Parameters
        ----------
        source : np.ndarray
            2D image array. Shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            DoG response image (float64), same shape as input.
            Positive values indicate bright blobs, negative indicate
            dark blobs at the selected scale.

        Raises
        ------
        ValueError
            If source is not 2D.
        """
        if source.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {source.shape}")

        p = self._resolve_params(kwargs)
        s1, s2 = p['sigma1'], p['sigma2']

        # Ensure sigma1 < sigma2
        if s1 > s2:
            s1, s2 = s2, s1

        image = source.astype(np.float64)
        blur1 = gaussian_filter(image, sigma=s1, mode='nearest')
        blur2 = gaussian_filter(image, sigma=s2, mode='nearest')

        return blur1 - blur2
