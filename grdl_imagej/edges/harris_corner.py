# -*- coding: utf-8 -*-
"""
Harris Corner Detection - Structure tensor-based corner/feature detection.

Detects corner points where the image gradient has significant variation
in multiple directions. Computes the structure tensor, derives the Harris
corner response function, and applies non-maximum suppression.

Particularly useful for:
- Feature detection for image co-registration
- Interest point detection for SAR/EO matching
- Pre-processing for feature-based alignment pipelines

Attribution
-----------
Algorithm: Harris & Stephens, "A Combined Corner and Edge Detector",
Alvey Vision Conference, 1988.
Related to structure tensor ops in ``imagej-ops``.

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
from scipy.ndimage import gaussian_filter, maximum_filter

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.params import Desc, Range
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI, IM.SWIR, IM.MWIR, IM.LWIR],
                category=PC.EDGES)
@processor_version('1.0.0')
class HarrisCornerDetector(ImageTransform):
    """Harris corner detector.

    Computes the Harris corner response:

    ``R = det(M) - k * trace(M)^2``

    where ``M`` is the structure tensor (second moment matrix) smoothed
    by a Gaussian, then applies thresholding and non-maximum suppression.

    Parameters
    ----------
    sigma : float
        Gaussian smoothing sigma for the structure tensor. Default is 1.5.
    k : float
        Harris free parameter. Typical range 0.04-0.06. Default is 0.04.
    threshold : float
        Corner response threshold as a fraction of the maximum response.
        Default is 0.01.
    nms_radius : int
        Non-maximum suppression radius. Default is 3.

    Notes
    -----
    Reference: Harris & Stephens, Alvey Vision Conference, 1988.

    The output is the corner response map after thresholding and NMS.
    Non-corner pixels are zero; corner pixels retain their response value.

    Examples
    --------
    >>> from grdl_imagej import HarrisCornerDetector
    >>> harris = HarrisCornerDetector(sigma=1.5, k=0.04, threshold=0.01)
    >>> corners = harris.apply(pan_image)
    """

    __imagej_source__ = 'imagej-ops/features/harris'
    __imagej_version__ = '1.0.0'
    __gpu_compatible__ = True

    sigma: Annotated[float, Range(min=0.5, max=5.0),
                      Desc('Gaussian sigma for structure tensor')] = 1.5
    k: Annotated[float, Range(min=0.01, max=0.15),
                  Desc('Harris free parameter')] = 0.04
    threshold: Annotated[float, Range(min=0.0, max=1.0),
                          Desc('Corner response threshold (fraction of max)')] = 0.01
    nms_radius: Annotated[int, Range(min=1, max=15),
                           Desc('Non-maximum suppression radius')] = 3

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Detect corners in a 2D image.

        Parameters
        ----------
        source : np.ndarray
            2D image array. Shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Corner response map (float64). Non-corner pixels are 0.

        Raises
        ------
        ValueError
            If source is not 2D.
        """
        if source.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {source.shape}")

        p = self._resolve_params(kwargs)
        image = source.astype(np.float64)

        # Compute image gradients (central differences)
        iy, ix = np.gradient(image)

        # Structure tensor components
        ixx = ix * ix
        iyy = iy * iy
        ixy = ix * iy

        # Smooth structure tensor with Gaussian
        sigma = p['sigma']
        sxx = gaussian_filter(ixx, sigma=sigma, mode='nearest')
        syy = gaussian_filter(iyy, sigma=sigma, mode='nearest')
        sxy = gaussian_filter(ixy, sigma=sigma, mode='nearest')

        # Harris response: R = det(M) - k * trace(M)^2
        det_m = sxx * syy - sxy * sxy
        trace_m = sxx + syy
        response = det_m - p['k'] * trace_m * trace_m

        # Threshold
        thresh = p['threshold'] * response.max() if response.max() > 0 else 0
        response[response < thresh] = 0.0

        # Non-maximum suppression
        nms_size = 2 * p['nms_radius'] + 1
        local_max = maximum_filter(response, size=nms_size, mode='nearest')
        response[response < local_max] = 0.0

        return response
