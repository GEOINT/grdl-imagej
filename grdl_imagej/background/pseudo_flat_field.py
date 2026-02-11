# -*- coding: utf-8 -*-
"""
Pseudo Flat-Field Correction - Illumination normalization via Gaussian division.

Corrects uneven illumination (vignetting, shading) by dividing the image
by a heavily blurred version of itself. Simpler and faster than rolling
ball for illumination normalization.

Particularly useful for:
- Correcting vignetting in aerial/satellite imagery
- Normalizing uneven illumination in microscopy-derived images
- Pre-processing for thresholding when illumination is non-uniform
- Flat-field correction without a reference flat image

Attribution
-----------
Standard flat-field correction technique. Related to ImageJ macro-based
``Pseudo_Flat_Field_Correction.java``.
Reference: Model, "Intensity calibration and flat-field correction for
fluorescence microscopes", Current Protocols in Cytometry, 2001.

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
from scipy.ndimage import gaussian_filter

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.params import Desc, Range
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI, IM.HSI, IM.SWIR, IM.MWIR, IM.LWIR],
                category=PC.BACKGROUND)
@processor_version('1.0.0')
class PseudoFlatField(ImageTransform):
    """Pseudo flat-field illumination correction.

    Divides the image by a heavily Gaussian-blurred version of itself
    to remove low-frequency illumination gradients.

    ``output = input / GaussianBlur(input, blur_radius)``

    Parameters
    ----------
    blur_radius : float
        Gaussian sigma for estimating the illumination field. Should
        be large enough to blur out all image features, leaving only
        the illumination gradient. Default is 50.0.
    normalize_output : bool
        If True, normalize the corrected image to [0, 1]. Default is True.

    Notes
    -----
    The blur radius should be significantly larger than the largest
    feature of interest. A value of 50-200 pixels is typical for
    satellite imagery.

    Division by near-zero values is handled by adding a small epsilon
    to the denominator.

    Examples
    --------
    >>> from grdl_imagej import PseudoFlatField
    >>> pff = PseudoFlatField(blur_radius=100.0)
    >>> corrected = pff.apply(vignetted_image)
    """

    __imagej_source__ = 'ij/plugin/filter/Pseudo_Flat_Field_Correction.java'
    __imagej_version__ = '1.0.0'
    __gpu_compatible__ = True

    blur_radius: Annotated[float, Range(min=10.0, max=500.0),
                            Desc('Gaussian sigma for illumination estimation')] = 50.0
    normalize_output: Annotated[bool, Desc('Normalize output to [0, 1]')] = True

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply pseudo flat-field correction to a 2D image.

        Parameters
        ----------
        source : np.ndarray
            2D image array. Shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Corrected image (float64), same shape as input.

        Raises
        ------
        ValueError
            If source is not 2D.
        """
        if source.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {source.shape}")

        p = self._resolve_params(kwargs)

        image = source.astype(np.float64)
        background = gaussian_filter(image, sigma=p['blur_radius'], mode='nearest')

        # Divide with epsilon to avoid division by zero
        eps = np.finfo(np.float64).eps
        corrected = image / (background + eps)

        if p['normalize_output']:
            lo, hi = corrected.min(), corrected.max()
            if hi - lo > eps:
                corrected = (corrected - lo) / (hi - lo)
            else:
                corrected = np.zeros_like(corrected)

        return corrected
