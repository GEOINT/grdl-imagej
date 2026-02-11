# -*- coding: utf-8 -*-
"""
Gabor Filter Bank - Multi-orientation, multi-frequency texture filters.

Applies a bank of Gabor filters at multiple orientations and frequencies.
Each filter is a Gaussian-modulated sinusoidal plane wave sensitive to a
specific spatial frequency and orientation. Essential for texture
classification in land cover mapping.

Attribution
-----------
Algorithm: Jain & Farrokhnia, "Unsupervised Texture Segmentation Using
Gabor Filters", Pattern Recognition, 24(12), 1991.
Java source: Fiji Trainable Segmentation — ``GaborFilter.java``.
Repository: https://github.com/fiji/Trainable_Segmentation (GPL-2,
independent reimplementation).

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
from grdl.image_processing.params import Desc, Range
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC


def _build_gabor_kernel(
    sigma: float, theta: float, lambda_: float, gamma: float, psi: float
) -> np.ndarray:
    """Build a 2D Gabor kernel.

    Parameters
    ----------
    sigma : float
        Gaussian envelope width.
    theta : float
        Orientation angle in radians.
    lambda_ : float
        Wavelength of the sinusoidal component.
    gamma : float
        Spatial aspect ratio.
    psi : float
        Phase offset.

    Returns
    -------
    np.ndarray
        2D Gabor kernel.
    """
    # Kernel size: 3 sigma in each direction
    half = int(np.ceil(3.0 * sigma))
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)

    # Rotation
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x_theta = x * cos_t + y * sin_t
    y_theta = -x * sin_t + y * cos_t

    # Gabor function
    gaussian = np.exp(
        -(x_theta ** 2 + gamma ** 2 * y_theta ** 2) / (2.0 * sigma ** 2)
    )
    sinusoidal = np.cos(2.0 * np.pi * x_theta / lambda_ + psi)

    return gaussian * sinusoidal


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI, IM.HSI, IM.SWIR, IM.MWIR, IM.LWIR],
                category=PC.FILTERS)
@processor_version('1.0.0')
class GaborFilterBank(ImageTransform):
    """Gabor filter bank for texture analysis.

    Applies Gabor filters at ``n_orientations`` evenly spaced angles
    in [0, pi) and returns the maximum response across all orientations
    at each pixel (texture energy map).

    Parameters
    ----------
    sigma : float
        Gaussian envelope width. Default is 3.0.
    n_orientations : int
        Number of orientation angles spanning [0, pi). Default is 8.
    lambda_ : float
        Wavelength of the sinusoidal component. Default is 10.0.
    gamma : float
        Spatial aspect ratio. Default is 0.5.
    psi : float
        Phase offset in radians. Default is 0.0.

    Notes
    -----
    Reference: Jain & Farrokhnia, Pattern Recognition, 24(12), 1991.
    Independent reimplementation of Fiji's ``GaborFilter.java`` (GPL-2).

    The output is the maximum Gabor response across all orientations,
    which represents the dominant texture energy at each pixel.

    Examples
    --------
    >>> from grdl_imagej import GaborFilterBank
    >>> gabor = GaborFilterBank(sigma=3.0, n_orientations=8, lambda_=10.0)
    >>> texture_energy = gabor.apply(msi_band)
    """

    __imagej_source__ = 'fiji/Trainable_Segmentation/GaborFilter.java'
    __imagej_version__ = '1.0.0'
    __gpu_compatible__ = False

    sigma: Annotated[float, Range(min=1.0, max=20.0),
                      Desc('Gaussian envelope width')] = 3.0
    n_orientations: Annotated[int, Range(min=2, max=32),
                               Desc('Number of orientations')] = 8
    lambda_: Annotated[float, Range(min=2.0, max=50.0),
                        Desc('Wavelength of sinusoidal component')] = 10.0
    gamma: Annotated[float, Range(min=0.1, max=1.0),
                      Desc('Spatial aspect ratio')] = 0.5
    psi: Annotated[float, Range(min=0.0, max=3.14159),
                    Desc('Phase offset in radians')] = 0.0

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply Gabor filter bank to a 2D image.

        Parameters
        ----------
        source : np.ndarray
            2D image array. Shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Maximum Gabor response across orientations (float64).

        Raises
        ------
        ValueError
            If source is not 2D.
        """
        if source.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {source.shape}")

        p = self._resolve_params(kwargs)
        image = source.astype(np.float64)

        n_orient = p['n_orientations']
        max_response = np.zeros_like(image)

        for k in range(n_orient):
            theta = np.pi * k / n_orient
            kernel = _build_gabor_kernel(
                p['sigma'], theta, p['lambda_'], p['gamma'], p['psi']
            )
            response = convolve(image, kernel, mode='nearest')
            # Take absolute value for energy
            np.maximum(max_response, np.abs(response), out=max_response)

        return max_response
