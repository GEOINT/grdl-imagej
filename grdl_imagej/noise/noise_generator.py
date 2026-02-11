# -*- coding: utf-8 -*-
"""
Noise Generator - Adds synthetic noise to images.

Supports Gaussian, Poisson (shot noise), salt-and-pepper, and speckle
(multiplicative) noise models. Essential for testing denoising algorithms
and data augmentation.

Attribution
-----------
ImageJ implementation: ``ij/plugin/filter/Filters.java``,
``ij/plugin/Noise.java`` in ImageJ 1.54j.
ImageJ 1.x source is in the public domain.
Related: ``imagej-ops`` — ``image/noise/`` (BSD-2).

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
from typing import Annotated, Any, Optional

# Third-party
import numpy as np

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.params import Desc, Options, Range
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI, IM.HSI, IM.SWIR, IM.MWIR, IM.LWIR],
                category=PC.NOISE)
@processor_version('1.54j')
class NoiseGenerator(ImageTransform):
    """Synthetic noise generator for image corruption/augmentation.

    Adds various types of synthetic noise to images for testing
    denoising algorithms and data augmentation.

    Parameters
    ----------
    noise_type : str
        Noise model: ``'gaussian'``, ``'poisson'``, ``'salt_pepper'``,
        or ``'speckle'``. Default is ``'gaussian'``.
    sigma : float
        Standard deviation for Gaussian and speckle noise. Default is 25.0.
    density : float
        Pixel density for salt-and-pepper noise (fraction of affected
        pixels). Default is 0.05.
    seed : int or None
        Random seed for reproducibility. Default is None (non-deterministic).

    Notes
    -----
    - **Gaussian**: ``output = input + N(0, sigma)``
    - **Poisson**: ``output = Poisson(input)`` (shot noise)
    - **Salt-pepper**: random pixels set to min/max at given density
    - **Speckle**: ``output = input + input * N(0, sigma / 255)``

    Output is **not** clipped — callers can clip if needed.

    Examples
    --------
    >>> from grdl_imagej import NoiseGenerator
    >>> ng = NoiseGenerator(noise_type='gaussian', sigma=25.0, seed=42)
    >>> noisy = ng.apply(clean_image)
    """

    __imagej_source__ = 'ij/plugin/Noise.java'
    __imagej_version__ = '1.54j'
    __gpu_compatible__ = False

    noise_type: Annotated[str, Options('gaussian', 'poisson', 'salt_pepper', 'speckle'),
                           Desc('Noise model')] = 'gaussian'
    sigma: Annotated[float, Range(min=0.1, max=100.0),
                      Desc('Std dev for Gaussian/speckle noise')] = 25.0
    density: Annotated[float, Range(min=0.0, max=0.5),
                        Desc('Density for salt-and-pepper noise')] = 0.05
    seed: Annotated[Optional[int], Desc('Random seed for reproducibility')] = None

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Add synthetic noise to a 2D image.

        Parameters
        ----------
        source : np.ndarray
            2D image array. Shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Noisy image (float64), same shape as input.

        Raises
        ------
        ValueError
            If source is not 2D.
        """
        if source.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {source.shape}")

        p = self._resolve_params(kwargs)
        rng = np.random.default_rng(p['seed'])
        image = source.astype(np.float64)
        noise_type = p['noise_type']

        if noise_type == 'gaussian':
            noise = rng.normal(0.0, p['sigma'], image.shape)
            return image + noise

        elif noise_type == 'poisson':
            # Poisson noise requires non-negative values
            safe = np.maximum(image, 0.0)
            return rng.poisson(safe).astype(np.float64)

        elif noise_type == 'salt_pepper':
            result = image.copy()
            density = p['density']
            # Determine salt/pepper values: use image range if dynamic,
            # else fall back to 0/255 for flat images
            lo, hi = image.min(), image.max()
            if hi - lo < 1e-10:
                lo, hi = 0.0, 255.0
            # Salt (high value)
            salt_mask = rng.random(image.shape) < density / 2
            result[salt_mask] = hi
            # Pepper (low value)
            pepper_mask = rng.random(image.shape) < density / 2
            result[pepper_mask] = lo
            return result

        else:  # speckle
            noise = rng.normal(0.0, p['sigma'] / 255.0, image.shape)
            return image + image * noise
