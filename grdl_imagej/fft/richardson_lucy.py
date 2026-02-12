# -*- coding: utf-8 -*-
"""
Richardson-Lucy Deconvolution - Iterative ML deconvolution.

Iterative maximum-likelihood deconvolution restoring images blurred by
a known PSF. Each iteration multiplies the current estimate by a
correction factor derived from the observed/re-blurred ratio.

Attribution
-----------
Algorithm: Richardson, "Bayesian-Based Iterative Method of Image
Restoration", JOSA, 62(1), 1972. Lucy, "An iterative technique for
the rectification of observed distributions", The Astronomical
Journal, 79, 1974.

imagej-ops implementation:
``src/main/java/net/imagej/ops/deconvolve/RichardsonLucyC.java``
Source: https://github.com/imagej/imagej-ops (BSD-2).
This is an independent NumPy reimplementation following the published algorithm.

Dependencies
------------
numpy
scipy

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
from scipy.signal import fftconvolve

# GRDL internal
from grdl.image_processing.base import ImageTransform, BandwiseTransformMixin
from grdl.image_processing.params import Desc, Range
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI, IM.LWIR, IM.MWIR],
                category=PC.FFT)
@processor_version('0.40.0')
class RichardsonLucy(BandwiseTransformMixin, ImageTransform):
    """Richardson-Lucy iterative deconvolution, ported from imagej-ops.

    Restores images blurred by a known point spread function using
    iterative maximum-likelihood estimation.

    The PSF must be passed via the ``psf`` keyword argument.

    Parameters
    ----------
    n_iterations : int
        Number of RL iterations. Default 20.
    regularization : float
        Tikhonov-Miller regularization strength. 0 = no regularization.
        Default 0.0.
    non_circulant : bool
        If True, apply edge correction for non-circulant boundary
        conditions. Default False.

    Notes
    -----
    Independent reimplementation of imagej-ops ``RichardsonLucyC.java``
    (BSD-2). Follows Richardson (1972) and Lucy (1974).

    Each iteration:
      estimate *= correlate(observed / convolve(estimate, PSF), PSF_flipped)

    Uses FFT-based convolution for speed.

    Examples
    --------
    >>> from grdl_imagej import RichardsonLucy
    >>> rl = RichardsonLucy(n_iterations=30)
    >>> restored = rl.apply(blurred_image, psf=point_spread_function)
    """

    __imagej_source__ = 'imagej-ops/deconvolve/RichardsonLucyC.java'
    __imagej_version__ = '0.40.0'
    __gpu_compatible__ = True

    n_iterations: Annotated[int, Range(min=1, max=200),
                            Desc('Number of RL iterations')] = 20
    regularization: Annotated[float, Range(min=0.0, max=0.1),
                              Desc('Tikhonov regularization')] = 0.0
    non_circulant: Annotated[bool, Desc('Non-circulant edge handling')] = False

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply Richardson-Lucy deconvolution.

        Parameters
        ----------
        source : np.ndarray
            2D blurred image. Shape ``(rows, cols)``.
        psf : np.ndarray
            2D point spread function.

        Returns
        -------
        np.ndarray
            Restored image, dtype float64.

        Raises
        ------
        ValueError
            If source is not 2D or psf is not provided.
        """
        if source.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {source.shape}")

        p = self._resolve_params(kwargs)

        psf = kwargs.get('psf', None)
        if psf is None:
            raise ValueError("'psf' keyword argument is required")
        psf = np.asarray(psf, dtype=np.float64)

        observed = source.astype(np.float64)
        # Normalize PSF
        psf_sum = psf.sum()
        if psf_sum > 0:
            psf = psf / psf_sum

        psf_flipped = psf[::-1, ::-1]
        estimate = observed.copy()
        eps = 1e-12

        # Non-circulant correction factor
        if p['non_circulant']:
            ones = np.ones_like(observed)
            correction = fftconvolve(ones, psf, mode='same')
            correction = np.maximum(correction, eps)
        else:
            correction = None

        for _ in range(p['n_iterations']):
            # Re-blur estimate
            reblurred = fftconvolve(estimate, psf, mode='same')
            reblurred = np.maximum(reblurred, eps)

            # Ratio
            ratio = observed / reblurred

            # Correlate with flipped PSF
            update = fftconvolve(ratio, psf_flipped, mode='same')

            if correction is not None:
                update /= correction

            # Regularization: Tikhonov-Miller
            if p['regularization'] > 0:
                reg = p['regularization']
                update = update / (1.0 + reg)

            estimate *= update
            estimate = np.maximum(estimate, 0.0)

        return estimate
