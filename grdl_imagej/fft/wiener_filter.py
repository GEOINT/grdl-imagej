# -*- coding: utf-8 -*-
"""
Wiener Filter Deconvolution - Frequency-domain regularized deconvolution.

Single-step frequency-domain deconvolution dividing image spectrum by
PSF spectrum, regularized by noise-to-signal ratio to avoid noise
amplification.

Attribution
-----------
Algorithm: Wiener (1949); Gonzalez & Woods, "Digital Image Processing",
Chapter 5.

imagej-ops implementation:
``src/main/java/net/imagej/ops/filter/ifft/`` and deconvolution utilities.
Source: https://github.com/imagej/imagej-ops (BSD-2).
This is an independent NumPy reimplementation following the published algorithm.

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
from grdl.image_processing.base import ImageTransform, BandwiseTransformMixin
from grdl.image_processing.params import Desc, Range
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI, IM.LWIR, IM.MWIR],
                category=PC.FFT)
@processor_version('0.40.0')
class WienerFilter(BandwiseTransformMixin, ImageTransform):
    """Wiener filter deconvolution, ported from imagej-ops.

    Single-step frequency-domain deconvolution using the Wiener filter
    formulation with SNR-based regularization.

    The PSF must be passed via the ``psf`` keyword argument.

    Parameters
    ----------
    snr : float
        Estimated signal-to-noise ratio. Higher values produce sharper
        restoration; lower values produce smoother results. Default 10.0.
    clip_negative : bool
        If True, clip negative values to zero in the output. Default True.

    Notes
    -----
    Independent reimplementation of imagej-ops deconvolution utilities
    (BSD-2). Follows Wiener (1949).

    Wiener filter formula:
      F_restored = conj(H) * F_image / (|H|^2 + 1/SNR)

    where H = FFT(PSF), F_image = FFT(image).

    Examples
    --------
    >>> from grdl_imagej import WienerFilter
    >>> wf = WienerFilter(snr=15.0)
    >>> restored = wf.apply(blurred_image, psf=point_spread_function)
    """

    __imagej_source__ = 'imagej-ops/filter/ifft/'
    __imagej_version__ = '0.40.0'
    __gpu_compatible__ = True

    snr: Annotated[float, Range(min=0.001, max=100.0),
                   Desc('Signal-to-noise ratio estimate')] = 10.0
    clip_negative: Annotated[bool, Desc('Clip negative values to zero')] = True

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply Wiener filter deconvolution.

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
            If source is not 2D or psf not provided.
        """
        if source.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {source.shape}")

        p = self._resolve_params(kwargs)

        psf = kwargs.get('psf', None)
        if psf is None:
            raise ValueError("'psf' keyword argument is required")
        psf = np.asarray(psf, dtype=np.float64)

        image = source.astype(np.float64)
        rows, cols = image.shape

        # Zero-pad PSF to image size
        psf_padded = np.zeros_like(image)
        pr, pc = psf.shape
        psf_padded[:pr, :pc] = psf
        # Center the PSF
        psf_padded = np.roll(psf_padded, -(pr // 2), axis=0)
        psf_padded = np.roll(psf_padded, -(pc // 2), axis=1)

        # Normalize
        psf_sum = psf_padded.sum()
        if psf_sum > 0:
            psf_padded /= psf_sum

        # FFT
        F_image = np.fft.fft2(image)
        H = np.fft.fft2(psf_padded)

        # Wiener filter
        H_conj = np.conj(H)
        H_sq = np.abs(H) ** 2
        nsr = 1.0 / p['snr']

        F_restored = H_conj * F_image / (H_sq + nsr)

        result = np.real(np.fft.ifft2(F_restored))

        if p['clip_negative']:
            result = np.maximum(result, 0.0)

        return result
