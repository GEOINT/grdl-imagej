# -*- coding: utf-8 -*-
"""
Phase Correlation - Translational shift estimation via FFT.

Estimates the translational shift between two images using the
normalized cross-power spectrum in the frequency domain. Sub-pixel
accurate, very fast via FFT, and robust to intensity differences.

Particularly useful for:
- SAR/EO image co-registration
- Multi-temporal image alignment
- Sub-pixel shift estimation for pan-sharpening
- Detecting translational offsets in image mosaics

Attribution
-----------
Algorithm: Kuglin & Hines, "The Phase Correlation Image Alignment Method",
IEEE Conf. Cybernetics & Society, 1975.
Sub-pixel: Guizar-Sicairos et al., "Efficient subpixel image registration
algorithms", Optics Letters, 33(2), 2008.
Related: ``imagej-ops`` — ``CorrelateFFTC.java`` (BSD-2).

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
from grdl.image_processing.params import Desc, Options, Range
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC


def _apply_window(image: np.ndarray, window_type: str) -> np.ndarray:
    """Apply apodization window to reduce spectral leakage."""
    if window_type == 'none':
        return image
    rows, cols = image.shape
    if window_type == 'hann':
        wr = np.hanning(rows)
        wc = np.hanning(cols)
    else:  # blackman
        wr = np.blackman(rows)
        wc = np.blackman(cols)
    return image * np.outer(wr, wc)


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI, IM.SWIR, IM.MWIR, IM.LWIR],
                category=PC.FFT)
@processor_version('1.0.0')
class PhaseCorrelation(ImageTransform):
    """Phase correlation for translational shift estimation.

    Computes the normalized cross-power spectrum between the input
    image and a reference image to determine the translational offset.

    The reference image is passed via the ``reference`` keyword argument
    to ``apply()``. If no reference is provided, the correlation surface
    (identity) is returned.

    The detected shift is stored as the ``last_shift`` attribute after
    ``apply()`` is called: a ``(dy, dx)`` tuple.

    Parameters
    ----------
    upsample_factor : int
        Sub-pixel accuracy factor. 1 = integer-pixel accuracy.
        Higher values give finer sub-pixel estimates. Default is 10.
    normalize : bool
        Use normalized cross-power spectrum (phase-only). Default is True.
    window : str
        Apodization window to reduce spectral leakage. Default is 'hann'.

    Notes
    -----
    References: Kuglin & Hines (1975), Guizar-Sicairos et al. (2008).

    The output is the phase correlation surface. The peak location
    indicates the translational shift. Use ``last_shift`` attribute
    to retrieve the detected ``(dy, dx)`` offset.

    Examples
    --------
    >>> from grdl_imagej import PhaseCorrelation
    >>> pc = PhaseCorrelation(upsample_factor=10)
    >>> corr_surface = pc.apply(shifted_image, reference=reference_image)
    >>> dy, dx = pc.last_shift
    """

    __imagej_source__ = 'imagej-ops/filter/correlate/CorrelateFFTC.java'
    __imagej_version__ = '1.0.0'
    __gpu_compatible__ = True

    upsample_factor: Annotated[int, Range(min=1, max=100),
                                Desc('Sub-pixel accuracy factor')] = 10
    normalize: Annotated[bool, Desc('Use normalized cross-power spectrum')] = True
    window: Annotated[str, Options('hann', 'blackman', 'none'),
                       Desc('Apodization window')] = 'hann'

    last_shift: tuple = (0.0, 0.0)

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute phase correlation between source and reference.

        Parameters
        ----------
        source : np.ndarray
            2D image array. Shape ``(rows, cols)``.
        reference : np.ndarray
            2D reference image (same shape as source). Passed as
            keyword argument.

        Returns
        -------
        np.ndarray
            Phase correlation surface (float64). Peak indicates shift.

        Raises
        ------
        ValueError
            If source is not 2D or reference shape does not match.
        """
        if source.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {source.shape}")

        p = self._resolve_params(kwargs)
        reference = kwargs.get('reference', None)

        image = source.astype(np.float64)

        if reference is None:
            self.last_shift = (0.0, 0.0)
            return np.zeros_like(image)

        if reference.shape != image.shape:
            raise ValueError(
                f"Reference shape {reference.shape} must match source {image.shape}"
            )

        ref = reference.astype(np.float64)

        # Apply windowing
        win_type = p['window']
        img_w = _apply_window(image, win_type)
        ref_w = _apply_window(ref, win_type)

        # FFT
        f1 = np.fft.fft2(img_w)
        f2 = np.fft.fft2(ref_w)

        # Cross-power spectrum
        cross = f1 * np.conj(f2)

        if p['normalize']:
            eps = np.finfo(np.float64).eps
            cross = cross / (np.abs(cross) + eps)

        # Inverse FFT to get correlation surface
        corr = np.fft.ifft2(cross).real

        # Find peak (integer pixel)
        rows, cols = corr.shape
        peak_idx = np.unravel_index(np.argmax(corr), corr.shape)
        dy, dx = peak_idx

        # Wrap shifts to [-N/2, N/2)
        if dy > rows // 2:
            dy -= rows
        if dx > cols // 2:
            dx -= cols

        # Sub-pixel refinement via parabolic interpolation
        if p['upsample_factor'] > 1:
            py, px = peak_idx
            if 0 < py < rows - 1:
                denom = 2.0 * corr[py, px] - corr[py - 1, px] - corr[py + 1, px]
                if abs(denom) > 1e-10:
                    sub_y = (corr[py - 1, px] - corr[py + 1, px]) / (2.0 * denom)
                    dy = float(dy) + sub_y
            if 0 < px < cols - 1:
                denom = 2.0 * corr[py, px] - corr[py, px - 1] - corr[py, px + 1]
                if abs(denom) > 1e-10:
                    sub_x = (corr[py, px - 1] - corr[py, px + 1]) / (2.0 * denom)
                    dx = float(dx) + sub_x

        self.last_shift = (float(dy), float(dx))

        return corr
