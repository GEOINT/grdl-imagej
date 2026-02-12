# -*- coding: utf-8 -*-
"""
FFT Custom Filter - User-defined frequency-domain filtering.

Applies user-defined frequency-domain filter mask via FFT. Allows
arbitrary spatial frequency filtering beyond simple bandpass.

Attribution
-----------
ImageJ implementation: ``ij/plugin/FFT.java``, ``ij/plugin/filter/FFTFilter.java``
Source: https://github.com/imagej/ImageJ (public domain).
This is an independent NumPy reimplementation.

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
from grdl.image_processing.params import Desc, Options
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC

WINDOW_TYPES = ('none', 'hanning', 'hamming', 'blackman')


def _next_power_of_2(n: int) -> int:
    """Return smallest power of 2 >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


def _make_window_2d(rows: int, cols: int, window_type: str) -> np.ndarray:
    """Create a 2D apodization window."""
    if window_type == 'none':
        return np.ones((rows, cols))
    funcs = {
        'hanning': np.hanning,
        'hamming': np.hamming,
        'blackman': np.blackman,
    }
    func = funcs[window_type]
    wr = func(rows)
    wc = func(cols)
    return np.outer(wr, wc)


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI],
                category=PC.FFT)
@processor_version('1.54j')
class FFTCustomFilter(BandwiseTransformMixin, ImageTransform):
    """FFT custom filter, ported from ImageJ.

    Applies a user-defined frequency-domain mask to the image via FFT.
    The mask must be passed via the ``mask`` keyword argument.

    Parameters
    ----------
    window : str
        Apodization window to reduce edge artifacts. Default ``'hanning'``.
    pad_to_power_of_2 : bool
        Zero-pad to next power of 2 for FFT efficiency. Default True.

    Notes
    -----
    Port of ``ij/plugin/FFT.java`` from ImageJ 1.54j (public domain).

    Pipeline: apply window -> zero-pad -> FFT -> multiply by mask ->
    IFFT -> crop to original size.

    Examples
    --------
    >>> from grdl_imagej import FFTCustomFilter
    >>> filt = FFTCustomFilter(window='hanning')
    >>> result = filt.apply(image, mask=freq_domain_mask)
    """

    __imagej_source__ = 'ij/plugin/FFT.java'
    __imagej_version__ = '1.54j'
    __gpu_compatible__ = True

    window: Annotated[str, Options(*WINDOW_TYPES),
                      Desc('Apodization window')] = 'hanning'
    pad_to_power_of_2: Annotated[bool,
                                 Desc('Zero-pad to power of 2')] = True

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply custom frequency-domain filter.

        Parameters
        ----------
        source : np.ndarray
            2D image. Shape ``(rows, cols)``.
        mask : np.ndarray
            2D frequency-domain filter mask.

        Returns
        -------
        np.ndarray
            Filtered image, dtype float64, same shape as input.

        Raises
        ------
        ValueError
            If source is not 2D or mask not provided.
        """
        if source.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {source.shape}")

        p = self._resolve_params(kwargs)

        freq_mask = kwargs.get('mask', None)
        if freq_mask is None:
            raise ValueError("'mask' keyword argument is required")
        freq_mask = np.asarray(freq_mask, dtype=np.float64)

        image = source.astype(np.float64)
        rows, cols = image.shape

        # Apply window
        win = _make_window_2d(rows, cols, p['window'])
        windowed = image * win

        # Pad
        if p['pad_to_power_of_2']:
            pr = _next_power_of_2(rows)
            pc = _next_power_of_2(cols)
        else:
            pr, pc = rows, cols

        padded = np.zeros((pr, pc), dtype=np.float64)
        padded[:rows, :cols] = windowed

        # Resize mask to padded size
        if freq_mask.shape != (pr, pc):
            mask_resized = np.zeros((pr, pc), dtype=np.float64)
            mr = min(freq_mask.shape[0], pr)
            mc = min(freq_mask.shape[1], pc)
            mask_resized[:mr, :mc] = freq_mask[:mr, :mc]
        else:
            mask_resized = freq_mask

        # FFT -> multiply -> IFFT
        F = np.fft.fft2(padded)
        F_filtered = F * mask_resized
        result = np.real(np.fft.ifft2(F_filtered))

        # Crop to original size
        return result[:rows, :cols]
