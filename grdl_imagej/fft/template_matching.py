# -*- coding: utf-8 -*-
"""
Template Matching - Normalized cross-correlation for pattern detection.

Locates template pattern within larger image using normalized
cross-correlation (NCC). Returns correlation map where peaks indicate
template locations.

Attribution
-----------
Algorithm: Lewis, "Fast Normalized Cross-Correlation", Vision Interface, 1995.

imagej-ops implementation:
``src/main/java/net/imagej/ops/filter/correlate/CorrelateFFTC.java``
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
from scipy.signal import fftconvolve

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.params import Desc, Options, Range
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC

MATCHING_METHODS = ('ncc', 'zncc', 'ssd')


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI],
                category=PC.FFT)
@processor_version('0.40.0')
class TemplateMatching(ImageTransform):
    """Template matching via normalized cross-correlation, ported from imagej-ops.

    Computes a correlation map between the input image and a template.
    Peaks in the map indicate locations where the template matches.

    The template must be passed via the ``template`` keyword argument.

    Parameters
    ----------
    method : str
        Matching method. ``'ncc'`` (normalized cross-correlation),
        ``'zncc'`` (zero-mean NCC), or ``'ssd'`` (sum of squared
        differences, inverted). Default ``'zncc'``.
    threshold : float
        Minimum correlation value for a valid match. Default 0.5.

    Notes
    -----
    Independent reimplementation of imagej-ops ``CorrelateFFTC.java``
    (BSD-2). Algorithm follows Lewis (1995).

    ZNCC normalizes both image patches and template to zero-mean,
    unit-variance before correlation.

    Examples
    --------
    >>> from grdl_imagej import TemplateMatching
    >>> tm = TemplateMatching(method='zncc')
    >>> corr_map = tm.apply(image, template=small_patch)
    """

    __imagej_source__ = 'imagej-ops/filter/correlate/CorrelateFFTC.java'
    __imagej_version__ = '0.40.0'
    __gpu_compatible__ = True

    method: Annotated[str, Options(*MATCHING_METHODS),
                      Desc('Matching method')] = 'zncc'
    threshold: Annotated[float, Range(min=0.0, max=1.0),
                         Desc('Peak detection threshold')] = 0.5

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute template matching correlation map.

        Parameters
        ----------
        source : np.ndarray
            2D image. Shape ``(rows, cols)``.
        template : np.ndarray
            2D template to search for.

        Returns
        -------
        np.ndarray
            Correlation map, same shape as input, dtype float64.

        Raises
        ------
        ValueError
            If source is not 2D or template not provided.
        """
        if source.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {source.shape}")

        p = self._resolve_params(kwargs)

        template = kwargs.get('template', None)
        if template is None:
            raise ValueError("'template' keyword argument is required")
        template = np.asarray(template, dtype=np.float64)

        image = source.astype(np.float64)
        method = p['method']
        tr, tc = template.shape

        if method == 'ssd':
            # Sum of squared differences (inverted so peaks = matches)
            template_flipped = template[::-1, ::-1]
            cross = fftconvolve(image, template_flipped, mode='same')
            # SSD = sum(I^2) - 2*cross + sum(T^2)
            local_sum_sq = fftconvolve(image ** 2, np.ones_like(template), mode='same')
            template_sq_sum = np.sum(template ** 2)
            ssd = local_sum_sq - 2 * cross + template_sq_sum
            # Invert and normalize to [0, 1]
            max_ssd = np.max(ssd)
            if max_ssd > 0:
                result = 1.0 - ssd / max_ssd
            else:
                result = np.ones_like(image)
            return result

        # NCC / ZNCC
        if method == 'zncc':
            tmpl = template - template.mean()
        else:
            tmpl = template.copy()

        tmpl_flipped = tmpl[::-1, ::-1]
        tmpl_norm = np.sqrt(np.sum(tmpl ** 2))
        if tmpl_norm < 1e-10:
            return np.zeros_like(image)

        # Cross-correlation via FFT
        cross = fftconvolve(image, tmpl_flipped, mode='same')

        if method == 'zncc':
            # Local mean and std
            ones = np.ones_like(template)
            n_pixels = float(tr * tc)
            local_sum = fftconvolve(image, ones, mode='same')
            local_sum_sq = fftconvolve(image ** 2, ones, mode='same')
            local_mean = local_sum / n_pixels
            local_var = local_sum_sq / n_pixels - local_mean ** 2
            local_var = np.maximum(local_var, 0.0)
            local_std = np.sqrt(local_var) * np.sqrt(n_pixels)

            # ZNCC: subtract local mean contribution
            cross_zm = cross - local_mean * np.sum(tmpl)

            denom = local_std * tmpl_norm
            safe_denom = np.where(denom > 1e-10, denom, 1.0)
            result = np.where(denom > 1e-10, cross_zm / safe_denom, 0.0)
        else:
            # NCC
            local_sum_sq = fftconvolve(image ** 2, np.ones_like(template), mode='same')
            local_norm = np.sqrt(np.maximum(local_sum_sq, 1e-10))
            result = cross / (local_norm * tmpl_norm)

        return np.clip(result, -1.0, 1.0)
