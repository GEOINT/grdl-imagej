# -*- coding: utf-8 -*-
"""
Frangi Vesselness / Tubeness Filter - Multi-scale vessel enhancement.

Derives vesselness measure from Hessian matrix eigenvalues. Enhances
tubular structures (vessels, ridges, linear features like roads/rivers)
while suppressing blobs and background. Multi-scale capable.

Attribution
-----------
Algorithm: Frangi et al., "Multiscale Vessel Enhancement Filtering",
MICCAI, Springer LNCS 1496, 1998.

imagej-ops implementation:
``src/main/java/net/imagej/ops/filter/tubeness/DefaultTubeness.java``
and ``src/main/java/net/imagej/ops/filter/hessian/``
Source: https://github.com/imagej/imagej-ops (BSD-2).
This is an independent NumPy reimplementation following the published
algorithm.

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
from grdl.image_processing.base import ImageTransform, BandwiseTransformMixin
from grdl.image_processing.params import Desc, Range
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC


def _hessian_2d(image: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Hessian matrix components using Gaussian 2nd derivatives.

    Parameters
    ----------
    image : np.ndarray
        2D image array.
    sigma : float
        Gaussian scale for derivative computation.

    Returns
    -------
    hxx, hxy, hyy : np.ndarray
        Second-order partial derivatives.
    """
    # Smooth with Gaussian first, then compute numerical 2nd derivatives.
    # This is equivalent to convolving with 2nd derivative of Gaussian.
    smoothed = gaussian_filter(image, sigma=sigma)

    # Second derivatives via finite differences on smoothed image
    # Scale-normalized: multiply by sigma^2
    hxx = np.diff(smoothed, n=2, axis=1)
    # Pad to original size
    hxx = np.pad(hxx, ((0, 0), (1, 1)), mode='edge')

    hyy = np.diff(smoothed, n=2, axis=0)
    hyy = np.pad(hyy, ((1, 1), (0, 0)), mode='edge')

    # Mixed partial: d²I/dxdy
    dx = np.gradient(smoothed, axis=1)
    hxy = np.gradient(dx, axis=0)

    # Scale normalization
    hxx *= sigma ** 2
    hxy *= sigma ** 2
    hyy *= sigma ** 2

    return hxx, hxy, hyy


def _eigenvalues_2d(hxx: np.ndarray, hxy: np.ndarray,
                    hyy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute eigenvalues of 2x2 Hessian at each pixel.

    Returns eigenvalues sorted by absolute value: |lambda1| <= |lambda2|.
    """
    trace = hxx + hyy
    diff = hxx - hyy
    discriminant = np.sqrt(diff ** 2 + 4.0 * hxy ** 2)

    ev1 = 0.5 * (trace + discriminant)
    ev2 = 0.5 * (trace - discriminant)

    # Sort by absolute value: |lambda1| <= |lambda2|
    abs1 = np.abs(ev1)
    abs2 = np.abs(ev2)

    lambda1 = np.where(abs1 <= abs2, ev1, ev2)
    lambda2 = np.where(abs1 <= abs2, ev2, ev1)

    return lambda1, lambda2


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI],
                category=PC.FILTERS)
@processor_version('0.40.0')
class FrangiVesselness(BandwiseTransformMixin, ImageTransform):
    """Frangi vesselness / tubeness filter, ported from imagej-ops.

    Enhances tubular structures using multi-scale Hessian analysis.
    At each scale (sigma), the Hessian eigenvalues classify local
    geometry: vessel-like structures produce one small and one large
    eigenvalue.

    Parameters
    ----------
    scale_min : float
        Minimum Gaussian scale for multi-scale analysis. Default 1.0.
    scale_max : float
        Maximum Gaussian scale. Default 4.0.
    scale_step : float
        Step between scales. Default 1.0.
    alpha : float
        Plate-vs-line sensitivity parameter. Controls suppression of
        blob-like structures. Default 0.5.
    beta : float
        Blob sensitivity parameter. Controls suppression of
        background (low structure). Default 0.5.
    black_ridges : bool
        If True, detect dark ridges on bright background (negative
        eigenvalues). If False, detect bright ridges on dark
        background. Default True.

    Notes
    -----
    Independent reimplementation of imagej-ops ``DefaultTubeness.java``
    (BSD-2). Algorithm follows Frangi et al. (MICCAI 1998).

    For 2D images, the vesselness at each scale is:

        V(s) = exp(-R_B² / (2*beta²)) * (1 - exp(-S² / (2*c²)))

    where R_B = lambda1 / lambda2 (blobness ratio) and
    S = sqrt(lambda1² + lambda2²) (Frobenius norm, second-order
    structureness).

    The final output is the maximum vesselness across all scales.

    Examples
    --------
    Enhance roads/rivers in satellite imagery:

    >>> from grdl_imagej import FrangiVesselness
    >>> frangi = FrangiVesselness(scale_min=1.0, scale_max=8.0)
    >>> vessels = frangi.apply(pan_image)
    """

    __imagej_source__ = 'imagej-ops/filter/tubeness/DefaultTubeness.java'
    __imagej_version__ = '0.40.0'
    __gpu_compatible__ = True

    scale_min: Annotated[float, Range(min=0.5, max=50.0),
                         Desc('Minimum Gaussian scale')] = 1.0
    scale_max: Annotated[float, Range(min=0.5, max=50.0),
                         Desc('Maximum Gaussian scale')] = 4.0
    scale_step: Annotated[float, Range(min=0.1, max=10.0),
                          Desc('Scale step size')] = 1.0
    alpha: Annotated[float, Range(min=0.1, max=1.0),
                     Desc('Plate-vs-line sensitivity')] = 0.5
    beta: Annotated[float, Range(min=0.1, max=1.0),
                    Desc('Blob sensitivity')] = 0.5
    black_ridges: Annotated[bool, Desc('Detect dark ridges on bright background')] = True

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply Frangi vesselness filter.

        Parameters
        ----------
        source : np.ndarray
            2D grayscale image. Shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Vesselness response, same shape as input, dtype float64.
            Values in [0, 1].

        Raises
        ------
        ValueError
            If source is not 2D or scale_min > scale_max.
        """
        if source.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {source.shape}")

        p = self._resolve_params(kwargs)

        image = source.astype(np.float64)
        alpha = p['alpha']
        beta_val = p['beta']
        black = p['black_ridges']

        # Build scale list
        s_min = p['scale_min']
        s_max = p['scale_max']
        s_step = p['scale_step']

        if s_min > s_max:
            raise ValueError(
                f"scale_min ({s_min}) must be <= scale_max ({s_max})"
            )

        sigmas = []
        s = s_min
        while s <= s_max + 1e-10:
            sigmas.append(s)
            s += s_step

        if not sigmas:
            sigmas = [s_min]

        # Precompute constants
        two_alpha_sq = 2.0 * alpha * alpha
        two_beta_sq = 2.0 * beta_val * beta_val

        # Auto-compute c (structureness threshold) as half the max Frobenius norm
        # We'll compute it during the first pass
        max_vesselness = np.zeros_like(image)

        for sigma in sigmas:
            hxx, hxy, hyy = _hessian_2d(image, sigma)
            lambda1, lambda2 = _eigenvalues_2d(hxx, hxy, hyy)

            # Dark ridges (valleys): second derivative perpendicular is
            # positive (lambda2 > 0). Bright ridges: lambda2 < 0.
            if black:
                vessel_mask = lambda2 > 0
            else:
                vessel_mask = lambda2 < 0

            # Avoid division by zero
            abs_lambda2 = np.abs(lambda2)
            safe_lambda2 = np.where(abs_lambda2 > 0, abs_lambda2, 1.0)

            # Blobness ratio R_B = |lambda1| / |lambda2|
            r_b = np.abs(lambda1) / safe_lambda2

            # Second-order structureness S = sqrt(lambda1^2 + lambda2^2)
            s_sq = lambda1 ** 2 + lambda2 ** 2

            # Auto-set c from max S across image
            c_sq = np.max(s_sq) * 0.5
            if c_sq < 1e-10:
                c_sq = 1.0

            # Frangi vesselness
            vesselness = (
                np.exp(-r_b ** 2 / two_alpha_sq)
                * (1.0 - np.exp(-s_sq / (2.0 * c_sq)))
            )

            # Zero out non-vessel regions
            vesselness = np.where(vessel_mask, vesselness, 0.0)

            # Take maximum across scales
            max_vesselness = np.maximum(max_vesselness, vesselness)

        return max_vesselness
