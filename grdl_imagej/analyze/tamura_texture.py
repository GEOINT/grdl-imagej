# -*- coding: utf-8 -*-
"""
Tamura Texture Features - Perceptually motivated texture descriptors.

Computes coarseness, contrast, and directionality features designed
to correspond to human visual texture perception.

Attribution
-----------
Algorithm: Tamura, Mori & Yamawaki, "Textural Features Corresponding to
Visual Perception", IEEE Trans. SMC, 8(6), 1978.

imagej-ops implementation:
``src/main/java/net/imagej/ops/features/tamura/``
(DefaultCoarseness, DefaultContrast, DefaultDirectionality)
Source: https://github.com/imagej/imagej-ops (BSD-2).
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
from scipy.ndimage import uniform_filter

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.params import Desc, Range
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC


def _compute_coarseness(image: np.ndarray, n_scales: int) -> float:
    """Compute Tamura coarseness.

    For each pixel, compute average intensity in windows of increasing
    size (2^k), then compute differences. The optimal scale is the one
    with maximum difference. Coarseness is the average optimal scale.
    """
    rows, cols = image.shape
    best_k = np.zeros_like(image, dtype=np.float64)
    best_diff = np.zeros_like(image)

    for k in range(1, n_scales + 1):
        size = 2 ** k
        # Moving average
        avg = uniform_filter(image, size=size, mode='nearest')

        # Horizontal difference
        half = size // 2
        diff_h = np.zeros_like(image)
        diff_h[:, half:cols - half] = np.abs(
            avg[:, size:] - avg[:, :cols - size]
        ) if size < cols else diff_h[:, half:cols - half]

        # Vertical difference
        diff_v = np.zeros_like(image)
        diff_v[half:rows - half, :] = np.abs(
            avg[size:, :] - avg[:rows - size, :]
        ) if size < rows else diff_v[half:rows - half, :]

        # Max of h and v
        diff = np.maximum(diff_h, diff_v)
        mask = diff > best_diff
        best_diff[mask] = diff[mask]
        best_k[mask] = 2.0 ** k

    return float(np.mean(best_k))


def _compute_contrast(image: np.ndarray) -> float:
    """Compute Tamura contrast.

    Contrast = sigma / (kurtosis^0.25) where sigma is std dev and
    kurtosis is the 4th moment / sigma^4.
    """
    mu = np.mean(image)
    sigma = np.std(image)
    if sigma < 1e-10:
        return 0.0

    # Kurtosis (excess kurtosis + 3)
    m4 = np.mean((image - mu) ** 4)
    kurtosis = m4 / (sigma ** 4)

    if kurtosis < 1e-10:
        return 0.0

    return float(sigma / (kurtosis ** 0.25))


def _compute_directionality(image: np.ndarray, n_bins: int) -> float:
    """Compute Tamura directionality.

    Based on gradient orientation histogram. Higher values indicate
    more directional texture.
    """
    # Compute gradients
    gy = np.gradient(image, axis=0)
    gx = np.gradient(image, axis=1)

    # Gradient magnitude and orientation
    mag = np.sqrt(gx ** 2 + gy ** 2)
    theta = np.arctan2(gy, gx)  # [-pi, pi]

    # Threshold: only use pixels with significant gradient
    mag_threshold = mag.mean()
    mask = mag > mag_threshold

    if not np.any(mask):
        return 0.0

    # Build orientation histogram (quantize to [0, pi))
    angles = theta[mask] % np.pi
    hist, _ = np.histogram(angles, bins=n_bins, range=(0, np.pi))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total < 1:
        return 0.0
    hist /= total

    # Directionality: 1 - entropy_normalized
    # More peaked histogram = more directional
    mask_h = hist > 0
    entropy = -np.sum(hist[mask_h] * np.log2(hist[mask_h]))
    max_entropy = np.log2(n_bins)
    if max_entropy < 1e-10:
        return 0.0

    return float(1.0 - entropy / max_entropy)


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI, IM.HSI],
                category=PC.ANALYZE)
@processor_version('0.40.0')
class TamuraTexture(ImageTransform):
    """Tamura texture features, ported from imagej-ops.

    Computes three perceptually motivated texture descriptors:
    coarseness, contrast, and directionality.

    Parameters
    ----------
    n_scales : int
        Number of scales for coarseness computation. Default 5.
    histogram_bins : int
        Number of bins for directionality histogram. Default 64.

    Notes
    -----
    Independent reimplementation of imagej-ops Tamura feature classes
    (BSD-2). Algorithm follows Tamura, Mori & Yamawaki (IEEE Trans.
    SMC, 1978).

    Output is a 3-band image: [coarseness, contrast, directionality].

    Examples
    --------
    >>> from grdl_imagej import TamuraTexture
    >>> tt = TamuraTexture(n_scales=5)
    >>> features = tt.apply(image)  # shape (H, W, 3)
    """

    __imagej_source__ = 'imagej-ops/features/tamura/'
    __imagej_version__ = '0.40.0'
    __gpu_compatible__ = False

    n_scales: Annotated[int, Range(min=3, max=8),
                        Desc('Number of scales for coarseness')] = 5
    histogram_bins: Annotated[int, Range(min=16, max=128),
                              Desc('Bins for directionality histogram')] = 64

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute Tamura texture features.

        Parameters
        ----------
        source : np.ndarray
            2D grayscale image. Shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Feature map of shape ``(rows, cols, 3)`` where bands are
            [coarseness, contrast, directionality]. Values are
            broadcast uniformly (global features).
        """
        if source.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {source.shape}")

        p = self._resolve_params(kwargs)
        image = source.astype(np.float64)

        coarseness = _compute_coarseness(image, p['n_scales'])
        contrast = _compute_contrast(image)
        directionality = _compute_directionality(image, p['histogram_bins'])

        rows, cols = image.shape
        result = np.empty((rows, cols, 3), dtype=np.float64)
        result[:, :, 0] = coarseness
        result[:, :, 1] = contrast
        result[:, :, 2] = directionality

        return result
