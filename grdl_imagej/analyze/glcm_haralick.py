# -*- coding: utf-8 -*-
"""
GLCM / Haralick Texture Features - Gray-Level Co-occurrence Matrix analysis.

Computes Gray-Level Co-occurrence Matrices and derives Haralick texture
descriptors: energy (ASM), entropy, contrast, correlation, homogeneity,
dissimilarity, and variance. Essential for land cover classification in
remote sensing and texture-based image analysis.

Attribution
-----------
Algorithm: Haralick, Shanmugam & Dinstein, "Textural Features for Image
Classification", IEEE Trans. SMC, 3(6), 1973.

imagej-ops implementation:
``src/main/java/net/imagej/ops/image/cooccurrenceMatrix/CooccurrenceMatrix2D.java``
and ``src/main/java/net/imagej/ops/features/haralick/`` (14 feature classes).
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
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.params import Desc, Options, Range
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC


HARALICK_FEATURES = (
    'energy', 'entropy', 'contrast', 'correlation',
    'homogeneity', 'dissimilarity', 'variance',
)

# Angle offsets: (dy, dx) for each direction
_ANGLE_OFFSETS = {
    '0': (0, 1),
    '45': (-1, 1),
    '90': (-1, 0),
    '135': (-1, -1),
}


def _build_glcm(image: np.ndarray, distance: int, dy: int, dx: int,
                 n_levels: int, symmetric: bool) -> np.ndarray:
    """Build a single GLCM for a given offset direction.

    Parameters
    ----------
    image : np.ndarray
        Quantized 2D image with values in [0, n_levels).
    distance : int
        Pixel offset distance.
    dy, dx : int
        Direction unit vector components.
    n_levels : int
        Number of gray levels.
    symmetric : bool
        If True, count both (i,j) and (j,i) pairs.

    Returns
    -------
    np.ndarray
        Normalized GLCM of shape (n_levels, n_levels).
    """
    rows, cols = image.shape
    oy = dy * distance
    ox = dx * distance

    # Determine valid pixel ranges
    r_start = max(0, -oy)
    r_end = min(rows, rows - oy)
    c_start = max(0, -ox)
    c_end = min(cols, cols - ox)

    glcm = np.zeros((n_levels, n_levels), dtype=np.float64)

    ref = image[r_start:r_end, c_start:c_end]
    neighbor = image[r_start + oy:r_end + oy, c_start + ox:c_end + ox]

    # Vectorized accumulation
    indices = ref.astype(np.intp) * n_levels + neighbor.astype(np.intp)
    np.add.at(glcm.ravel(), indices.ravel(), 1)

    if symmetric:
        glcm = glcm + glcm.T

    total = glcm.sum()
    if total > 0:
        glcm /= total

    return glcm


def _compute_features(glcm: np.ndarray) -> dict[str, float]:
    """Compute Haralick features from a normalized GLCM.

    Parameters
    ----------
    glcm : np.ndarray
        Normalized GLCM of shape (n_levels, n_levels).

    Returns
    -------
    dict
        Feature name → value mapping.
    """
    n = glcm.shape[0]
    i_idx = np.arange(n)
    j_idx = np.arange(n)
    ii, jj = np.meshgrid(i_idx, j_idx, indexing='ij')

    # Marginals
    px = glcm.sum(axis=1)  # row marginal
    py = glcm.sum(axis=0)  # col marginal

    mu_x = np.sum(i_idx * px)
    mu_y = np.sum(j_idx * py)
    sigma_x = np.sqrt(np.sum((i_idx - mu_x) ** 2 * px))
    sigma_y = np.sqrt(np.sum((j_idx - mu_y) ** 2 * py))

    features = {}

    # Energy (Angular Second Moment)
    features['energy'] = float(np.sum(glcm ** 2))

    # Entropy
    mask = glcm > 0
    features['entropy'] = float(-np.sum(glcm[mask] * np.log2(glcm[mask])))

    # Contrast
    features['contrast'] = float(np.sum((ii - jj) ** 2 * glcm))

    # Correlation
    if sigma_x > 0 and sigma_y > 0:
        features['correlation'] = float(
            np.sum((ii - mu_x) * (jj - mu_y) * glcm) / (sigma_x * sigma_y)
        )
    else:
        features['correlation'] = 0.0

    # Homogeneity (Inverse Difference Moment)
    features['homogeneity'] = float(np.sum(glcm / (1.0 + (ii - jj) ** 2)))

    # Dissimilarity
    features['dissimilarity'] = float(np.sum(np.abs(ii - jj) * glcm))

    # Variance (sum of variances)
    features['variance'] = float(
        np.sum((ii - mu_x) ** 2 * glcm) + np.sum((jj - mu_y) ** 2 * glcm)
    )

    return features


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI, IM.HSI],
                category=PC.ANALYZE)
@processor_version('0.40.0')
class GLCMHaralick(ImageTransform):
    """GLCM / Haralick texture features, ported from imagej-ops.

    Computes Gray-Level Co-occurrence Matrices and derives Haralick
    texture descriptors. The output is a multi-band image where each
    band corresponds to one requested texture feature.

    Parameters
    ----------
    distance : int
        Pixel offset for co-occurrence pairs. Default 1.
    angle : str
        Direction for co-occurrence pairs. ``'0'`` (horizontal),
        ``'45'``, ``'90'`` (vertical), ``'135'``, or ``'all'``
        (average over all four directions). Default ``'all'``.
    n_gray_levels : int
        Number of quantization levels for the GLCM. Default 64.
    symmetric : bool
        Whether to compute symmetric GLCM (count both (i,j) and
        (j,i) pairs). Default True.
    features : str
        Comma-separated list of features to compute, or ``'all'``.
        Available: energy, entropy, contrast, correlation,
        homogeneity, dissimilarity, variance. Default ``'all'``.

    Notes
    -----
    Independent reimplementation of imagej-ops
    ``CooccurrenceMatrix2D.java`` and the Haralick feature classes
    (BSD-2). The algorithm follows Haralick et al. (1973).

    The output shape is ``(rows, cols, n_features)`` when multiple
    features are requested, or ``(rows, cols)`` for a single feature.
    Features are computed in a sliding window centered on each pixel.

    For efficiency, this implementation computes a single global GLCM
    (or per-angle GLCMs) and returns the feature values as a uniform
    map. For spatially varying texture, use a tiled approach externally.

    Examples
    --------
    Compute all Haralick features:

    >>> from grdl_imagej import GLCMHaralick
    >>> glcm = GLCMHaralick(distance=1, angle='all', n_gray_levels=64)
    >>> features = glcm.apply(satellite_pan)

    Compute only contrast and entropy:

    >>> glcm = GLCMHaralick(features='contrast,entropy')
    >>> result = glcm.apply(image)
    """

    __imagej_source__ = 'imagej-ops/CooccurrenceMatrix2D.java'
    __imagej_version__ = '0.40.0'
    __gpu_compatible__ = False

    distance: Annotated[int, Range(min=1, max=10), Desc('Pixel offset')] = 1
    angle: Annotated[str, Options('0', '45', '90', '135', 'all'),
                     Desc('Co-occurrence direction')] = 'all'
    n_gray_levels: Annotated[int, Options(32, 64, 128, 256),
                             Desc('Quantization levels')] = 64
    symmetric: Annotated[bool, Desc('Symmetric GLCM')] = True
    features: Annotated[str, Desc('Comma-separated features or "all"')] = 'all'

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute GLCM-based Haralick texture features.

        Parameters
        ----------
        source : np.ndarray
            2D grayscale image. Shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Feature map. Shape ``(rows, cols, n_features)`` for
            multiple features, or ``(rows, cols)`` for a single
            feature. Each band contains the feature value computed
            from the global GLCM, broadcast to every pixel.

        Raises
        ------
        ValueError
            If source is not 2D or an unknown feature is requested.
        """
        if source.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {source.shape}")

        p = self._resolve_params(kwargs)

        # Parse requested features
        feat_str = p['features'].strip()
        if feat_str == 'all':
            feat_names = list(HARALICK_FEATURES)
        else:
            feat_names = [f.strip() for f in feat_str.split(',')]
            for f in feat_names:
                if f not in HARALICK_FEATURES:
                    raise ValueError(
                        f"Unknown feature '{f}'. Available: {HARALICK_FEATURES}"
                    )

        image = source.astype(np.float64)
        n_levels = p['n_gray_levels']

        # Quantize to [0, n_levels)
        imin = image.min()
        imax = image.max()
        if imax > imin:
            quantized = ((image - imin) / (imax - imin) * (n_levels - 1)).astype(np.intp)
        else:
            quantized = np.zeros_like(image, dtype=np.intp)
        quantized = np.clip(quantized, 0, n_levels - 1)

        # Determine angles to compute
        angle = p['angle']
        if angle == 'all':
            angles = list(_ANGLE_OFFSETS.keys())
        else:
            angles = [angle]

        # Build GLCMs and compute features
        all_features = []
        for ang in angles:
            dy, dx = _ANGLE_OFFSETS[ang]
            glcm = _build_glcm(
                quantized, p['distance'], dy, dx,
                n_levels, p['symmetric'],
            )
            all_features.append(_compute_features(glcm))

        # Average features across angles
        avg_features = {}
        for fname in feat_names:
            avg_features[fname] = np.mean([f[fname] for f in all_features])

        # Build output array
        rows, cols = source.shape
        if len(feat_names) == 1:
            return np.full((rows, cols), avg_features[feat_names[0]], dtype=np.float64)
        else:
            result = np.empty((rows, cols, len(feat_names)), dtype=np.float64)
            for i, fname in enumerate(feat_names):
                result[:, :, i] = avg_features[fname]
            return result
