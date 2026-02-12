# -*- coding: utf-8 -*-
"""
Non-Local Means Denoising - Port of Fiji's Non-Local Means Denoise plugin.

Denoises by averaging pixels with similar local neighborhoods across
the image, exploiting non-local self-similarity. Superior to Gaussian
smoothing for structured noise like SAR speckle.

Attribution
-----------
Algorithm: Buades, Coll & Morel, "A Non-Local Algorithm for Image
Denoising", CVPR 2005.

Fiji implementation: ``de.fzj.jungle.denoise.NonLocalMeansDenoise.java``
Source: https://github.com/fiji/Non_Local_Means_Denoise (GPL-2).
This is an independent NumPy reimplementation following the published
algorithm, not a derivative of the GPL source.

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


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI, IM.HSI, IM.SWIR, IM.MWIR, IM.LWIR],
                category=PC.NOISE)
@processor_version('1.0.0')
class NonLocalMeans(BandwiseTransformMixin, ImageTransform):
    """Non-Local Means denoising, ported from Fiji.

    For each pixel, computes a weighted average of all pixels within
    a search window, where weights depend on the similarity of their
    local patches. Preserves texture and repeated structures better
    than local smoothing methods.

    Parameters
    ----------
    sigma : float
        Estimated noise standard deviation. Used to set the default
        filtering strength ``h``. Default 15.0.
    patch_radius : int
        Half-size of the comparison patches. A patch is
        ``(2*patch_radius+1) x (2*patch_radius+1)`` pixels.
        Default 3.
    search_radius : int
        Half-size of the search window around each pixel.
        Default 11.
    h : float
        Filtering strength (decay parameter). Larger values produce
        stronger smoothing. If 0.0, defaults to ``sigma``. Default 0.0.

    Notes
    -----
    Independent reimplementation of Fiji's ``NonLocalMeansDenoise.java``
    (GPL-2). The algorithm follows Buades, Coll & Morel (CVPR 2005).

    For each pixel p:
      1. Extract patch P_p of size (2*patch_radius+1)^2
      2. For each pixel q in search window:
         - Compute squared patch distance d² = ||P_p - P_q||²
         - Weight w(p,q) = exp(-d² / h²)
      3. Output(p) = Σ w(p,q) * I(q) / Σ w(p,q)

    Examples
    --------
    Denoise SAR speckle:

    >>> from grdl_imagej import NonLocalMeans
    >>> nlm = NonLocalMeans(sigma=25.0, patch_radius=3, search_radius=11)
    >>> denoised = nlm.apply(noisy_sar)

    Strong denoising with large search window:

    >>> nlm = NonLocalMeans(sigma=40.0, search_radius=21, h=50.0)
    >>> result = nlm.apply(very_noisy_image)
    """

    __imagej_source__ = 'fiji/Non_Local_Means_Denoise/NonLocalMeansDenoise.java'
    __imagej_version__ = '1.0.0'
    __gpu_compatible__ = True

    sigma: Annotated[float, Range(min=1.0, max=100.0),
                     Desc('Noise standard deviation estimate')] = 15.0
    patch_radius: Annotated[int, Range(min=1, max=7),
                            Desc('Half-size of comparison patches')] = 3
    search_radius: Annotated[int, Range(min=5, max=31),
                             Desc('Half-size of search window')] = 11
    h: Annotated[float, Range(min=0.0, max=200.0),
                 Desc('Filtering strength (0 = use sigma)')] = 0.0

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply Non-Local Means denoising to a 2D image.

        Parameters
        ----------
        source : np.ndarray
            2D image array. Shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Denoised image, same shape as input, dtype float64.

        Raises
        ------
        ValueError
            If source is not 2D.
        """
        if source.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {source.shape}")

        p = self._resolve_params(kwargs)

        image = source.astype(np.float64)
        rows, cols = image.shape

        pr = p['patch_radius']
        sr = p['search_radius']
        h_val = p['h'] if p['h'] > 0.0 else p['sigma']
        h2 = h_val * h_val
        patch_area = (2 * pr + 1) ** 2

        # Pad image for patch and search window extraction
        pad = sr + pr
        padded = np.pad(image, pad, mode='reflect')

        result = np.zeros_like(image)

        # Precompute: use vectorized offset-based approach
        # For each offset (dy, dx) in the search window, compute the
        # patch distance for all pixels simultaneously
        weight_sum = np.zeros_like(image)
        weighted_val = np.zeros_like(image)

        for dy in range(-sr, sr + 1):
            for dx in range(-sr, sr + 1):
                # Compute squared patch distance for all pixels at once
                # using a running sum over the patch window
                diff_sq = np.zeros_like(image)
                for py in range(-pr, pr + 1):
                    for px in range(-pr, pr + 1):
                        # Reference pixel's patch element
                        ry = pad + py
                        rx = pad + px
                        ref_slice = padded[ry:ry + rows, rx:rx + cols]

                        # Neighbor pixel's patch element
                        ny = pad + dy + py
                        nx = pad + dx + px
                        nbr_slice = padded[ny:ny + rows, nx:nx + cols]

                        diff_sq += (ref_slice - nbr_slice) ** 2

                # Normalize by patch area
                dist_sq = diff_sq / patch_area

                # Compute weights
                w = np.exp(-dist_sq / h2)

                # Accumulate
                nbr_y = pad + dy
                nbr_x = pad + dx
                neighbor_vals = padded[nbr_y:nbr_y + rows, nbr_x:nbr_x + cols]

                weight_sum += w
                weighted_val += w * neighbor_vals

        # Normalize
        result = weighted_val / np.maximum(weight_sum, 1e-10)

        return result
