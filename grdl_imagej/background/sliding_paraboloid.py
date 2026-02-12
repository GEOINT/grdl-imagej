# -*- coding: utf-8 -*-
"""
Sliding Paraboloid Background - Alternative to rolling ball for background subtraction.

Background subtraction using a sliding paraboloid algorithm. Slides a
parabolic surface under/over the intensity profile in 4 directions
(horizontal, vertical, two diagonals) and combines results.

Attribution
-----------
Algorithm: Sternberg, "Biomedical Image Processing", IEEE Computer, 16(1), 1983.

ImageJ implementation:
``ij/plugin/filter/BackgroundSubtracter.java`` (``slidingParaboloidFloatBackground``)
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
from grdl.image_processing.params import Desc, Range
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC


def _slide_parabola_1d(profile: np.ndarray, radius: float) -> np.ndarray:
    """Slide a parabola under a 1D intensity profile.

    The parabola has the form: y = -x^2 / (4 * radius).
    For each center position j, the parabola value at position i is:
        profile[j] - coeff * (i - j)^2
    The background envelope is the pointwise maximum over all centers.

    Uses a vectorized O(n^2) approach for correctness.

    Parameters
    ----------
    profile : np.ndarray
        1D intensity profile.
    radius : float
        Paraboloid radius parameter.

    Returns
    -------
    np.ndarray
        Background envelope (same length as profile).
    """
    n = len(profile)
    coeff = 1.0 / (4.0 * radius)
    positions = np.arange(n, dtype=np.float64)
    envelope = np.full(n, -np.inf)

    for j in range(n):
        vals = profile[j] - coeff * (positions - j) ** 2
        envelope = np.maximum(envelope, vals)

    return envelope


def _slide_paraboloid_direction(image: np.ndarray, radius: float,
                                axis: int, diagonal: int = 0) -> np.ndarray:
    """Apply sliding paraboloid along one direction.

    Parameters
    ----------
    image : np.ndarray
        2D image.
    radius : float
        Paraboloid radius.
    axis : int
        0 = vertical (along rows), 1 = horizontal (along cols).
    diagonal : int
        0 = not diagonal, 1 = main diagonal, -1 = anti-diagonal.

    Returns
    -------
    np.ndarray
        Background estimate from this direction.
    """
    rows, cols = image.shape
    bg = np.full_like(image, -np.inf)

    if diagonal == 0:
        if axis == 1:  # Horizontal
            for r in range(rows):
                bg[r, :] = _slide_parabola_1d(image[r, :], radius)
        else:  # Vertical
            for c in range(cols):
                bg[:, c] = _slide_parabola_1d(image[:, c], radius)
    elif diagonal == 1:  # Main diagonal (top-left to bottom-right)
        # Scale radius for diagonal (sqrt(2) longer)
        diag_radius = radius * np.sqrt(2)
        for offset in range(-(rows - 1), cols):
            diag = image.diagonal(offset)
            if len(diag) < 2:
                continue
            env = _slide_parabola_1d(diag, diag_radius)
            for k in range(len(diag)):
                if offset >= 0:
                    r, c = k, k + offset
                else:
                    r, c = k - offset, k
                bg[r, c] = max(bg[r, c], env[k])
    else:  # Anti-diagonal (top-right to bottom-left)
        diag_radius = radius * np.sqrt(2)
        flipped = np.fliplr(image)
        for offset in range(-(rows - 1), cols):
            diag = flipped.diagonal(offset)
            if len(diag) < 2:
                continue
            env = _slide_parabola_1d(diag, diag_radius)
            for k in range(len(diag)):
                if offset >= 0:
                    r, c = k, (cols - 1) - (k + offset)
                else:
                    r, c = k - offset, (cols - 1) - k
                bg[r, c] = max(bg[r, c], env[k])

    return bg


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI, IM.LWIR, IM.MWIR],
                category=PC.BACKGROUND)
@processor_version('1.54j')
class SlidingParaboloid(BandwiseTransformMixin, ImageTransform):
    """Sliding paraboloid background subtraction, ported from ImageJ.

    Alternative to rolling ball. Slides parabolic surface in 4
    directions and takes the pointwise minimum as background.

    Parameters
    ----------
    radius : float
        Paraboloid radius. Larger values produce smoother backgrounds.
        Default 50.0.
    light_background : bool
        If True, estimate bright background (slide from above).
        Default False.

    Notes
    -----
    Port of ``BackgroundSubtracter.java`` (``slidingParaboloidFloatBackground``)
    from ImageJ 1.54j (public domain).

    Examples
    --------
    >>> from grdl_imagej import SlidingParaboloid
    >>> sp = SlidingParaboloid(radius=50.0)
    >>> foreground = sp.apply(image)
    """

    __imagej_source__ = 'ij/plugin/filter/BackgroundSubtracter.java'
    __imagej_version__ = '1.54j'
    __gpu_compatible__ = False

    radius: Annotated[float, Range(min=1.0, max=500.0),
                      Desc('Paraboloid radius')] = 50.0
    light_background: Annotated[bool,
                                Desc('Light background mode')] = False

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply sliding paraboloid background subtraction.

        Parameters
        ----------
        source : np.ndarray
            2D image. Shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Background-subtracted image, dtype float64.
        """
        if source.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {source.shape}")

        p = self._resolve_params(kwargs)
        image = source.astype(np.float64)
        radius = p['radius']

        if p['light_background']:
            image = -image

        # Compute background in 4 directions and take pointwise minimum
        bg_h = _slide_paraboloid_direction(image, radius, axis=1)
        bg_v = _slide_paraboloid_direction(image, radius, axis=0)
        bg_d1 = _slide_paraboloid_direction(image, radius, axis=0, diagonal=1)
        bg_d2 = _slide_paraboloid_direction(image, radius, axis=0, diagonal=-1)

        background = np.maximum(np.maximum(bg_h, bg_v), np.maximum(bg_d1, bg_d2))

        result = image - background

        if p['light_background']:
            result = -result

        return np.maximum(result, 0.0)
