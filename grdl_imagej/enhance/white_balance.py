# -*- coding: utf-8 -*-
"""
White Balance / Color Normalization - Illumination color cast correction.

Normalizes color balance using gray-world, white-patch (Retinex), or
percentile methods. Corrects illumination color cast for consistent color
representation in multi-temporal analysis.

Attribution
-----------
Algorithm: Buchsbaum, "A Spatial Processor Model for Object Colour
Perception", J. Franklin Institute, 310(1), 1980.
Related: ImageJ ``ij/plugin/filter/GrayWorld.java``.
ImageJ 1.x source is in the public domain.

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


@processor_tags(modalities=[IM.EO, IM.MSI, IM.PAN],
                category=PC.ENHANCE)
@processor_version('1.0.0')
class WhiteBalance(ImageTransform):
    """White balance / color normalization.

    Corrects color cast using one of three methods:

    - **gray_world**: Scale each channel so its mean equals the global
      gray value (average of all channel means).
    - **white_patch**: Scale each channel so its maximum equals 1.0
      (Retinex / max-RGB).
    - **percentile**: Scale each channel so the Nth percentile equals 1.0.

    Parameters
    ----------
    method : str
        Normalization method. Default is ``'gray_world'``.
    percentile : float
        Percentile threshold for the ``'percentile'`` method.
        Default is 1.0 (99th percentile maps to white).

    Notes
    -----
    Reference: Buchsbaum, J. Franklin Institute, 310(1), 1980.

    Input should be a 3-channel (RGB) image with values in [0, 1] or
    [0, 255]. Output is clipped to [0, 1].

    Examples
    --------
    >>> from grdl_imagej import WhiteBalance
    >>> wb = WhiteBalance(method='gray_world')
    >>> corrected = wb.apply(color_cast_image)
    """

    __imagej_source__ = 'ij/plugin/filter/GrayWorld.java'
    __imagej_version__ = '1.0.0'
    __gpu_compatible__ = True

    method: Annotated[str, Options('gray_world', 'white_patch', 'percentile'),
                       Desc('Normalization method')] = 'gray_world'
    percentile: Annotated[float, Range(min=0.1, max=5.0),
                           Desc('Percentile for percentile method')] = 1.0

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply white balance correction to a 3-channel image.

        Parameters
        ----------
        source : np.ndarray
            3-channel image array. Shape ``(rows, cols, 3)``.

        Returns
        -------
        np.ndarray
            Color-corrected image (float64), clipped to [0, 1].

        Raises
        ------
        ValueError
            If source is not 3D with 3 channels.
        """
        if source.ndim != 3 or source.shape[2] != 3:
            raise ValueError(
                f"Expected 3-channel image (rows, cols, 3), got shape {source.shape}"
            )

        p = self._resolve_params(kwargs)
        image = source.astype(np.float64)
        method = p['method']
        eps = np.finfo(np.float64).eps

        if method == 'gray_world':
            # Scale each channel so its mean equals the global mean
            channel_means = np.array([image[..., c].mean() for c in range(3)])
            global_mean = channel_means.mean()
            result = np.empty_like(image)
            for c in range(3):
                scale = global_mean / (channel_means[c] + eps)
                result[..., c] = image[..., c] * scale

        elif method == 'white_patch':
            # Scale each channel so its max equals 1.0
            result = np.empty_like(image)
            for c in range(3):
                cmax = image[..., c].max()
                result[..., c] = image[..., c] / (cmax + eps)

        else:  # percentile
            result = np.empty_like(image)
            pct = 100.0 - p['percentile']
            for c in range(3):
                pval = np.percentile(image[..., c], pct)
                result[..., c] = image[..., c] / (pval + eps)

        return np.clip(result, 0.0, 1.0)
