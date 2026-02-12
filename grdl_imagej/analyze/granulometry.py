# -*- coding: utf-8 -*-
"""
Granulometry - Size distribution via morphological openings.

Applies morphological openings with increasing structuring element size
and measures residual volume to produce a size distribution curve.

Attribution
-----------
Algorithm: Matheron, "Random Sets and Integral Geometry", Wiley, 1975.

MorphoLibJ implementation:
``src/main/java/inra/ijpb/measure/Granulometry.java``
Source: https://github.com/ijpb/MorphoLibJ (LGPL-3).
This is an independent NumPy reimplementation.

Dependencies
------------
numpy
Reuses existing MorphologicalFilter.

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

from grdl_imagej.binary.morphology import MorphologicalFilter


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI],
                category=PC.ANALYZE)
@processor_version('1.6.0')
class Granulometry(ImageTransform):
    """Granulometry (size distribution analysis), ported from MorphoLibJ.

    Applies morphological openings (or closings) with increasing SE
    radius and computes the volume curve derivative. The output is a
    1D size distribution encoded as a 2D image (row 0 = distribution,
    rest = zeros).

    Parameters
    ----------
    max_radius : int
        Maximum structuring element radius. Default 20.
    step : int
        Radius increment. Default 1.
    se_shape : str
        Structuring element shape. Default ``'disk'``.
    type : str
        Operation type: ``'opening'`` or ``'closing'``. Default ``'opening'``.

    Notes
    -----
    Independent reimplementation of MorphoLibJ ``Granulometry.java``
    (LGPL-3). Follows Matheron (1975).

    The output is a 2D array of shape ``(1, max_radius // step + 1)``
    containing the granulometric size distribution (derivative of
    the volume curve). Each column corresponds to a radius value.

    Examples
    --------
    >>> from grdl_imagej import Granulometry
    >>> g = Granulometry(max_radius=30, step=2)
    >>> size_dist = g.apply(image)
    """

    __imagej_source__ = 'MorphoLibJ/measure/Granulometry.java'
    __imagej_version__ = '1.6.0'
    __gpu_compatible__ = False

    max_radius: Annotated[int, Range(min=1, max=100),
                          Desc('Maximum SE radius')] = 20
    step: Annotated[int, Range(min=1, max=10),
                    Desc('Radius increment')] = 1
    se_shape: Annotated[str, Options('disk', 'square'),
                        Desc('Structuring element shape')] = 'disk'
    type: Annotated[str, Options('opening', 'closing'),
                    Desc('Operation type')] = 'opening'

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute granulometric size distribution.

        Parameters
        ----------
        source : np.ndarray
            2D grayscale image. Shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Size distribution as 2D array of shape
            ``(1, n_radii)`` where n_radii is the number of
            tested SE sizes. dtype float64.
        """
        if source.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {source.shape}")

        p = self._resolve_params(kwargs)
        image = source.astype(np.float64)

        radii = list(range(1, p['max_radius'] + 1, p['step']))
        op = 'open' if p['type'] == 'opening' else 'close'
        shape = p['se_shape']

        # Volume curve: sum of pixel values after opening at each radius
        volumes = [float(np.sum(image))]  # radius=0 = original
        for r in radii:
            mf = MorphologicalFilter(operation=op, radius=r, kernel_shape=shape)
            opened = mf.apply(image)
            volumes.append(float(np.sum(opened)))

        # Size distribution = negative derivative of volume curve
        volumes = np.array(volumes)
        distribution = -np.diff(volumes)
        # Normalize
        total = distribution.sum()
        if total > 0:
            distribution /= total

        return distribution.reshape(1, -1)
