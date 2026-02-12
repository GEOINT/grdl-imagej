# -*- coding: utf-8 -*-
"""
Kill Borders - Remove connected components touching the image border.

Eliminates partial objects at image edges before measurement. Uses
geodesic reconstruction by dilation with border pixels as markers.

Attribution
-----------
Algorithm: Soille, "Morphological Image Analysis", Springer, 2nd ed., 2003.

MorphoLibJ implementation: uses ``GeodesicReconstructionByDilation`` internally.
Source: https://github.com/ijpb/MorphoLibJ (LGPL-3).
This is an independent NumPy reimplementation.

Dependencies
------------
numpy
Depends on MorphologicalReconstruction (T2-06).

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
from grdl.image_processing.params import Desc, Options
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC

from grdl_imagej.binary.morphological_reconstruction import reconstruct_by_dilation


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI],
                category=PC.BINARY)
@processor_version('1.6.0')
class KillBorders(ImageTransform):
    """Remove border-touching components, ported from MorphoLibJ.

    Creates a marker from border pixels, performs geodesic
    reconstruction by dilation, then subtracts to remove all
    components connected to the border.

    Parameters
    ----------
    connectivity : int
        4 or 8 connectivity. Default 8.

    Notes
    -----
    Independent reimplementation of MorphoLibJ border-killing
    functionality (LGPL-3). Uses ``reconstruct_by_dilation`` from
    the MorphologicalReconstruction processor.

    Examples
    --------
    >>> from grdl_imagej import KillBorders
    >>> kb = KillBorders(connectivity=8)
    >>> cleaned = kb.apply(binary_or_labeled_image)
    """

    __imagej_source__ = 'MorphoLibJ/GeodesicReconstructionByDilation.java'
    __imagej_version__ = '1.6.0'
    __gpu_compatible__ = False

    connectivity: Annotated[int, Options(4, 8),
                            Desc('Pixel connectivity')] = 8

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Remove border-touching components.

        Parameters
        ----------
        source : np.ndarray
            2D image (binary or grayscale). Shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Image with border components removed, dtype float64.
        """
        if source.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {source.shape}")

        p = self._resolve_params(kwargs)
        image = source.astype(np.float64)
        rows, cols = image.shape

        # Create border marker: copy border pixels, zero interior
        marker = np.zeros_like(image)
        marker[0, :] = image[0, :]       # top row
        marker[-1, :] = image[-1, :]     # bottom row
        marker[:, 0] = image[:, 0]       # left column
        marker[:, -1] = image[:, -1]     # right column

        # Reconstruct from border markers
        reconstructed = reconstruct_by_dilation(marker, image, p['connectivity'])

        # Subtract: remove border-connected components
        return image - reconstructed
