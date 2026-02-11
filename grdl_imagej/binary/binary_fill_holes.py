# -*- coding: utf-8 -*-
"""
Binary Fill Holes - Flood-fill interior holes in binary objects.

Fills interior holes in binary objects by flood-filling background from
image edges and inverting. Part of ImageJ's Process > Binary > Fill Holes.

Attribution
-----------
ImageJ implementation: ``ij/plugin/filter/Binary.java`` (``fill()`` method),
``ij/process/BinaryProcessor.java`` in ImageJ 1.54j.
ImageJ 1.x source is in the public domain.

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
from scipy.ndimage import binary_fill_holes as _scipy_fill_holes, generate_binary_structure

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.params import Desc, Options
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI],
                category=PC.BINARY)
@processor_version('1.54j')
class BinaryFillHoles(ImageTransform):
    """Fills interior holes in binary objects.

    Flood-fills background from all border pixels, then inverts so
    that interior holes become foreground.

    Parameters
    ----------
    connectivity : int
        Connectivity for defining holes. 4-connectivity uses a
        cross-shaped structuring element; 8-connectivity uses a
        full 3x3 square. Default is 8 (matching ImageJ).

    Notes
    -----
    Port of ``ij/plugin/filter/Binary.java`` ``fill()`` from
    ImageJ 1.54j (public domain). Delegates to
    ``scipy.ndimage.binary_fill_holes`` with the appropriate
    structuring element.

    Input is thresholded at 0.5 to ensure binary interpretation.

    Examples
    --------
    >>> from grdl_imagej import BinaryFillHoles
    >>> filler = BinaryFillHoles(connectivity=8)
    >>> filled = filler.apply(binary_mask_with_holes)
    """

    __imagej_source__ = 'ij/plugin/filter/Binary.java'
    __imagej_version__ = '1.54j'
    __gpu_compatible__ = False

    connectivity: Annotated[int, Options(4, 8),
                             Desc('Flood fill connectivity')] = 8

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Fill holes in a binary 2D image.

        Parameters
        ----------
        source : np.ndarray
            2D binary image. Shape ``(rows, cols)``. Non-zero pixels
            are treated as foreground.

        Returns
        -------
        np.ndarray
            Binary image with holes filled (float64, values 0.0 or 1.0).

        Raises
        ------
        ValueError
            If source is not 2D.
        """
        if source.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {source.shape}")

        p = self._resolve_params(kwargs)

        # Map connectivity to scipy structuring element
        # connectivity=1 → 4-connected, connectivity=2 → 8-connected
        conn = 1 if p['connectivity'] == 4 else 2
        struct = generate_binary_structure(2, conn)

        binary = source > 0.5
        filled = _scipy_fill_holes(binary, structure=struct)

        return filled.astype(np.float64)
