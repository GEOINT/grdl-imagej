# -*- coding: utf-8 -*-
"""
Morphological Reconstruction - Geodesic reconstruction by dilation or erosion.

Fundamental building block for h-maxima, h-minima, regional maxima/minima,
fill holes, and many advanced morphological operations. Uses an efficient
hybrid algorithm combining raster/anti-raster scans with queue-based
propagation.

Attribution
-----------
Algorithm: Vincent, "Morphological Grayscale Reconstruction in Image
Analysis: Applications and Efficient Algorithms", IEEE Trans. Image
Processing, 2(2), 1993.

MorphoLibJ implementation:
``src/main/java/inra/ijpb/morphology/geodrec/GeodesicReconstructionByDilation.java``
and ``GeodesicReconstructionByErosion.java``
Source: https://github.com/ijpb/MorphoLibJ (LGPL-3).
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
from collections import deque
from typing import Annotated, Any

# Third-party
import numpy as np

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.params import Desc, Options
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC

_NEIGHBORS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_NEIGHBORS_8 = [(-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1)]


def reconstruct_by_dilation(marker: np.ndarray, mask: np.ndarray,
                            connectivity: int = 4) -> np.ndarray:
    """Geodesic reconstruction by dilation.

    Parameters
    ----------
    marker : np.ndarray
        Marker image (must be <= mask pointwise).
    mask : np.ndarray
        Mask image (upper bound).
    connectivity : int
        4 or 8 connectivity.

    Returns
    -------
    np.ndarray
        Reconstructed image.
    """
    result = np.minimum(marker.astype(np.float64), mask.astype(np.float64))
    mask_f = mask.astype(np.float64)
    rows, cols = result.shape
    neighbors = _NEIGHBORS_4 if connectivity == 4 else _NEIGHBORS_8

    # Forward raster scan
    for r in range(rows):
        for c in range(cols):
            val = result[r, c]
            for dy, dx in neighbors:
                nr, nc = r + dy, c + dx
                if 0 <= nr < rows and 0 <= nc < cols:
                    if nr < r or (nr == r and nc < c):
                        val = max(val, result[nr, nc])
            result[r, c] = min(val, mask_f[r, c])

    # Backward raster scan + queue initialization
    queue = deque()
    for r in range(rows - 1, -1, -1):
        for c in range(cols - 1, -1, -1):
            val = result[r, c]
            for dy, dx in neighbors:
                nr, nc = r + dy, c + dx
                if 0 <= nr < rows and 0 <= nc < cols:
                    if nr > r or (nr == r and nc > c):
                        val = max(val, result[nr, nc])
            result[r, c] = min(val, mask_f[r, c])

            # Queue pixels that could propagate further
            for dy, dx in neighbors:
                nr, nc = r + dy, c + dx
                if 0 <= nr < rows and 0 <= nc < cols:
                    if result[nr, nc] < result[r, c] and result[nr, nc] < mask_f[nr, nc]:
                        queue.append((r, c))
                        break

    # Queue-based propagation
    while queue:
        r, c = queue.popleft()
        for dy, dx in neighbors:
            nr, nc = r + dy, c + dx
            if 0 <= nr < rows and 0 <= nc < cols:
                new_val = min(result[r, c], mask_f[nr, nc])
                if new_val > result[nr, nc]:
                    result[nr, nc] = new_val
                    queue.append((nr, nc))

    return result


def reconstruct_by_erosion(marker: np.ndarray, mask: np.ndarray,
                           connectivity: int = 4) -> np.ndarray:
    """Geodesic reconstruction by erosion.

    Parameters
    ----------
    marker : np.ndarray
        Marker image (must be >= mask pointwise).
    mask : np.ndarray
        Mask image (lower bound).
    connectivity : int
        4 or 8 connectivity.

    Returns
    -------
    np.ndarray
        Reconstructed image.
    """
    result = np.maximum(marker.astype(np.float64), mask.astype(np.float64))
    mask_f = mask.astype(np.float64)
    rows, cols = result.shape
    neighbors = _NEIGHBORS_4 if connectivity == 4 else _NEIGHBORS_8

    # Forward raster scan
    for r in range(rows):
        for c in range(cols):
            val = result[r, c]
            for dy, dx in neighbors:
                nr, nc = r + dy, c + dx
                if 0 <= nr < rows and 0 <= nc < cols:
                    if nr < r or (nr == r and nc < c):
                        val = min(val, result[nr, nc])
            result[r, c] = max(val, mask_f[r, c])

    # Backward raster scan + queue initialization
    queue = deque()
    for r in range(rows - 1, -1, -1):
        for c in range(cols - 1, -1, -1):
            val = result[r, c]
            for dy, dx in neighbors:
                nr, nc = r + dy, c + dx
                if 0 <= nr < rows and 0 <= nc < cols:
                    if nr > r or (nr == r and nc > c):
                        val = min(val, result[nr, nc])
            result[r, c] = max(val, mask_f[r, c])

            for dy, dx in neighbors:
                nr, nc = r + dy, c + dx
                if 0 <= nr < rows and 0 <= nc < cols:
                    if result[nr, nc] > result[r, c] and result[nr, nc] > mask_f[nr, nc]:
                        queue.append((r, c))
                        break

    # Queue-based propagation
    while queue:
        r, c = queue.popleft()
        for dy, dx in neighbors:
            nr, nc = r + dy, c + dx
            if 0 <= nr < rows and 0 <= nc < cols:
                new_val = max(result[r, c], mask_f[nr, nc])
                if new_val < result[nr, nc]:
                    result[nr, nc] = new_val
                    queue.append((nr, nc))

    return result


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI],
                category=PC.BINARY)
@processor_version('1.6.0')
class MorphologicalReconstruction(ImageTransform):
    """Geodesic morphological reconstruction, ported from MorphoLibJ.

    Reconstructs a marker image under the constraint of a mask image
    using iterative geodesic dilation or erosion.

    The ``source`` image is used as the mask. The marker must be
    passed via the ``marker`` keyword argument.

    Parameters
    ----------
    type : str
        Reconstruction type: ``'by_dilation'`` (marker <= mask,
        reconstructs upward) or ``'by_erosion'`` (marker >= mask,
        reconstructs downward). Default ``'by_dilation'``.
    connectivity : int
        4 or 8 connectivity. Default 4.

    Notes
    -----
    Independent reimplementation of MorphoLibJ
    ``GeodesicReconstructionByDilation.java`` (LGPL-3). Algorithm
    follows Vincent (IEEE Trans. Image Processing, 1993).

    Uses hybrid raster-scan + FIFO queue algorithm for efficiency.

    Examples
    --------
    >>> from grdl_imagej import MorphologicalReconstruction
    >>> mr = MorphologicalReconstruction(type='by_dilation')
    >>> result = mr.apply(mask_image, marker=marker_image)
    """

    __imagej_source__ = 'MorphoLibJ/geodrec/GeodesicReconstructionByDilation.java'
    __imagej_version__ = '1.6.0'
    __gpu_compatible__ = False

    type: Annotated[str, Options('by_dilation', 'by_erosion'),
                    Desc('Reconstruction type')] = 'by_dilation'
    connectivity: Annotated[int, Options(4, 8),
                            Desc('Pixel connectivity')] = 4

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply morphological reconstruction.

        Parameters
        ----------
        source : np.ndarray
            2D mask image. Shape ``(rows, cols)``.
        marker : np.ndarray
            2D marker image with same shape. Must be provided as
            keyword argument.

        Returns
        -------
        np.ndarray
            Reconstructed image, dtype float64.

        Raises
        ------
        ValueError
            If source is not 2D, marker missing, or shape mismatch.
        """
        if source.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {source.shape}")

        p = self._resolve_params(kwargs)

        marker = kwargs.get('marker', None)
        if marker is None:
            raise ValueError("'marker' keyword argument is required")

        marker = np.asarray(marker)
        if marker.shape != source.shape:
            raise ValueError(
                f"Marker shape {marker.shape} does not match "
                f"source shape {source.shape}"
            )

        if p['type'] == 'by_dilation':
            return reconstruct_by_dilation(marker, source, p['connectivity'])
        else:
            return reconstruct_by_erosion(marker, source, p['connectivity'])
