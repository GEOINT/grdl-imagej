# -*- coding: utf-8 -*-
"""
Marker-Controlled Watershed - Port of MorphoLibJ watershed segmentation.

Watershed segmentation using provided markers to control region growing,
preventing over-segmentation. Operates on gradient magnitude or the
input image directly. Each marker defines a catchment basin; pixels are
processed in order of increasing intensity/gradient.

Attribution
-----------
Algorithm: Meyer, "Morphological Segmentation", J. Visual Communication
and Image Representation, 1(1), 1990.

MorphoLibJ implementation:
``src/main/java/inra/ijpb/watershed/MarkerControlledWatershedTransform2D.java``
Source: https://github.com/ijpb/MorphoLibJ (LGPL-3).
This is an independent NumPy reimplementation following the published
algorithm.

Legland et al., "MorphoLibJ: integrated library and plugins for
mathematical morphology with ImageJ", Bioinformatics, 32(22), 2016.

Dependencies
------------
numpy
heapq

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
import heapq
from typing import Annotated, Any

# Third-party
import numpy as np
from scipy.ndimage import sobel

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.params import Desc, Options
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC

# Watershed label for boundary pixels
WATERSHED_LINE = 0

# Neighbor offsets for 4- and 8-connectivity
_NEIGHBORS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_NEIGHBORS_8 = [(-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1)]


def _compute_gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Compute gradient magnitude using Sobel operator.

    Parameters
    ----------
    image : np.ndarray
        2D grayscale image.

    Returns
    -------
    np.ndarray
        Gradient magnitude image.
    """
    gx = sobel(image, axis=1).astype(np.float64)
    gy = sobel(image, axis=0).astype(np.float64)
    return np.sqrt(gx ** 2 + gy ** 2)


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI],
                category=PC.SEGMENTATION)
@processor_version('1.6.0')
class MarkerControlledWatershed(ImageTransform):
    """Marker-controlled watershed segmentation, ported from MorphoLibJ.

    Performs watershed flooding from labeled marker regions on a
    grayscale or gradient image. Each marker grows into a catchment
    basin. Where two basins meet, a watershed line (label 0) is
    placed.

    The input ``source`` must be a 2D grayscale image. Marker labels
    are passed via the ``markers`` keyword argument to ``apply()``.

    Parameters
    ----------
    connectivity : int
        Pixel connectivity: 4 (cross) or 8 (full neighborhood).
        Default 4.
    use_gradient : bool
        If True, internally compute gradient magnitude and flood on
        that surface. If False, flood on the raw input intensity.
        Default True.

    Notes
    -----
    Independent reimplementation of MorphoLibJ
    ``MarkerControlledWatershedTransform2D.java`` (LGPL-3). The
    algorithm follows Meyer (1990).

    Priority-queue-based flooding:
      1. Initialize queue with boundary pixels of each marker region
      2. While queue is not empty:
         a. Pop pixel with lowest priority (intensity/gradient)
         b. For each unlabeled neighbor:
            - Assign same label as current pixel
            - Push neighbor onto queue with its intensity
         c. If neighbor already has a different label → watershed line

    The ``markers`` argument must be a 2D integer array of the same
    shape as ``source``, where 0 = background and positive integers
    are region labels.

    Examples
    --------
    Segment cells with known markers:

    >>> from grdl_imagej import MarkerControlledWatershed
    >>> mcw = MarkerControlledWatershed(connectivity=4, use_gradient=True)
    >>> labels = mcw.apply(grayscale_image, markers=marker_labels)
    """

    __imagej_source__ = 'MorphoLibJ/watershed/MarkerControlledWatershedTransform2D.java'
    __imagej_version__ = '1.6.0'
    __gpu_compatible__ = False

    connectivity: Annotated[int, Options(4, 8),
                            Desc('Pixel connectivity')] = 4
    use_gradient: Annotated[bool,
                            Desc('Compute gradient internally')] = True

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply marker-controlled watershed segmentation.

        Parameters
        ----------
        source : np.ndarray
            2D grayscale image. Shape ``(rows, cols)``.
        markers : np.ndarray
            2D integer label image with same shape as source.
            0 = background, positive integers = marker labels.
            Must be provided as keyword argument.

        Returns
        -------
        np.ndarray
            Label image (int32). Same shape as input. 0 = watershed
            lines, positive integers = basin labels.

        Raises
        ------
        ValueError
            If source is not 2D, markers not provided, or shape mismatch.
        """
        if source.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {source.shape}")

        p = self._resolve_params(kwargs)

        markers = kwargs.get('markers', None)
        if markers is None:
            raise ValueError("'markers' keyword argument is required")

        markers = np.asarray(markers)
        if markers.shape != source.shape:
            raise ValueError(
                f"Markers shape {markers.shape} does not match "
                f"source shape {source.shape}"
            )

        # Choose flooding surface
        image = source.astype(np.float64)
        if p['use_gradient']:
            surface = _compute_gradient_magnitude(image)
        else:
            surface = image.copy()

        rows, cols = surface.shape

        # Choose neighbor offsets
        if p['connectivity'] == 4:
            neighbors = _NEIGHBORS_4
        else:
            neighbors = _NEIGHBORS_8

        # Initialize label array and visited flags
        labels = markers.astype(np.int32).copy()
        in_queue = np.zeros((rows, cols), dtype=bool)

        # Priority queue: (priority, row, col)
        heap = []

        # Seed queue with boundary pixels of each marker
        marker_mask = labels > 0
        for r in range(rows):
            for c in range(cols):
                if not marker_mask[r, c]:
                    continue
                # Check if this marker pixel has an unlabeled neighbor
                for dy, dx in neighbors:
                    nr, nc = r + dy, c + dx
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if labels[nr, nc] == 0 and not in_queue[nr, nc]:
                            heapq.heappush(heap, (surface[nr, nc], nr, nc))
                            in_queue[nr, nc] = True

        # Flood
        while heap:
            _, r, c = heapq.heappop(heap)

            # Find label of neighboring marked pixels
            neighbor_label = 0
            is_boundary = False
            for dy, dx in neighbors:
                nr, nc = r + dy, c + dx
                if 0 <= nr < rows and 0 <= nc < cols:
                    nl = labels[nr, nc]
                    if nl > 0:
                        if neighbor_label == 0:
                            neighbor_label = nl
                        elif nl != neighbor_label:
                            is_boundary = True
                            break

            if is_boundary:
                labels[r, c] = WATERSHED_LINE
            elif neighbor_label > 0:
                labels[r, c] = neighbor_label
                # Push unlabeled neighbors
                for dy, dx in neighbors:
                    nr, nc = r + dy, c + dx
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if labels[nr, nc] == 0 and not in_queue[nr, nc]:
                            heapq.heappush(heap, (surface[nr, nc], nr, nc))
                            in_queue[nr, nc] = True

        return labels
