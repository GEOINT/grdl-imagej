# -*- coding: utf-8 -*-
"""
Hough Transform - Line and circle detection in edge images.

Detects lines and circles by mapping edge points to parameter space
and finding accumulator peaks corresponding to geometric primitives.

Attribution
-----------
Algorithm: Duda & Hart, "Use of the Hough Transformation to Detect
Lines and Curves in Pictures", 1972. Yuen et al., 1990.

Fiji implementation: ``src/main/java/Hough_Circle.java`` (circles).
Source: https://github.com/fiji/Hough_Circle (GPL-2).
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
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.params import Desc, Options, Range
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI],
                category=PC.FIND_MAXIMA)
@processor_version('1.0.0')
class HoughTransform(ImageTransform):
    """Hough Transform for line and circle detection, ported from Fiji.

    Maps edge pixels to parameter space and returns the accumulator
    image. Peaks correspond to detected geometric primitives.

    Parameters
    ----------
    mode : str
        Detection mode: ``'lines'`` or ``'circles'``. Default ``'lines'``.
    threshold : float
        Accumulator threshold as fraction of max. Peaks above this
        are considered detections. Default 0.5.
    min_radius : int
        Minimum circle radius (circles mode only). Default 10.
    max_radius : int
        Maximum circle radius (circles mode only). Default 100.
    rho_resolution : float
        Rho (distance) resolution for line detection. Default 1.0.
    theta_resolution : float
        Theta resolution in degrees for line detection. Default 1.0.

    Notes
    -----
    Independent reimplementation. Lines mode returns a (rho, theta)
    accumulator. Circles mode returns an (x, y) accumulator summed
    over all tested radii.

    Examples
    --------
    >>> from grdl_imagej import HoughTransform
    >>> ht = HoughTransform(mode='lines', threshold=0.5)
    >>> accumulator = ht.apply(edge_image)
    """

    __imagej_source__ = 'fiji/Hough_Circle/Hough_Circle.java'
    __imagej_version__ = '1.0.0'
    __gpu_compatible__ = False

    mode: Annotated[str, Options('lines', 'circles'),
                    Desc('Detection mode')] = 'lines'
    threshold: Annotated[float, Range(min=0.0, max=1.0),
                         Desc('Accumulator threshold (fraction of max)')] = 0.5
    min_radius: Annotated[int, Range(min=1, max=500),
                          Desc('Min circle radius')] = 10
    max_radius: Annotated[int, Range(min=5, max=1000),
                          Desc('Max circle radius')] = 100
    rho_resolution: Annotated[float, Range(min=0.1, max=5.0),
                              Desc('Rho resolution for lines')] = 1.0
    theta_resolution: Annotated[float, Range(min=0.1, max=10.0),
                                Desc('Theta resolution in degrees')] = 1.0

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply Hough Transform.

        Parameters
        ----------
        source : np.ndarray
            2D edge image. Shape ``(rows, cols)``. Non-zero pixels
            are treated as edge points.

        Returns
        -------
        np.ndarray
            For lines: accumulator of shape ``(n_rho, n_theta)``.
            For circles: accumulator of shape ``(rows, cols)``.

        Raises
        ------
        ValueError
            If source is not 2D.
        """
        if source.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {source.shape}")

        p = self._resolve_params(kwargs)
        image = source.astype(np.float64)

        if p['mode'] == 'lines':
            return self._hough_lines(image, p)
        else:
            return self._hough_circles(image, p)

    def _hough_lines(self, image: np.ndarray, p: dict) -> np.ndarray:
        """Hough transform for lines."""
        rows, cols = image.shape
        diag = int(np.ceil(np.sqrt(rows ** 2 + cols ** 2)))

        rho_res = p['rho_resolution']
        theta_res = p['theta_resolution']

        n_rho = int(2 * diag / rho_res) + 1
        n_theta = int(180.0 / theta_res)

        accumulator = np.zeros((n_rho, n_theta), dtype=np.float64)

        # Precompute theta values
        thetas = np.deg2rad(np.arange(0, 180, theta_res))
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)

        # Find edge points
        edge_y, edge_x = np.where(image > 0)

        for i in range(len(edge_y)):
            y, x = edge_y[i], edge_x[i]
            rhos = x * cos_t + y * sin_t
            rho_idx = ((rhos + diag) / rho_res).astype(np.intp)
            valid = (rho_idx >= 0) & (rho_idx < n_rho)
            theta_idx = np.arange(n_theta)
            accumulator[rho_idx[valid], theta_idx[valid]] += 1

        return accumulator

    def _hough_circles(self, image: np.ndarray, p: dict) -> np.ndarray:
        """Hough transform for circles."""
        rows, cols = image.shape
        accumulator = np.zeros((rows, cols), dtype=np.float64)

        edge_y, edge_x = np.where(image > 0)

        for r in range(p['min_radius'], p['max_radius'] + 1):
            # Precompute circle offsets
            n_points = max(int(2 * np.pi * r), 36)
            angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
            dx = np.round(r * np.cos(angles)).astype(np.intp)
            dy = np.round(r * np.sin(angles)).astype(np.intp)

            for i in range(len(edge_y)):
                cy = edge_y[i] - dy
                cx = edge_x[i] - dx
                valid = (cy >= 0) & (cy < rows) & (cx >= 0) & (cx < cols)
                accumulator[cy[valid], cx[valid]] += 1

        return accumulator
