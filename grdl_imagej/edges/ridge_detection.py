# -*- coding: utf-8 -*-
"""
Ridge Detection - Steger's algorithm for curvilinear structure detection.

Detects ridges and valleys using Hessian eigenvalues with hysteresis
thresholding. Superior to edge detection for line-like features.

Attribution
-----------
Algorithm: Steger, "An Unbiased Detector of Curvilinear Structures",
IEEE PAMI, 20(2), 1998.

Fiji implementation:
``src/main/java/de/biomedical_imaging/ij/steger/LineDetector.java``
Source: https://github.com/fiji/Ridge_Detection (GPL-2).
This is an independent NumPy reimplementation.

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
from scipy.ndimage import gaussian_filter, label

# GRDL internal
from grdl.image_processing.base import ImageTransform, BandwiseTransformMixin
from grdl.image_processing.params import Desc, Range
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI],
                category=PC.EDGES)
@processor_version('1.4.0')
class RidgeDetection(BandwiseTransformMixin, ImageTransform):
    """Ridge detection via Steger's algorithm, ported from Fiji.

    Detects curvilinear structures using Hessian eigenvalue analysis
    with hysteresis linking and length filtering.

    Parameters
    ----------
    sigma : float
        Gaussian scale matching expected line width. Default 2.0.
    lower_threshold : float
        Hysteresis low threshold on eigenvalue magnitude. Default 3.0.
    upper_threshold : float
        Hysteresis high threshold. Default 7.0.
    min_line_length : int
        Minimum connected ridge length in pixels. Default 0.
    darkline : bool
        If True, detect dark lines on bright background. Default False.

    Notes
    -----
    Independent reimplementation of Fiji Ridge_Detection (GPL-2).
    Algorithm follows Steger (IEEE PAMI, 1998).

    Pipeline: Gaussian smoothing -> Hessian -> eigenvalue analysis ->
    ridge strength thresholding -> hysteresis linking -> length filtering.

    Examples
    --------
    >>> from grdl_imagej import RidgeDetection
    >>> rd = RidgeDetection(sigma=2.0, lower_threshold=3.0, upper_threshold=7.0)
    >>> ridges = rd.apply(image)
    """

    __imagej_source__ = 'fiji/Ridge_Detection/LineDetector.java'
    __imagej_version__ = '1.4.0'
    __gpu_compatible__ = True

    sigma: Annotated[float, Range(min=0.5, max=10.0),
                     Desc('Line width / Gaussian scale')] = 2.0
    lower_threshold: Annotated[float, Range(min=0.0, max=255.0),
                               Desc('Hysteresis low threshold')] = 3.0
    upper_threshold: Annotated[float, Range(min=0.0, max=255.0),
                               Desc('Hysteresis high threshold')] = 7.0
    min_line_length: Annotated[int, Range(min=0, max=1000),
                               Desc('Minimum line length')] = 0
    darkline: Annotated[bool, Desc('Detect dark lines on bright background')] = False

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Detect ridges/curvilinear structures.

        Parameters
        ----------
        source : np.ndarray
            2D grayscale image. Shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Binary ridge map (float64, 0.0 or 1.0).
        """
        if source.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {source.shape}")

        p = self._resolve_params(kwargs)
        image = source.astype(np.float64)
        sigma = p['sigma']

        # Smooth
        smoothed = gaussian_filter(image, sigma=sigma)

        # Compute Hessian via finite differences on smoothed image
        hxx = np.diff(smoothed, n=2, axis=1)
        hxx = np.pad(hxx, ((0, 0), (1, 1)), mode='edge')
        hyy = np.diff(smoothed, n=2, axis=0)
        hyy = np.pad(hyy, ((1, 1), (0, 0)), mode='edge')
        dx = np.gradient(smoothed, axis=1)
        hxy = np.gradient(dx, axis=0)

        # Scale normalization
        hxx *= sigma ** 2
        hxy *= sigma ** 2
        hyy *= sigma ** 2

        # Eigenvalues of 2x2 Hessian
        trace = hxx + hyy
        diff = hxx - hyy
        discriminant = np.sqrt(diff ** 2 + 4.0 * hxy ** 2)

        ev1 = 0.5 * (trace + discriminant)
        ev2 = 0.5 * (trace - discriminant)

        # Select eigenvalue with largest absolute value (max curvature)
        abs1 = np.abs(ev1)
        abs2 = np.abs(ev2)
        max_ev = np.where(abs1 >= abs2, ev1, ev2)

        # Ridge strength
        if p['darkline']:
            # Dark lines: positive second derivative (concave up)
            strength = np.maximum(max_ev, 0.0)
        else:
            # Bright lines: negative second derivative
            strength = np.maximum(-max_ev, 0.0)

        # Hysteresis thresholding
        strong = strength >= p['upper_threshold']
        weak = strength >= p['lower_threshold']

        # Label connected components in weak mask, keep those with strong pixels
        labeled, n_features = label(weak)
        result = np.zeros_like(image)

        for lbl in range(1, n_features + 1):
            component = labeled == lbl
            if np.any(strong & component):
                # Length filtering
                if p['min_line_length'] > 0:
                    if np.sum(component) >= p['min_line_length']:
                        result[component] = 1.0
                else:
                    result[component] = 1.0

        return result
