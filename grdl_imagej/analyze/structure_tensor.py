# -*- coding: utf-8 -*-
"""
Structure Tensor / Orientation Analysis - Port of Fiji OrientationJ.

Computes local structure tensor (gradient outer product smoothed by
Gaussian) and extracts orientation, coherence, and energy. Characterizes
local image anisotropy and dominant orientation. Critical for SAR
polarimetry analysis and texture characterization.

Attribution
-----------
Algorithm: Jahne, "Digital Image Processing", Springer, Chapter 13.
Rezakhaniha et al., Biomechanics and Modeling in Mechanobiology,
11(3-4), 2012.

Fiji implementation: OrientationJ — ``orientation/StructureTensor.java``
Source: https://github.com/fiji/OrientationJ (GPL-2).
This is an independent NumPy reimplementation following the published
algorithm, not a derivative of the GPL source.

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
from scipy.ndimage import gaussian_filter, sobel

# GRDL internal
from grdl.image_processing.base import ImageTransform, BandwiseTransformMixin
from grdl.image_processing.params import Desc, Options, Range
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC

OUTPUT_MODES = ('orientation', 'coherence', 'energy', 'all')


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI],
                category=PC.ANALYZE)
@processor_version('2.0.0')
class StructureTensor(BandwiseTransformMixin, ImageTransform):
    """Structure tensor / orientation analysis, ported from OrientationJ.

    Computes the structure tensor at each pixel from image gradients,
    smoothed with a Gaussian window, then extracts orientation,
    coherence, and energy from the tensor eigenvalues.

    Parameters
    ----------
    sigma : float
        Gaussian window sigma for tensor smoothing. Controls the
        scale of the analysis. Default 2.0.
    output : str
        Output type: ``'orientation'`` (angle in radians, -pi/2 to
        pi/2), ``'coherence'`` (0 to 1, anisotropy measure),
        ``'energy'`` (gradient magnitude), or ``'all'`` (3-band
        stack). Default ``'all'``.

    Notes
    -----
    Independent reimplementation of Fiji OrientationJ
    ``StructureTensor.java`` (GPL-2). The algorithm follows Jahne
    (Springer, Chapter 13).

    The structure tensor is:

        J = [[<Ix*Ix>  <Ix*Iy>],
             [<Ix*Iy>  <Iy*Iy>]]

    where <.> denotes Gaussian smoothing. From the 2x2 eigenvalue
    decomposition:

    - orientation = 0.5 * atan2(2 * Jxy, Jxx - Jyy)
    - coherence = (lambda1 - lambda2) / (lambda1 + lambda2)
    - energy = lambda1 + lambda2 = Jxx + Jyy (trace)

    Examples
    --------
    Compute all orientation features:

    >>> from grdl_imagej import StructureTensor
    >>> st = StructureTensor(sigma=2.0, output='all')
    >>> result = st.apply(image)  # shape (H, W, 3)

    Compute only coherence:

    >>> st = StructureTensor(sigma=3.0, output='coherence')
    >>> coherence = st.apply(image)  # shape (H, W)
    """

    __imagej_source__ = 'fiji/OrientationJ/StructureTensor.java'
    __imagej_version__ = '2.0.0'
    __gpu_compatible__ = True

    sigma: Annotated[float, Range(min=0.5, max=20.0),
                     Desc('Gaussian window for tensor smoothing')] = 2.0
    output: Annotated[str, Options(*OUTPUT_MODES),
                      Desc('Output type')] = 'all'

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute structure tensor features.

        Parameters
        ----------
        source : np.ndarray
            2D grayscale image. Shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Depending on ``output`` parameter:
            - ``'orientation'``: ``(rows, cols)`` angles in radians
            - ``'coherence'``: ``(rows, cols)`` values in [0, 1]
            - ``'energy'``: ``(rows, cols)`` non-negative values
            - ``'all'``: ``(rows, cols, 3)`` stack [orientation, coherence, energy]

        Raises
        ------
        ValueError
            If source is not 2D.
        """
        if source.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {source.shape}")

        p = self._resolve_params(kwargs)

        image = source.astype(np.float64)
        sigma = p['sigma']

        # Compute image gradients using Sobel operator
        ix = sobel(image, axis=1).astype(np.float64)  # dI/dx
        iy = sobel(image, axis=0).astype(np.float64)  # dI/dy

        # Compute products of gradients
        ixx = ix * ix
        ixy = ix * iy
        iyy = iy * iy

        # Smooth with Gaussian (structure tensor averaging)
        jxx = gaussian_filter(ixx, sigma=sigma)
        jxy = gaussian_filter(ixy, sigma=sigma)
        jyy = gaussian_filter(iyy, sigma=sigma)

        # Analytic eigenvalue decomposition of 2x2 symmetric matrix
        # [[jxx, jxy], [jxy, jyy]]
        # trace = jxx + jyy
        # det = jxx * jyy - jxy * jxy
        # discriminant = sqrt((jxx - jyy)^2 + 4*jxy^2)
        trace = jxx + jyy
        diff = jxx - jyy
        discriminant = np.sqrt(diff ** 2 + 4.0 * jxy ** 2)

        lambda1 = 0.5 * (trace + discriminant)
        lambda2 = 0.5 * (trace - discriminant)

        # Orientation: dominant direction (angle of eigenvector for lambda1)
        orientation = 0.5 * np.arctan2(2.0 * jxy, diff)

        # Coherence: anisotropy measure in [0, 1]
        denom = lambda1 + lambda2
        safe_denom = np.where(denom > 0, denom, 1.0)
        coherence = np.where(denom > 0, (lambda1 - lambda2) / safe_denom, 0.0)

        # Energy: total gradient magnitude (trace of tensor)
        energy = trace

        out_mode = p['output']
        if out_mode == 'orientation':
            return orientation
        elif out_mode == 'coherence':
            return coherence
        elif out_mode == 'energy':
            return energy
        else:  # 'all'
            return np.stack([orientation, coherence, energy], axis=-1)
