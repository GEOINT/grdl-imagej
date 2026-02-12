# -*- coding: utf-8 -*-
"""
ROF Total Variation Denoising - Rudin-Osher-Fatemi TV regularization.

Minimizes total variation while maintaining fidelity to the noisy input.
Produces piecewise-smooth results with sharp edges.

Attribution
-----------
Algorithm: Rudin, Osher & Fatemi, "Nonlinear Total Variation Based Noise
Removal Algorithms", Physica D, 60(1-4), 1992.

imagej-ops implementation: variational methods in filter package.
Source: https://github.com/imagej/imagej-ops (BSD-2).
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


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI, IM.HSI, IM.SWIR, IM.MWIR, IM.LWIR],
                category=PC.NOISE)
@processor_version('0.40.0')
class ROFDenoise(BandwiseTransformMixin, ImageTransform):
    """ROF total variation denoising, ported from imagej-ops.

    Parameters
    ----------
    lambda_ : float
        Regularization weight. Higher values preserve more detail
        but retain more noise. Default 0.1.
    n_iterations : int
        Number of gradient descent iterations. Default 100.
    dt : float
        Time step for gradient descent. Must be <= 0.25 for stability.
        Default 0.125.

    Notes
    -----
    Independent reimplementation following Rudin, Osher & Fatemi (1992).

    PDE: ``u_{n+1} = u_n + dt * (div(grad(u)/|grad(u)|) + lambda*(f - u))``

    Examples
    --------
    >>> from grdl_imagej import ROFDenoise
    >>> rof = ROFDenoise(lambda_=0.1, n_iterations=100)
    >>> denoised = rof.apply(noisy_image)
    """

    __imagej_source__ = 'imagej-ops/filter/variational'
    __imagej_version__ = '0.40.0'
    __gpu_compatible__ = True

    lambda_: Annotated[float, Range(min=0.01, max=10.0),
                       Desc('Regularization weight')] = 0.1
    n_iterations: Annotated[int, Range(min=10, max=500),
                            Desc('Gradient descent iterations')] = 100
    dt: Annotated[float, Range(min=0.01, max=0.25),
                  Desc('Time step')] = 0.125

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply ROF total variation denoising.

        Parameters
        ----------
        source : np.ndarray
            2D image. Shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Denoised image, dtype float64.
        """
        if source.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {source.shape}")

        p = self._resolve_params(kwargs)
        f = source.astype(np.float64)
        u = f.copy()
        dt = p['dt']
        lam = p['lambda_']
        eps = 1e-8

        for _ in range(p['n_iterations']):
            # Compute gradients
            ux = np.diff(u, axis=1, append=u[:, -1:])
            uy = np.diff(u, axis=0, append=u[-1:, :])

            # Gradient magnitude
            grad_mag = np.sqrt(ux ** 2 + uy ** 2 + eps)

            # Normalized gradient
            nx = ux / grad_mag
            ny = uy / grad_mag

            # Divergence of normalized gradient
            # div = d(nx)/dx + d(ny)/dy
            dnx = nx - np.roll(nx, 1, axis=1)
            dnx[:, 0] = nx[:, 0]
            dny = ny - np.roll(ny, 1, axis=0)
            dny[0, :] = ny[0, :]
            div = dnx + dny

            # Update
            u = u + dt * (div + lam * (f - u))

        return u
