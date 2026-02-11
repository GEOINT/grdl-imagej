# -*- coding: utf-8 -*-
"""
MathOperations - Per-pixel mathematical operations.

Applies per-pixel math operations: add, subtract, multiply, divide, log,
exp, sqrt, square, abs, reciprocal, min, max, nan_to_num. Corresponds
to ImageJ's Process > Math submenu.

Attribution
-----------
ImageJ implementation: ``ij/plugin/filter/Filters.java``,
``ij/process/ImageProcessor.java`` (``add()``, ``multiply()``, ``log()``,
``exp()``, etc.) in ImageJ 1.54j.
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
from grdl.image_processing.params import Desc, Options
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI, IM.HSI, IM.SWIR, IM.MWIR, IM.LWIR],
                category=PC.MATH)
@processor_version('1.54j')
class MathOperations(ImageTransform):
    """Per-pixel mathematical operations, ported from ImageJ 1.54j.

    Applies scalar arithmetic, transcendental, or utility operations
    to every pixel in the image.

    Parameters
    ----------
    operation : str
        Operation to apply. One of: ``'add'``, ``'subtract'``,
        ``'multiply'``, ``'divide'``, ``'log'``, ``'exp'``, ``'sqrt'``,
        ``'square'``, ``'abs'``, ``'reciprocal'``, ``'min'``, ``'max'``,
        ``'nan_to_num'``.
    value : float
        Scalar operand for ``add``, ``subtract``, ``multiply``,
        ``divide``, ``min``, ``max``. Default is 0.0.
    nan_replacement : float
        Replacement value for NaN pixels in ``nan_to_num``.
        Default is 0.0.

    Notes
    -----
    Port of ``ij/process/ImageProcessor.java`` math methods from
    ImageJ 1.54j (public domain).

    - ``log`` uses ``np.log`` (natural log); zero/negative values produce
      ``-inf`` / ``nan``.
    - ``reciprocal`` computes ``1.0 / pixel``; zero pixels produce ``inf``.
    - ``divide`` by zero produces ``inf`` / ``nan``.

    Examples
    --------
    >>> from grdl_imagej import MathOperations
    >>> mo = MathOperations(operation='add', value=50.0)
    >>> brightened = mo.apply(dark_image)

    >>> mo = MathOperations(operation='log')
    >>> log_image = mo.apply(intensity_image)
    """

    __imagej_source__ = 'ij/process/ImageProcessor.java'
    __imagej_version__ = '1.54j'
    __gpu_compatible__ = True

    operation: Annotated[str, Options(
        'add', 'subtract', 'multiply', 'divide', 'log', 'exp', 'sqrt',
        'square', 'abs', 'reciprocal', 'min', 'max', 'nan_to_num'),
        Desc('Math operation to apply')] = 'add'
    value: Annotated[float, Desc('Scalar operand')] = 0.0
    nan_replacement: Annotated[float, Desc('Replacement value for NaN')] = 0.0

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply per-pixel math operation to a 2D image.

        Parameters
        ----------
        source : np.ndarray
            2D image array. Shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Transformed image (float64), same shape as input.

        Raises
        ------
        ValueError
            If source is not 2D.
        """
        if source.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {source.shape}")

        p = self._resolve_params(kwargs)
        image = source.astype(np.float64)
        op = p['operation']
        val = p['value']

        if op == 'add':
            return image + val
        elif op == 'subtract':
            return image - val
        elif op == 'multiply':
            return image * val
        elif op == 'divide':
            return image / val
        elif op == 'log':
            return np.log(image)
        elif op == 'exp':
            return np.exp(image)
        elif op == 'sqrt':
            return np.sqrt(image)
        elif op == 'square':
            return image * image
        elif op == 'abs':
            return np.abs(image)
        elif op == 'reciprocal':
            return 1.0 / image
        elif op == 'min':
            return np.minimum(image, val)
        elif op == 'max':
            return np.maximum(image, val)
        else:  # nan_to_num
            return np.where(np.isnan(image), p['nan_replacement'], image)
