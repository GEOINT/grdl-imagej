# -*- coding: utf-8 -*-
"""
TypeConverter - Image data type conversion with scaling.

Converts images between data types (uint8, uint16, float32, float64)
with configurable scaling, clamping, and normalization. Corresponds
to ImageJ's Image > Type conversions.

Attribution
-----------
ImageJ implementation: ``ij/process/ImageConverter.java``,
``ij/process/ImageProcessor.java`` in ImageJ 1.54j.
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

_DTYPE_MAP = {
    'uint8': np.uint8,
    'uint16': np.uint16,
    'float32': np.float32,
    'float64': np.float64,
}

_INT_INFO = {
    np.uint8: (0, 255),
    np.uint16: (0, 65535),
}


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI, IM.HSI, IM.SWIR, IM.MWIR, IM.LWIR],
                category=PC.MATH)
@processor_version('1.54j')
class TypeConverter(ImageTransform):
    """Image data type converter, ported from ImageJ 1.54j.

    Converts between uint8, uint16, float32, and float64 with
    optional scaling and normalization.

    Parameters
    ----------
    target_type : str
        Target data type. One of ``'uint8'``, ``'uint16'``,
        ``'float32'``, ``'float64'``. Default is ``'uint8'``.
    scale : bool
        If True, linearly scale values to fit the target range when
        converting between integer types. Default is True.
    normalize : bool
        If True and target is a float type, normalize to [0, 1].
        Default is False.

    Notes
    -----
    Port of ``ij/process/ImageConverter.java`` from ImageJ 1.54j
    (public domain).

    Scaling behavior:
    - Integer → integer (with scale): linear mapping from source range
      to target range.
    - Float → integer (with scale): maps [min, max] to target range.
    - Integer → float (with normalize): maps source range to [0, 1].
    - Without scaling: values are clamped and cast directly.

    Examples
    --------
    >>> from grdl_imagej import TypeConverter
    >>> tc = TypeConverter(target_type='uint8', scale=True)
    >>> byte_image = tc.apply(float_image)
    """

    __imagej_source__ = 'ij/process/ImageConverter.java'
    __imagej_version__ = '1.54j'
    __gpu_compatible__ = True

    target_type: Annotated[str, Options('uint8', 'uint16', 'float32', 'float64'),
                            Desc('Target data type')] = 'uint8'
    scale: Annotated[bool, Desc('Scale values when converting')] = True
    normalize: Annotated[bool, Desc('Normalize to [0,1] for float output')] = False

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Convert image data type.

        Parameters
        ----------
        source : np.ndarray
            2D image array. Shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Converted image with the requested dtype.

        Raises
        ------
        ValueError
            If source is not 2D.
        """
        if source.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {source.shape}")

        p = self._resolve_params(kwargs)
        target_dtype = _DTYPE_MAP[p['target_type']]
        image = source.astype(np.float64)

        # Normalize to [0, 1] if requested and target is float
        if p['normalize'] and np.issubdtype(target_dtype, np.floating):
            lo, hi = image.min(), image.max()
            eps = np.finfo(np.float64).eps
            if hi - lo > eps:
                image = (image - lo) / (hi - lo)
            else:
                image = np.zeros_like(image)
            return image.astype(target_dtype)

        # Scale to target integer range
        if p['scale'] and target_dtype in _INT_INFO:
            lo, hi = image.min(), image.max()
            t_lo, t_hi = _INT_INFO[target_dtype]
            eps = np.finfo(np.float64).eps
            if hi - lo > eps:
                image = (image - lo) / (hi - lo) * (t_hi - t_lo) + t_lo
            else:
                image = np.full_like(image, (t_hi + t_lo) / 2.0)
            image = np.clip(image, t_lo, t_hi)
            return image.astype(target_dtype)

        # Direct cast with clamping for integer targets
        if target_dtype in _INT_INFO:
            t_lo, t_hi = _INT_INFO[target_dtype]
            image = np.clip(image, t_lo, t_hi)

        return image.astype(target_dtype)
