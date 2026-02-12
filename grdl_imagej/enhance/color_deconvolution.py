# -*- coding: utf-8 -*-
"""
Color Deconvolution - Spectral unmixing via stain matrix inversion.

Separates multi-stain/multi-spectral imagery into individual channel
contributions using a known mixing matrix. Based on Beer-Lambert law.

Attribution
-----------
Algorithm: Ruifrok & Johnston, "Quantification of histochemical staining
by color deconvolution", Analytical and Quantitative Cytology and
Histology, 23(4), 2001.

Fiji implementation:
``src/main/java/sc/fiji/colourDeconvolution/Colour_Deconvolution.java``
Source: https://github.com/fiji/Colour_Deconvolution (GPL-2).
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
from grdl.image_processing.params import Desc, Options
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC

STAIN_PRESETS = ('custom', 'H_E', 'H_DAB', 'FastRed_FastBlue')

# Predefined stain matrices (rows = stain vectors in OD space)
_PRESET_MATRICES = {
    'H_E': np.array([
        [0.6442, 0.7166, 0.2668],  # Hematoxylin
        [0.0927, 0.9545, 0.2832],  # Eosin
        [0.6340, 0.0010, 0.7734],  # Residual
    ]),
    'H_DAB': np.array([
        [0.6500, 0.7041, 0.2862],  # Hematoxylin
        [0.2687, 0.5706, 0.7767],  # DAB
        [0.7110, 0.4250, 0.5607],  # Residual
    ]),
    'FastRed_FastBlue': np.array([
        [0.2115, 0.8510, 0.4788],  # Fast Red
        [0.5449, 0.0605, 0.8365],  # Fast Blue
        [0.0010, 0.9975, 0.0458],  # Residual
    ]),
}


@processor_tags(modalities=[IM.MSI, IM.EO, IM.PAN],
                category=PC.ENHANCE)
@processor_version('3.0.0')
class ColorDeconvolution(ImageTransform):
    """Color deconvolution / spectral unmixing, ported from Fiji.

    Separates multi-channel imagery into component contributions
    using a mixing matrix.

    Parameters
    ----------
    stain_preset : str
        Predefined stain matrix: ``'H_E'`` (hematoxylin-eosin),
        ``'H_DAB'``, ``'FastRed_FastBlue'``, or ``'custom'``
        (provide via ``stain_matrix`` kwarg). Default ``'custom'``.
    normalize : bool
        Normalize stain vectors to unit length. Default True.

    Notes
    -----
    Independent reimplementation of Fiji ``Colour_Deconvolution.java``
    (GPL-2). Follows Ruifrok & Johnston (2001).

    Pipeline: convert to OD space (``OD = -log10(I / I_0)``) ->
    invert mixing matrix -> unmix -> return separated channels.

    For ``'custom'`` preset, pass a 3x3 stain matrix via the
    ``stain_matrix`` keyword argument.

    Examples
    --------
    >>> from grdl_imagej import ColorDeconvolution
    >>> cd = ColorDeconvolution(stain_preset='H_E')
    >>> channels = cd.apply(rgb_image)
    """

    __imagej_source__ = 'fiji/Colour_Deconvolution/Colour_Deconvolution.java'
    __imagej_version__ = '3.0.0'
    __gpu_compatible__ = True

    stain_preset: Annotated[str, Options(*STAIN_PRESETS),
                            Desc('Predefined stain matrix')] = 'custom'
    normalize: Annotated[bool, Desc('Normalize stain vectors')] = True

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply color deconvolution.

        Parameters
        ----------
        source : np.ndarray
            3-channel image. Shape ``(rows, cols, 3)``.
        stain_matrix : np.ndarray, optional
            3x3 mixing matrix (required if stain_preset='custom').

        Returns
        -------
        np.ndarray
            Unmixed channels, shape ``(rows, cols, 3)``, dtype float64.

        Raises
        ------
        ValueError
            If source is not 3-channel or stain matrix not provided.
        """
        if source.ndim != 3 or source.shape[2] != 3:
            raise ValueError(
                f"Expected 3-channel image, got shape {source.shape}"
            )

        p = self._resolve_params(kwargs)

        preset = p['stain_preset']
        if preset == 'custom':
            stain_matrix = kwargs.get('stain_matrix', None)
            if stain_matrix is None:
                raise ValueError(
                    "'stain_matrix' kwarg required when stain_preset='custom'"
                )
            M = np.asarray(stain_matrix, dtype=np.float64)
        else:
            M = _PRESET_MATRICES[preset].copy()

        if M.shape != (3, 3):
            raise ValueError(f"Stain matrix must be 3x3, got {M.shape}")

        # Normalize stain vectors
        if p['normalize']:
            for i in range(M.shape[0]):
                norm = np.linalg.norm(M[i])
                if norm > 0:
                    M[i] /= norm

        # Invert mixing matrix
        M_inv = np.linalg.inv(M)

        image = source.astype(np.float64)
        rows, cols = image.shape[:2]

        # Convert to optical density: OD = -log10(I / I_0)
        # Assume I_0 = 255 for 8-bit, or max value
        I_0 = max(image.max(), 1.0)
        safe_image = np.maximum(image / I_0, 1e-6)
        od = -np.log10(safe_image)

        # Reshape to (n_pixels, 3) for matrix multiplication
        od_flat = od.reshape(-1, 3)

        # Unmix: channels = OD @ M_inv^T
        unmixed = od_flat @ M_inv.T

        return unmixed.reshape(rows, cols, 3)
