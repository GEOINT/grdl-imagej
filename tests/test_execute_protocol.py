# -*- coding: utf-8 -*-
"""
Execute protocol tests for all 64 grdl-imagej processors.

Verifies that every ImageTransform subclass works correctly with the
``execute(metadata, source, **kwargs) -> (result, metadata)`` protocol.

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
2026-02-12
"""

from __future__ import annotations

import numpy as np
import pytest

import grdl_imagej
from grdl.IO.models.base import ImageMetadata
from grdl.image_processing.base import ImageTransform

ALL_NAMES = grdl_imagej.__all__

# ---------------------------------------------------------------------------
# Test-case factory
# ---------------------------------------------------------------------------

_RNG = np.random.RandomState(42)
_SOURCE_2D = _RNG.rand(50, 50) * 200
_SOURCE_3CH = (_RNG.rand(50, 50, 3) * 255).astype(np.float64)
_SOURCE_STACK = _RNG.rand(5, 50, 50) * 200

_META_2D = ImageMetadata(
    format='test', rows=50, cols=50, dtype='float64', bands=1,
)
_META_3CH = ImageMetadata(
    format='test', rows=50, cols=50, dtype='float64', bands=3,
)
_META_STACK = ImageMetadata(
    format='test', rows=50, cols=50, dtype='float64', bands=5,
)

# Processors that need 3-channel (H, W, 3) input
_THREE_CHANNEL = {'ColorSpaceConverter', 'WhiteBalance', 'ColorDeconvolution'}

# Processors that need 3-D stack (slices, H, W) input
_STACK_INPUT = {'ZProjection'}


def _make_processor(cls_name: str):
    """Instantiate a processor, handling special constructors."""
    cls = getattr(grdl_imagej, cls_name)
    if cls_name == 'Convolver':
        return cls(kernel=np.ones((3, 3)))
    if cls_name == 'ColorDeconvolution':
        return cls(stain_preset='H_E')
    return cls()


def _make_source(cls_name: str) -> tuple[np.ndarray, ImageMetadata]:
    """Return (source, metadata) appropriate for the processor."""
    if cls_name in _THREE_CHANNEL:
        return _SOURCE_3CH.copy(), _META_3CH
    if cls_name in _STACK_INPUT:
        return _SOURCE_STACK.copy(), _META_STACK
    return _SOURCE_2D.copy(), _META_2D


def _make_kwargs(cls_name: str, source: np.ndarray) -> dict:
    """Return extra kwargs needed by apply() for specific processors."""
    if cls_name == 'TemplateMatching':
        return {'template': source[:10, :10].copy()}
    if cls_name in ('RichardsonLucy', 'WienerFilter'):
        return {'psf': np.ones((5, 5)) / 25.0}
    if cls_name == 'MarkerControlledWatershed':
        markers = np.zeros(source.shape[:2], dtype=np.int32)
        markers[10, 10] = 1
        markers[40, 40] = 2
        return {'markers': markers}
    if cls_name == 'MorphologicalReconstruction':
        marker = source * 0.5  # marker must be <= source
        return {'marker': marker}
    if cls_name == 'FFTCustomFilter':
        # Frequency-domain mask, same shape as source
        rows, cols = source.shape[:2]
        mask = np.ones((rows, cols), dtype=np.float64)
        return {'mask': mask}
    if cls_name == 'ImageCalculator':
        return {'image2': source * 0.5}
    if cls_name == 'PhaseCorrelation':
        return {'reference': source.copy()}
    return {}


# ---------------------------------------------------------------------------
# Parametrized protocol tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls_name", ALL_NAMES)
def test_is_image_transform(cls_name):
    """Every grdl-imagej processor inherits from ImageTransform."""
    cls = getattr(grdl_imagej, cls_name)
    assert issubclass(cls, ImageTransform), (
        f"{cls_name} is not an ImageTransform subclass"
    )


@pytest.mark.parametrize("cls_name", ALL_NAMES)
def test_execute_protocol(cls_name):
    """execute() returns (ndarray, ImageMetadata) with correct shape info."""
    proc = _make_processor(cls_name)
    source, meta = _make_source(cls_name)
    kwargs = _make_kwargs(cls_name, source)

    out = proc.execute(meta, source, **kwargs)

    # Returns a 2-tuple
    assert isinstance(out, tuple), f"{cls_name}.execute() must return a tuple"
    assert len(out) == 2, f"{cls_name}.execute() must return 2 elements"

    result, out_meta = out

    # Result is ndarray
    assert isinstance(result, np.ndarray), (
        f"{cls_name}.execute() result must be ndarray, got {type(result)}"
    )

    # Metadata is ImageMetadata with updated dtype
    assert isinstance(out_meta, ImageMetadata), (
        f"{cls_name}.execute() metadata must be ImageMetadata"
    )
    assert out_meta.dtype == str(result.dtype), (
        f"{cls_name}: metadata dtype={out_meta.dtype} "
        f"doesn't match result dtype={result.dtype}"
    )

    # Shape reflected in metadata
    if result.ndim >= 2:
        assert out_meta.rows == result.shape[0], (
            f"{cls_name}: metadata rows={out_meta.rows} "
            f"doesn't match result shape[0]={result.shape[0]}"
        )
        assert out_meta.cols == result.shape[1], (
            f"{cls_name}: metadata cols={out_meta.cols} "
            f"doesn't match result shape[1]={result.shape[1]}"
        )


# ---------------------------------------------------------------------------
# Targeted protocol behavior tests
# ---------------------------------------------------------------------------

def test_metadata_property_set_during_execute():
    """self.metadata is accessible after execute() but None before."""
    from grdl_imagej import GaussianBlur

    proc = GaussianBlur(sigma=2.0)
    assert proc.metadata is None

    source, meta = _make_source('GaussianBlur')
    proc.execute(meta, source)
    assert proc.metadata is meta


def test_execute_matches_apply():
    """execute() result matches direct apply() result."""
    from grdl_imagej import UnsharpMask

    proc1 = UnsharpMask()
    proc2 = UnsharpMask()
    source, meta = _make_source('UnsharpMask')

    result_execute, _ = proc1.execute(meta, source)
    result_apply = proc2.apply(source)

    np.testing.assert_array_equal(result_execute, result_apply)


def test_metadata_chain_through_two_transforms():
    """Metadata propagates correctly through a two-step chain."""
    from grdl_imagej import GaussianBlur, EdgeDetector

    source, meta = _make_source('GaussianBlur')

    blur = GaussianBlur(sigma=2.0)
    result1, meta1 = blur.execute(meta, source)
    assert meta1.rows == result1.shape[0]

    edge = EdgeDetector()
    result2, meta2 = edge.execute(meta1, result1)
    assert meta2.rows == result2.shape[0]
    assert meta2.dtype == str(result2.dtype)


def test_convolver_execute_with_kernel():
    """Convolver (custom constructor) works through execute()."""
    from grdl_imagej import Convolver

    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64)
    proc = Convolver(kernel=kernel)
    source, meta = _make_source('Convolver')

    result, out_meta = proc.execute(meta, source)
    assert result.shape == source.shape
    assert out_meta.dtype == str(result.dtype)


def test_z_projection_3d_to_2d():
    """ZProjection reduces 3D stack to 2D, metadata updated."""
    from grdl_imagej import ZProjection

    proc = ZProjection(method='max')
    source, meta = _make_source('ZProjection')

    result, out_meta = proc.execute(meta, source)
    assert result.ndim == 2
    assert out_meta.bands == 1
    assert out_meta.rows == result.shape[0]
    assert out_meta.cols == result.shape[1]
