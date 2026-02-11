# -*- coding: utf-8 -*-
"""
Performance benchmarks for grdl-imagej image processing components.

Uses pytest-benchmark to track execution time for key operations.
Run with: ``pytest tests/test_benchmarks.py --benchmark-only``

Skip benchmarks during normal test runs with:
``pytest tests/ --benchmark-disable``

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
2026-02-10

Modified
--------
2026-02-10
"""

import numpy as np
import pytest

# Mark all tests in this module as benchmarks so they can be skipped
# during normal test runs: pytest tests/ -m "not benchmark"
pytestmark = pytest.mark.benchmark


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_image():
    """64x64 random image for fast benchmarks."""
    rng = np.random.RandomState(42)
    return rng.rand(64, 64) * 255.0


@pytest.fixture
def medium_image():
    """256x256 random image for realistic benchmarks."""
    rng = np.random.RandomState(42)
    return rng.rand(256, 256) * 255.0


@pytest.fixture
def stack_3d():
    """5-band 64x64 image stack."""
    rng = np.random.RandomState(42)
    return rng.rand(5, 64, 64) * 255.0


# ---------------------------------------------------------------------------
# CLAHE Benchmarks
# ---------------------------------------------------------------------------

class TestCLAHEBenchmarks:
    """Benchmark CLAHE vectorized vs reference implementation."""

    def test_clahe_vectorized(self, benchmark, medium_image):
        from grdl_imagej import CLAHE
        clahe = CLAHE(block_size=63, n_bins=256, max_slope=3.0)
        benchmark(clahe.apply, medium_image)

    def test_clahe_reference(self, benchmark, small_image):
        from grdl_imagej import CLAHE
        clahe = CLAHE(block_size=31, n_bins=256, max_slope=3.0)
        benchmark(clahe.apply_reference, small_image)


# ---------------------------------------------------------------------------
# SRM Benchmarks
# ---------------------------------------------------------------------------

class TestSRMBenchmarks:
    """Benchmark SRM with vectorized edge construction."""

    def test_srm_small(self, benchmark, small_image):
        from grdl_imagej import StatisticalRegionMerging
        srm = StatisticalRegionMerging(Q=25)
        benchmark(srm.apply, small_image)


# ---------------------------------------------------------------------------
# Spatial Filter Benchmarks
# ---------------------------------------------------------------------------

class TestSpatialFilterBenchmarks:
    """Benchmark spatial filter operations."""

    def test_rolling_ball(self, benchmark, small_image):
        from grdl_imagej import RollingBallBackground
        rb = RollingBallBackground(radius=10)
        benchmark(rb.apply, small_image)

    def test_unsharp_mask(self, benchmark, medium_image):
        from grdl_imagej import UnsharpMask
        usm = UnsharpMask(sigma=2.0, weight=0.6)
        benchmark(usm.apply, medium_image)

    def test_edge_detector_sobel(self, benchmark, medium_image):
        from grdl_imagej import EdgeDetector
        ed = EdgeDetector(method='sobel')
        benchmark(ed.apply, medium_image)

    def test_rank_filter_median(self, benchmark, small_image):
        from grdl_imagej import RankFilters
        rf = RankFilters(method='median', radius=2)
        benchmark(rf.apply, small_image)

    def test_gamma_correction(self, benchmark, medium_image):
        from grdl_imagej import GammaCorrection
        gc = GammaCorrection(gamma=0.5)
        benchmark(gc.apply, medium_image)


# ---------------------------------------------------------------------------
# Pipeline Benchmarks
# ---------------------------------------------------------------------------

class TestPipelineBenchmarks:
    """Benchmark Pipeline composition overhead."""

    def test_pipeline_three_steps(self, benchmark, medium_image):
        from grdl_imagej import GammaCorrection, EdgeDetector
        from grdl.image_processing import Pipeline

        pipe = Pipeline([
            GammaCorrection(gamma=0.5),
            GammaCorrection(gamma=2.0),
            EdgeDetector(method='sobel'),
        ])
        benchmark(pipe.apply, medium_image)
