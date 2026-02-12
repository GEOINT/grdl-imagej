# -*- coding: utf-8 -*-
"""
ImageJ/Fiji Ports - Classic image processing algorithms ported from ImageJ/Fiji.

Pure-NumPy reimplementations of widely used ImageJ and Fiji image processing
algorithms, selected for relevance to remotely sensed imagery (PAN, MSI, HSI,
SAR, thermal). Each class mirrors the original ImageJ/Fiji algorithm as closely
as possible, preserving default parameter values and algorithmic behavior.

All ported components inherit from ``ImageTransform`` and carry attribution
to the original ImageJ/Fiji authors. Version strings mirror the original
source version from which the port was derived.

Components (organized by ImageJ menu category)
------------------------------------------------
Process > Filters (filters/):
- RankFilters: Median, Min, Max, Mean, Variance, Despeckle
- UnsharpMask: Gaussian-based sharpening
- GaussianBlur: Isotropic/anisotropic Gaussian smoothing
- Convolver: Arbitrary 2D kernel convolution
- DifferenceOfGaussians: Band-pass filter via Gaussian subtraction (DoG)
- Shadows: Directional shadow/emboss effects
- Smooth: Fixed 3x3 mean smoothing
- Sharpen: Fixed 3x3 Laplacian sharpening
- VarianceFilter: Local variance / standard deviation
- EntropyFilter: Local Shannon entropy
- KuwaharaFilter: Edge-preserving quadrant-based smoothing
- LocalBinaryPatterns: LBP texture micro-pattern encoding
- GaborFilterBank: Multi-orientation Gabor texture filters
- FrangiVesselness: Multi-scale Hessian-based tubular structure enhancement

Process > Subtract Background (background/):
- RollingBallBackground: Background subtraction via Sternberg's rolling ball
- PseudoFlatField: Illumination correction via Gaussian division
- SlidingParaboloid: Sliding paraboloid background subtraction

Process > Binary (binary/):
- MorphologicalFilter: Erode, Dilate, Open, Close, TopHat, BlackHat, Gradient
- DistanceTransform: Euclidean Distance Map (EDM)
- Skeletonize: Zhang-Suen binary thinning
- BinaryOutline: 1-pixel-wide object outlines
- BinaryFillHoles: Flood-fill interior holes
- MorphologicalReconstruction: Geodesic reconstruction by dilation/erosion
- MorphologicalGradient: Beucher, internal, external morphological gradients
- MorphologicalLaplacian: Non-linear morphological Laplacian
- DirectionalFilter: Oriented line SE morphological filtering
- KillBorders: Remove border-touching connected components

Process > Enhance Contrast (enhance/):
- CLAHE: Contrast Limited Adaptive Histogram Equalization
- GammaCorrection: Power-law intensity transform
- ContrastEnhancer: Linear histogram stretching with saturation
- ColorSpaceConverter: RGB to/from HSB, Lab, YCbCr
- WhiteBalance: Gray-world, white-patch, percentile color normalization
- ColorDeconvolution: Spectral unmixing via stain matrix inversion

Process > Find Edges (edges/):
- EdgeDetector: Sobel, Prewitt, Roberts, LoG, Scharr
- HarrisCornerDetector: Structure tensor-based corner detection
- RidgeDetection: Steger's curvilinear structure detection

Process > FFT (fft/):
- FFTBandpassFilter: Frequency-domain bandpass and stripe suppression
- PhaseCorrelation: Translational shift estimation via FFT
- RichardsonLucy: Iterative ML deconvolution
- WienerFilter: Frequency-domain regularized deconvolution
- FFTCustomFilter: User-defined frequency-domain filtering
- TemplateMatching: Normalized cross-correlation pattern detection

Process > Find Maxima (find_maxima/):
- FindMaxima: Prominence-based peak/target detection
- HoughTransform: Line and circle detection via parameter space voting

Process > Math (math/):
- ImageCalculator: Pixel-wise arithmetic and logical operations
- MathOperations: Per-pixel math (add, log, exp, sqrt, etc.)
- TypeConverter: Image dtype conversion with scaling

Image > Adjust > Threshold (threshold/):
- AutoLocalThreshold: Local thresholding (Bernsen, Niblack, Sauvola, etc.)
- AutoThreshold: Global thresholding (Otsu, Triangle, Huang, Li, etc.)

Plugins > Segmentation (segmentation/):
- StatisticalRegionMerging: SRM region-based segmentation
- Watershed: EDT-based watershed for splitting touching objects
- MarkerControlledWatershed: Marker-controlled watershed segmentation
- ExtendedMinMax: H-minima/maxima and regional extrema detection

Image > Stacks (stacks/):
- ZProjection: Stack projection (max, mean, median, min, sum, std)

Analyze > Analyze Particles (analyze/):
- AnalyzeParticles: Connected component analysis with measurements
- GLCMHaralick: GLCM texture features (energy, entropy, contrast, etc.)
- StructureTensor: Orientation, coherence, and energy from gradient tensor
- Granulometry: Morphological size distribution analysis
- TamuraTexture: Perceptual texture features (coarseness, contrast, directionality)

Plugins > Noise (noise/):
- AnisotropicDiffusion: Perona-Malik edge-preserving smoothing
- BilateralFilter: Edge-preserving bilateral smoothing
- NoiseGenerator: Synthetic noise (Gaussian, Poisson, salt-pepper, speckle)
- NonLocalMeans: Non-local means denoising
- ROFDenoise: Rudin-Osher-Fatemi total variation denoising

Attribution
-----------
ImageJ is developed by Wayne Rasband at the U.S. National Institutes of Health.
ImageJ 1.x source code is in the public domain.

Fiji plugins (CLAHE, Auto Local Threshold, Statistical Region Merging,
Auto Threshold, Anisotropic Diffusion) are distributed under GPL-2.
This module provides independent reimplementations in NumPy, not
derivative works of the GPL source, but follows the same published
algorithms and cites the original authors.

Author
------
Steven Siebert
Jason Fritz

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-06

Modified
--------
2026-02-11
"""

# Process > Filters
from grdl_imagej.filters import (
    RankFilters, UnsharpMask, GaussianBlur, Convolver,
    DifferenceOfGaussians, Shadows, Smooth, Sharpen,
    VarianceFilter, EntropyFilter, KuwaharaFilter,
    LocalBinaryPatterns, GaborFilterBank, FrangiVesselness,
)

# Process > Subtract Background
from grdl_imagej.background import RollingBallBackground, PseudoFlatField, SlidingParaboloid

# Process > Binary
from grdl_imagej.binary import (
    MorphologicalFilter, DistanceTransform, Skeletonize,
    BinaryOutline, BinaryFillHoles, MorphologicalReconstruction,
    MorphologicalGradient, MorphologicalLaplacian,
    DirectionalFilter, KillBorders,
)

# Process > Enhance Contrast
from grdl_imagej.enhance import (
    CLAHE, GammaCorrection, ContrastEnhancer,
    ColorSpaceConverter, WhiteBalance, ColorDeconvolution,
)

# Process > Find Edges
from grdl_imagej.edges import EdgeDetector, HarrisCornerDetector, RidgeDetection

# Process > FFT
from grdl_imagej.fft import (
    FFTBandpassFilter, PhaseCorrelation, RichardsonLucy,
    WienerFilter, FFTCustomFilter, TemplateMatching,
)

# Process > Find Maxima
from grdl_imagej.find_maxima import FindMaxima, HoughTransform

# Process > Math
from grdl_imagej.math import ImageCalculator, MathOperations, TypeConverter

# Image > Adjust > Threshold
from grdl_imagej.threshold import AutoLocalThreshold, AutoThreshold

# Plugins > Segmentation
from grdl_imagej.segmentation import (
    StatisticalRegionMerging, Watershed, MarkerControlledWatershed, ExtendedMinMax,
)

# Image > Stacks
from grdl_imagej.stacks import ZProjection

# Analyze > Analyze Particles
from grdl_imagej.analyze import (
    AnalyzeParticles, GLCMHaralick, StructureTensor, Granulometry, TamuraTexture,
)

# Plugins > Noise
from grdl_imagej.noise import (
    AnisotropicDiffusion, BilateralFilter, NoiseGenerator, NonLocalMeans, ROFDenoise,
)

__all__ = [
    # Process > Filters
    'RankFilters', 'UnsharpMask', 'GaussianBlur', 'Convolver',
    'DifferenceOfGaussians', 'Shadows', 'Smooth', 'Sharpen',
    'VarianceFilter', 'EntropyFilter', 'KuwaharaFilter',
    'LocalBinaryPatterns', 'GaborFilterBank', 'FrangiVesselness',
    # Process > Subtract Background
    'RollingBallBackground', 'PseudoFlatField', 'SlidingParaboloid',
    # Process > Binary
    'MorphologicalFilter', 'DistanceTransform', 'Skeletonize',
    'BinaryOutline', 'BinaryFillHoles', 'MorphologicalReconstruction',
    'MorphologicalGradient', 'MorphologicalLaplacian',
    'DirectionalFilter', 'KillBorders',
    # Process > Enhance Contrast
    'CLAHE', 'GammaCorrection', 'ContrastEnhancer',
    'ColorSpaceConverter', 'WhiteBalance', 'ColorDeconvolution',
    # Process > Find Edges
    'EdgeDetector', 'HarrisCornerDetector', 'RidgeDetection',
    # Process > FFT
    'FFTBandpassFilter', 'PhaseCorrelation', 'RichardsonLucy',
    'WienerFilter', 'FFTCustomFilter', 'TemplateMatching',
    # Process > Find Maxima
    'FindMaxima', 'HoughTransform',
    # Process > Math
    'ImageCalculator', 'MathOperations', 'TypeConverter',
    # Image > Adjust > Threshold
    'AutoLocalThreshold', 'AutoThreshold',
    # Plugins > Segmentation
    'StatisticalRegionMerging', 'Watershed', 'MarkerControlledWatershed', 'ExtendedMinMax',
    # Image > Stacks
    'ZProjection',
    # Analyze > Analyze Particles
    'AnalyzeParticles', 'GLCMHaralick', 'StructureTensor', 'Granulometry', 'TamuraTexture',
    # Plugins > Noise
    'AnisotropicDiffusion', 'BilateralFilter', 'NoiseGenerator', 'NonLocalMeans', 'ROFDenoise',
]
