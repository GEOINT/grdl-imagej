# TODO — ImageJ/Fiji Plugin Porting Candidates

Comprehensive catalog of open-source ImageJ, Fiji, and ImageJ2/SciJava algorithms
that are strong candidates for porting into `grdl-imagej`. Each entry contains
everything an agent needs to implement the port: algorithm name, category, source
location, parameter spec, complexity estimate, and academic references.

**Base repository:** `https://github.com/imagej/ImageJ` (ImageJ 1.x, public domain)
**Fiji plugins:** `https://github.com/fiji` (GPL-2, independent reimplementation required)
**ImageJ2 Ops:** `https://github.com/imagej/imagej-ops` (BSD-2)
**MorphoLibJ:** `https://github.com/ijpb/MorphoLibJ` (LGPL-3)

> All ports must be **independent NumPy/SciPy reimplementations** that follow the
> same published algorithms and cite the original authors. See `CLAUDE.md` for
> the processor template and coding conventions.

---

## Already Ported (42 processors — excluded from all candidates below)

| # | Processor | Category | Module |
|---|-----------|----------|--------|
| 1 | RollingBallBackground | background | `grdl_imagej.background` |
| 2 | CLAHE | enhance | `grdl_imagej.enhance` |
| 3 | AutoLocalThreshold | threshold | `grdl_imagej.threshold` |
| 4 | UnsharpMask | filters | `grdl_imagej.filters` |
| 5 | FFTBandpassFilter | fft | `grdl_imagej.fft` |
| 6 | ZProjection | stacks | `grdl_imagej.stacks` |
| 7 | RankFilters | filters | `grdl_imagej.filters` |
| 8 | MorphologicalFilter | binary | `grdl_imagej.binary` |
| 9 | EdgeDetector | edges | `grdl_imagej.edges` |
| 10 | GammaCorrection | enhance | `grdl_imagej.enhance` |
| 11 | FindMaxima | find_maxima | `grdl_imagej.find_maxima` |
| 12 | StatisticalRegionMerging | segmentation | `grdl_imagej.segmentation` |
| 13 | GaussianBlur | filters | `grdl_imagej.filters` |
| 14 | Convolver | filters | `grdl_imagej.filters` |
| 15 | AutoThreshold | threshold | `grdl_imagej.threshold` |
| 16 | Watershed | segmentation | `grdl_imagej.segmentation` |
| 17 | AnalyzeParticles | analyze | `grdl_imagej.analyze` |
| 18 | ImageCalculator | math | `grdl_imagej.math` |
| 19 | ContrastEnhancer | enhance | `grdl_imagej.enhance` |
| 20 | DistanceTransform | binary | `grdl_imagej.binary` |
| 21 | Skeletonize | binary | `grdl_imagej.binary` |
| 22 | AnisotropicDiffusion | noise | `grdl_imagej.noise` |
| 23 | BilateralFilter | noise | `grdl_imagej.noise` |
| 24 | DifferenceOfGaussians | filters | `grdl_imagej.filters` |
| 25 | GaborFilterBank | filters | `grdl_imagej.filters` |
| 26 | LocalBinaryPatterns | filters | `grdl_imagej.filters` |
| 27 | HarrisCornerDetector | edges | `grdl_imagej.edges` |
| 28 | PhaseCorrelation | fft | `grdl_imagej.fft` |
| 29 | NoiseGenerator | noise | `grdl_imagej.noise` |
| 30 | ColorSpaceConverter | enhance | `grdl_imagej.enhance` |
| 31 | Shadows | filters | `grdl_imagej.filters` |
| 32 | EntropyFilter | filters | `grdl_imagej.filters` |
| 33 | VarianceFilter | filters | `grdl_imagej.filters` |
| 34 | KuwaharaFilter | filters | `grdl_imagej.filters` |
| 35 | BinaryFillHoles | binary | `grdl_imagej.binary` |
| 36 | BinaryOutline | binary | `grdl_imagej.binary` |
| 37 | PseudoFlatField | background | `grdl_imagej.background` |
| 38 | Smooth | filters | `grdl_imagej.filters` |
| 39 | Sharpen | filters | `grdl_imagej.filters` |
| 40 | MathOperations | math | `grdl_imagej.math` |
| 41 | TypeConverter | math | `grdl_imagej.math` |
| 42 | WhiteBalance | enhance | `grdl_imagej.enhance` |

---

## Porting Conventions

Each ported processor must:

1. Inherit from `grdl.image_processing.base.ImageTransform` (or `BandwiseTransformMixin`)
2. Use `@processor_version` and `@processor_tags` decorators from `grdl.image_processing.versioning`
3. Declare tunable parameters with `Annotated[type, Range(...)]`, `Annotated[type, Options(...)]`, or `Annotated[type, Desc(...)]` from `grdl.image_processing.params`
4. Include the standard file header (see `CLAUDE.md`)
5. Carry attribution to the original ImageJ/Fiji authors in the class docstring
6. Have unit tests with >75% line coverage and no test pageantry
7. Be registered in the appropriate subdirectory `__init__.py` and in `grdl_imagej/__init__.py`

---

## Priority Tiers

- **Tier 1 — High Value / Low Effort:** ~~Directly useful for GEOINT workflows, low porting complexity. Do these first.~~ **COMPLETE (20/20 ported 2026-02-11)**
- **Tier 2 — High Value / Medium Effort:** Important algorithms that require more implementation work.
- **Tier 3 — Specialized:** Useful for specific domains or as building blocks for higher-level tools.
- **Tier 4 — Advanced / High Effort:** Academically important, complex implementations.

---

## Tier 1 — High Value / Low Effort (COMPLETE — all 20 ported 2026-02-11)

### ~~T1-01. Bilateral Filter~~ DONE
- **Category:** `noise` (denoising)
- **Type:** ImageTransform
- **Description:** Edge-preserving smoothing filter combining a spatial Gaussian kernel with a range (intensity) Gaussian kernel. Smooths homogeneous regions while preserving edges. Critical for SAR speckle reduction and pre-processing.
- **Java source:** `imagej-ops` — `src/main/java/net/imagej/ops/filter/bilateral/DefaultBilateralFilter.java`
- **Repo:** `https://github.com/imagej/imagej-ops`
- **Parameters:**
  - `sigma_spatial: Annotated[float, Range(0.5, 50.0)]` — Spatial Gaussian std dev (neighborhood size)
  - `sigma_range: Annotated[float, Range(1.0, 255.0)]` — Intensity Gaussian std dev (edge sensitivity)
  - `radius: Annotated[int, Range(1, 25)]` — Kernel radius in pixels
- **Algorithm:** For each pixel, compute weighted average of neighbors where weight = G_spatial(distance) × G_range(|intensity_diff|). Separable approximation available for speed.
- **Complexity:** Low (~120-180 lines)
- **Dependencies:** numpy, scipy (gaussian_filter for comparison)
- **References:** Tomasi & Manduchi, "Bilateral Filtering for Gray and Color Images", ICCV 1998.

### ~~T1-02. Difference of Gaussians (DoG)~~ DONE
- **Category:** `filters`
- **Type:** ImageTransform (BandwiseTransformMixin)
- **Description:** Subtracts two Gaussian-blurred images at different scales. Approximates Laplacian of Gaussian (LoG). Key building block for blob detection and scale-space analysis in feature extraction pipelines.
- **Java source:** `imagej-ops` — `src/main/java/net/imagej/ops/filter/dog/DefaultDoG.java`
- **Repo:** `https://github.com/imagej/imagej-ops`
- **Parameters:**
  - `sigma1: Annotated[float, Range(0.5, 20.0)]` — Smaller Gaussian sigma
  - `sigma2: Annotated[float, Range(1.0, 40.0)]` — Larger Gaussian sigma (or use `k` ratio where sigma2 = k × sigma1)
- **Algorithm:** `DoG = GaussianBlur(image, sigma1) - GaussianBlur(image, sigma2)`. Trivially uses existing GaussianBlur.
- **Complexity:** Low (~40-60 lines)
- **Dependencies:** numpy; reuses existing `GaussianBlur`
- **References:** Marr & Hildreth, "Theory of Edge Detection", Proc. Royal Society London B, 207, 1980.

### ~~T1-03. Gabor Filter Bank~~ DONE
- **Category:** `filters` (texture)
- **Type:** ImageTransform (BandwiseTransformMixin)
- **Description:** Applies a bank of Gabor filters at multiple orientations and frequencies. Each filter is a Gaussian-modulated sinusoidal plane wave sensitive to a specific spatial frequency and orientation. Essential for texture classification in land cover mapping.
- **Java source:** Fiji Trainable Segmentation — `src/main/java/trainableSegmentation/GaborFilter.java`
- **Repo:** `https://github.com/fiji/Trainable_Segmentation`
- **Parameters:**
  - `sigma: Annotated[float, Range(1.0, 20.0)]` — Gaussian envelope width
  - `n_orientations: Annotated[int, Range(2, 32)]` — Number of orientations (default 8)
  - `lambda_: Annotated[float, Range(2.0, 50.0)]` — Wavelength of sinusoidal component
  - `gamma: Annotated[float, Range(0.1, 1.0)]` — Spatial aspect ratio (default 0.5)
  - `psi: Annotated[float, Range(0.0, 3.14159)]` — Phase offset
- **Algorithm:** For each orientation θ in [0, π): build kernel `g(x,y) = exp(-(x'^2 + γ²y'^2)/(2σ²)) × cos(2π x'/λ + ψ)` where x',y' are rotated coordinates. Convolve and stack results.
- **Complexity:** Low (~100-150 lines)
- **Dependencies:** numpy, scipy.ndimage.convolve
- **References:** Jain & Farrokhnia, "Unsupervised Texture Segmentation Using Gabor Filters", Pattern Recognition, 24(12), 1991.

### ~~T1-04. Local Binary Patterns (LBP)~~ DONE
- **Category:** `filters` (texture)
- **Type:** ImageTransform (BandwiseTransformMixin)
- **Description:** Encodes local texture micro-patterns by comparing each pixel to its circular neighborhood, producing binary codes. Variants include uniform LBP and rotation-invariant LBP. Powerful texture descriptor for terrain classification.
- **Java source:** `imagej-ops` — `src/main/java/net/imagej/ops/features/lbp2d/` (LBP2D feature ops)
- **Repo:** `https://github.com/imagej/imagej-ops`
- **Parameters:**
  - `radius: Annotated[int, Range(1, 5)]` — Neighborhood radius (default 1)
  - `n_neighbors: Annotated[int, Options(8, 16, 24)]` — Sampling points on circle (default 8)
  - `method: Annotated[str, Options("default", "uniform", "rotation_invariant")]` — LBP variant
- **Algorithm:** For each pixel: sample N points on circle of radius R using bilinear interpolation → threshold against center → encode as binary number → output code (or uniform pattern index).
- **Complexity:** Low (~100-150 lines)
- **Dependencies:** numpy
- **References:** Ojala, Pietikainen & Maenpaa, "Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns", IEEE PAMI, 24(7), 2002.

### ~~T1-05. Harris Corner Detection~~ DONE
- **Category:** `edges` (feature detection)
- **Type:** ImageTransform (BandwiseTransformMixin)
- **Description:** Detects corner points where the image gradient has significant variation in multiple directions. Computes structure tensor and derives corner response function with non-maximum suppression. Useful for feature matching in co-registration.
- **Java source:** Related to structure tensor in `imagej-ops` — `src/main/java/net/imagej/ops/features/`
- **Repo:** `https://github.com/imagej/imagej-ops`
- **Parameters:**
  - `sigma: Annotated[float, Range(0.5, 5.0)]` — Gaussian smoothing for structure tensor
  - `k: Annotated[float, Range(0.01, 0.15)]` — Harris free parameter (default 0.04)
  - `threshold: Annotated[float, Range(0.0, 1.0)]` — Corner response threshold (fraction of max)
  - `nms_radius: Annotated[int, Range(1, 15)]` — Non-maximum suppression radius
- **Algorithm:** Compute Ix, Iy gradients → form structure tensor M = [Ixx Ixy; Ixy Iyy] smoothed by Gaussian → R = det(M) - k·trace(M)² → threshold → NMS.
- **Complexity:** Low (~80-120 lines)
- **Dependencies:** numpy, scipy.ndimage (gaussian_filter)
- **References:** Harris & Stephens, "A Combined Corner and Edge Detector", Alvey Vision Conference, 1988.

### ~~T1-06. Phase Correlation~~ DONE
- **Category:** `fft` (registration)
- **Type:** ImageTransform
- **Description:** Estimates translational shift between two images using normalized cross-power spectrum in frequency domain. Sub-pixel accurate, very fast via FFT, robust to intensity differences. Critical for SAR/EO image co-registration.
- **Java source:** `imagej-ops` — `src/main/java/net/imagej/ops/filter/correlate/CorrelateFFTC.java`; ImageJ core — `ij/process/FHT.java`
- **Repo:** `https://github.com/imagej/imagej-ops`
- **Parameters:**
  - `upsample_factor: Annotated[int, Range(1, 100)]` — Sub-pixel accuracy factor (default 10)
  - `normalize: Annotated[bool, Desc("Use normalized cross-power spectrum")]` — (default True)
  - `window: Annotated[str, Options("hann", "blackman", "none")]` — Apodization window
- **Algorithm:** `F1 = FFT(img1)`, `F2 = FFT(img2)` → cross-power spectrum `R = (F1 · conj(F2)) / |F1 · conj(F2)|` → `IFFT(R)` → find peak → parabolic sub-pixel interpolation.
- **Complexity:** Low (~80-120 lines)
- **Dependencies:** numpy (np.fft)
- **References:** Kuglin & Hines, "The Phase Correlation Image Alignment Method", IEEE Conf. Cybernetics & Society, 1975. Guizar-Sicairos et al., "Efficient subpixel image registration algorithms", Optics Letters, 33(2), 2008.

### ~~T1-07. Noise Generator~~ DONE
- **Category:** `noise`
- **Type:** ImageTransform
- **Description:** Adds synthetic noise to images: Gaussian, Poisson (shot noise), salt-and-pepper, and speckle (multiplicative). Essential for testing denoising algorithms and data augmentation.
- **Java source:** ImageJ core — `ij/plugin/filter/Filters.java`, `ij/plugin/Noise.java`; `imagej-ops` — `src/main/java/net/imagej/ops/image/noise/`
- **Repo:** `https://github.com/imagej/ImageJ`
- **Parameters:**
  - `noise_type: Annotated[str, Options("gaussian", "poisson", "salt_pepper", "speckle")]` — Noise model
  - `sigma: Annotated[float, Range(0.1, 100.0)]` — Std dev for Gaussian noise
  - `density: Annotated[float, Range(0.0, 0.5)]` — Density for salt-and-pepper
  - `seed: Annotated[int, Desc("Random seed for reproducibility")]` — Optional seed
- **Algorithm:** Gaussian: `output = input + N(0, sigma)`. Poisson: `output = Poisson(input)`. Salt-pepper: random mask at density. Speckle: `output = input + input × N(0, sigma)`. Clip to valid range.
- **Complexity:** Low (~80-100 lines)
- **Dependencies:** numpy (np.random.Generator)
- **References:** Standard statistical noise models.

### ~~T1-08. Color Space Converter~~ DONE
- **Category:** `enhance` (color)
- **Type:** ImageTransform
- **Description:** Converts between color spaces: RGB ↔ HSB(HSV), CIE L\*a\*b\*, YCbCr. Useful for color-based segmentation and analysis where perceptual (Lab) or luminance-chrominance spaces are advantageous.
- **Java source:** ImageJ core — `ij/process/ColorProcessor.java` (`toHSB()`, `toFloat()`), `ij/plugin/ColorConverter.java`
- **Repo:** `https://github.com/imagej/ImageJ`
- **Parameters:**
  - `source_space: Annotated[str, Options("rgb", "hsb", "lab", "ycbcr")]`
  - `target_space: Annotated[str, Options("rgb", "hsb", "lab", "ycbcr")]`
  - `illuminant: Annotated[str, Options("D50", "D65")]` — For Lab conversion (default D65)
- **Algorithm:** Standard colorimetric formulas. HSB: standard conversion. Lab: linearize sRGB → XYZ (3×3 matrix) → Lab (nonlinear). YCbCr: ITU-R BT.601 coefficients. Luminance: 0.299R + 0.587G + 0.114B.
- **Complexity:** Low (~150-200 lines)
- **Dependencies:** numpy
- **References:** CIE (1976); Poynton, "Digital Video and HDTV", Morgan Kaufmann, 2003.

### ~~T1-09. Shadows (Emboss)~~ DONE
- **Category:** `filters`
- **Type:** ImageTransform (BandwiseTransformMixin)
- **Description:** Directional shadow/emboss effects using directional derivative kernels. Eight directional options (N, NE, E, SE, S, SW, W, NW). Useful for terrain feature enhancement in DEM-derived imagery.
- **Java source:** ImageJ core — `ij/plugin/filter/Shadows.java`
- **Repo:** `https://github.com/imagej/ImageJ`
- **Parameters:**
  - `direction: Annotated[str, Options("N", "NE", "E", "SE", "S", "SW", "W", "NW")]` — Shadow direction
  - `offset: Annotated[float, Range(0.0, 255.0)]` — Output offset (default 128 for 8-bit)
- **Algorithm:** Eight predefined 3×3 directional kernels. Standard 2D convolution with offset to show positive and negative gradients. Leverages existing Convolver infrastructure.
- **Complexity:** Low (~60-80 lines)
- **Dependencies:** numpy; reuses existing `Convolver`
- **References:** ImageJ source code (public domain).

### ~~T1-10. Entropy Filter (Local Entropy)~~ DONE
- **Category:** `filters` (texture)
- **Type:** ImageTransform (BandwiseTransformMixin)
- **Description:** Computes local Shannon entropy within a sliding window. Homogeneous regions have low entropy; textured/edge regions have high entropy. Useful as a texture feature for land cover classification.
- **Java source:** `imagej-ops` — `src/main/java/net/imagej/ops/stats/DefaultEntropy.java`
- **Repo:** `https://github.com/imagej/imagej-ops`
- **Parameters:**
  - `radius: Annotated[int, Range(1, 25)]` — Window radius (default 5)
  - `n_bins: Annotated[int, Range(16, 256)]` — Histogram bins for local probability (default 256)
- **Algorithm:** For each pixel: compute histogram of values in window → normalize to probabilities → H = -Σ p·log₂(p). Efficient implementation with integral histograms or generic_filter.
- **Complexity:** Low (~80-120 lines)
- **Dependencies:** numpy, scipy.ndimage (generic_filter)
- **References:** Shannon, "A Mathematical Theory of Communication", Bell System Technical Journal, 27(3), 1948.

### ~~T1-11. Variance / Std Dev Filter~~ DONE
- **Category:** `filters`
- **Type:** ImageTransform (BandwiseTransformMixin)
- **Description:** Computes local variance or standard deviation in a sliding window. Produces texture/variability maps for detecting change regions, noise estimation, and adaptive processing.
- **Java source:** `imagej-ops` — `src/main/java/net/imagej/ops/stats/DefaultVariance.java`
- **Repo:** `https://github.com/imagej/imagej-ops`
- **Parameters:**
  - `radius: Annotated[int, Range(1, 25)]` — Window radius
  - `output: Annotated[str, Options("variance", "std_dev")]` — Output type (default "std_dev")
- **Algorithm:** `Var = E[X²] - E[X]²` using `scipy.ndimage.uniform_filter` for both terms. For std dev: `sqrt(Var)`.
- **Complexity:** Low (~50-80 lines)
- **Dependencies:** numpy, scipy.ndimage
- **References:** Standard statistical filtering.

### ~~T1-12. Kuwahara Filter~~ DONE
- **Category:** `filters`
- **Type:** ImageTransform (BandwiseTransformMixin)
- **Description:** Edge-preserving smoothing that divides each pixel's neighborhood into four overlapping quadrants, computes mean and variance of each, assigns pixel the mean of the quadrant with minimum variance. Produces smooth regions with sharp edges.
- **Java source:** Fiji plugin implementations; related to `ij/plugin/filter/` family
- **Repo:** `https://github.com/fiji/fiji`
- **Parameters:**
  - `radius: Annotated[int, Range(1, 15)]` — Window half-size (default 3)
- **Algorithm:** For each pixel: define 4 overlapping quadrant windows → compute mean and variance in each → output = mean of quadrant with lowest variance. Efficient with NumPy sliding window views.
- **Complexity:** Low (~80-120 lines)
- **Dependencies:** numpy
- **References:** Kuwahara et al., "Processing of RI-angiocardiographic images", Digital Processing of Biomedical Images, Plenum Press, 1976.

### ~~T1-13. BinaryFillHoles~~ DONE
- **Category:** `binary`
- **Type:** ImageTransform
- **Description:** Fills interior holes in binary objects by flood-filling background from image edges and inverting. Part of ImageJ's Process → Binary → Fill Holes.
- **Java source:** ImageJ core — `ij/plugin/filter/Binary.java` (`fill()` method), `ij/process/BinaryProcessor.java`
- **Repo:** `https://github.com/imagej/ImageJ`
- **Parameters:**
  - `connectivity: Annotated[int, Options(4, 8)]` — Flood fill connectivity (default 8)
- **Algorithm:** 8-connected flood fill from all border pixels marking background → invert result. Stack-based flood fill to avoid recursion limits.
- **Complexity:** Low (~60-80 lines)
- **Dependencies:** numpy, scipy.ndimage (binary_fill_holes for reference)
- **References:** ImageJ source code (public domain).

### ~~T1-14. BinaryOutline~~ DONE
- **Category:** `binary`
- **Type:** ImageTransform
- **Description:** Reduces binary objects to 1-pixel-wide outlines by removing interior pixels fully surrounded by foreground. Express as `original - erode(original)`.
- **Java source:** ImageJ core — `ij/process/BinaryProcessor.java` (`outline()` method)
- **Repo:** `https://github.com/imagej/ImageJ`
- **Parameters:**
  - `connectivity: Annotated[int, Options(4, 8)]` — Neighborhood type (default 4)
- **Algorithm:** For each foreground pixel: check 4-connected neighbors → if all foreground, remove. Equivalently: `output = input AND NOT erode(input)`. Single pass or via existing MorphologicalFilter.
- **Complexity:** Low (~40-60 lines)
- **Dependencies:** numpy; reuses existing `MorphologicalFilter`
- **References:** ImageJ source code (public domain).

### ~~T1-15. Pseudo Flat-Field Correction~~ DONE
- **Category:** `background`
- **Type:** ImageTransform (BandwiseTransformMixin)
- **Description:** Corrects uneven illumination (vignetting, shading) by dividing image by a heavily blurred version of itself. Simpler than rolling ball for illumination normalization.
- **Java source:** ImageJ core — macro-based or `ij/plugin/filter/Pseudo_Flat_Field_Correction.java`
- **Repo:** `https://github.com/imagej/ImageJ`
- **Parameters:**
  - `blur_radius: Annotated[float, Range(10.0, 500.0)]` — Gaussian sigma for estimating illumination (default 50.0)
  - `normalize_output: Annotated[bool, Desc("Normalize output to [0,1]")]` — (default True)
- **Algorithm:** `output = input / GaussianBlur(input, blur_radius)`. Normalize if requested. Trivially uses existing GaussianBlur.
- **Complexity:** Low (~40-60 lines)
- **Dependencies:** numpy; reuses existing `GaussianBlur`
- **References:** Model, "Intensity calibration and flat-field correction for fluorescence microscopes", Current Protocols in Cytometry, 2001.

### ~~T1-16. Smooth (Mean Filter)~~ DONE
- **Category:** `filters`
- **Type:** ImageTransform (BandwiseTransformMixin)
- **Description:** Fixed 3×3 mean filter for simple noise reduction. ImageJ's "Smooth" command, distinct from configurable-radius mean in RankFilters.
- **Java source:** ImageJ core — `ij/process/ImageProcessor.java` (`smooth()` method)
- **Repo:** `https://github.com/imagej/ImageJ`
- **Parameters:** None (fixed 3×3 kernel).
- **Algorithm:** 3×3 averaging kernel (all weights = 1/9). Boundary: nearest-neighbor padding.
- **Complexity:** Low (~30-50 lines)
- **Dependencies:** scipy.ndimage (uniform_filter)
- **References:** ImageJ source code (public domain).

### ~~T1-17. Sharpen (Laplacian)~~ DONE
- **Category:** `filters`
- **Type:** ImageTransform (BandwiseTransformMixin)
- **Description:** Fixed 3×3 Laplacian-based sharpening kernel. ImageJ's "Sharpen" command, distinct from configurable UnsharpMask.
- **Java source:** ImageJ core — `ij/process/ImageProcessor.java` (`sharpen()` method)
- **Repo:** `https://github.com/imagej/ImageJ`
- **Parameters:** None (fixed 3×3 kernel).
- **Algorithm:** 3×3 kernel: center=12, edges=-2, corners=-1 (normalized, net gain=1/12). Equivalent to adding scaled Laplacian to original.
- **Complexity:** Low (~30-50 lines)
- **Dependencies:** numpy; reuses existing `Convolver`
- **References:** ImageJ source code (public domain).

### ~~T1-18. MathOperations (Per-pixel Math)~~ DONE
- **Category:** `math`
- **Type:** ImageTransform (BandwiseTransformMixin)
- **Description:** Per-pixel mathematical operations: add, multiply, min, max, log, exp, sqrt, abs, reciprocal, square, NaN replacement. Process → Math submenu.
- **Java source:** ImageJ core — `ij/plugin/filter/Filters.java`, `ij/process/ImageProcessor.java` (`add()`, `multiply()`, `log()`, `exp()`, etc.)
- **Repo:** `https://github.com/imagej/ImageJ`
- **Parameters:**
  - `operation: Annotated[str, Options("add", "subtract", "multiply", "divide", "log", "exp", "sqrt", "square", "abs", "reciprocal", "min", "max", "nan_to_num")]`
  - `value: Annotated[float, Desc("Scalar operand for add/subtract/multiply/divide/min/max")]`
  - `nan_replacement: Annotated[float, Desc("Replacement value for NaN")]` — (default 0.0)
- **Algorithm:** Direct NumPy ufunc mapping with clipping to valid range for integer types.
- **Complexity:** Low (~80-120 lines)
- **Dependencies:** numpy
- **References:** ImageJ source code (public domain).

### ~~T1-19. TypeConverter~~ DONE
- **Category:** `math`
- **Type:** ImageTransform
- **Description:** Converts images between data types (uint8, uint16, float32) with configurable scaling, clamping, and normalization. Image → Type conversions.
- **Java source:** ImageJ core — `ij/process/ImageConverter.java`, `ij/process/ImageProcessor.java`
- **Repo:** `https://github.com/imagej/ImageJ`
- **Parameters:**
  - `target_type: Annotated[str, Options("uint8", "uint16", "float32", "float64")]`
  - `scale: Annotated[bool, Desc("Scale values when converting")]` — (default True)
  - `normalize: Annotated[bool, Desc("Normalize to [0,1] for float output")]` — (default False)
- **Algorithm:** dtype casting with scaling. 8→16: ×256. 16→8: linear min/max scaling. Float→byte: normalize to [0,255]. Handle overflow/underflow/NaN.
- **Complexity:** Low (~60-100 lines)
- **Dependencies:** numpy
- **References:** ImageJ source code (public domain).

### ~~T1-20. White Balance / Color Normalization~~ DONE
- **Category:** `enhance` (color)
- **Type:** ImageTransform
- **Description:** Normalizes color balance using gray-world, white-patch (Retinex), or percentile methods. Corrects illumination color cast for consistent color representation in multi-temporal analysis.
- **Java source:** ImageJ core — `ij/plugin/filter/GrayWorld.java`
- **Repo:** `https://github.com/imagej/ImageJ`
- **Parameters:**
  - `method: Annotated[str, Options("gray_world", "white_patch", "percentile")]`
  - `percentile: Annotated[float, Range(0.1, 5.0)]` — For percentile method (default 1.0)
- **Algorithm:** Gray-world: scale each channel so mean = gray. White-patch: scale so max = white. Percentile: scale so Nth percentile = white.
- **Complexity:** Low (~60-100 lines)
- **Dependencies:** numpy
- **References:** Buchsbaum, "A Spatial Processor Model for Object Colour Perception", J. Franklin Institute, 310(1), 1980.

---

## Tier 2 — High Value / Medium Effort

### T2-01. GLCM / Haralick Texture Features
- **Category:** `analyze` (texture)
- **Type:** ImageTransform
- **Description:** Computes Gray-Level Co-occurrence Matrices and derives Haralick texture descriptors: energy (ASM), entropy, contrast, correlation, homogeneity, dissimilarity, variance. Essential for land cover classification in remote sensing.
- **Java source:** `imagej-ops` — `src/main/java/net/imagej/ops/image/cooccurrenceMatrix/CooccurrenceMatrix2D.java` and `src/main/java/net/imagej/ops/features/haralick/` (14 feature classes)
- **Repo:** `https://github.com/imagej/imagej-ops`
- **Parameters:**
  - `distance: Annotated[int, Range(1, 10)]` — Pixel offset (default 1)
  - `angle: Annotated[str, Options("0", "45", "90", "135", "all")]` — Direction (default "all")
  - `n_gray_levels: Annotated[int, Options(32, 64, 128, 256)]` — Quantization levels (default 64)
  - `symmetric: Annotated[bool, Desc("Symmetric GLCM")]` — (default True)
  - `features: list of str` — Which Haralick features to compute
- **Algorithm:** Quantize image → build co-occurrence matrix P[i,j] counting pixel pairs at given offset → normalize → compute features (energy = Σ P²; entropy = -Σ P·log(P); contrast = Σ |i-j|²·P; etc.).
- **Complexity:** Medium (~300-400 lines)
- **Dependencies:** numpy
- **References:** Haralick, Shanmugam & Dinstein, "Textural Features for Image Classification", IEEE Trans. SMC, 3(6), 1973.

### T2-02. Non-Local Means Denoising
- **Category:** `noise` (denoising)
- **Type:** ImageTransform (BandwiseTransformMixin)
- **Description:** Denoises by averaging pixels with similar local neighborhoods across the entire image, exploiting non-local self-similarity. Superior to Gaussian smoothing for structured noise like SAR speckle.
- **Java source:** Fiji — `src/main/java/de/fzj/jungle/denoise/NonLocalMeansDenoise.java`
- **Repo:** `https://github.com/fiji/Non_Local_Means_Denoise`
- **Parameters:**
  - `sigma: Annotated[float, Range(1.0, 100.0)]` — Noise std dev estimate
  - `patch_radius: Annotated[int, Range(1, 7)]` — Half-size of comparison patches (default 3)
  - `search_radius: Annotated[int, Range(5, 31)]` — Half-size of search window (default 11)
  - `h: Annotated[float, Range(1.0, 200.0)]` — Filtering strength (default = sigma)
- **Algorithm:** For each pixel: compare its patch to patches in search window → weight = exp(-||patch_diff||² / h²) → output = weighted average. Efficient with integral images for patch distance.
- **Complexity:** Medium (~200-300 lines)
- **Dependencies:** numpy, scipy
- **References:** Buades, Coll & Morel, "A Non-Local Algorithm for Image Denoising", CVPR 2005.

### T2-03. Structure Tensor / Orientation Analysis
- **Category:** `analyze` (texture/feature)
- **Type:** ImageTransform (BandwiseTransformMixin)
- **Description:** Computes local structure tensor (gradient outer product smoothed by Gaussian) and extracts orientation, coherence, and energy. Characterizes local image anisotropy and dominant orientation. Critical for SAR polarimetry analysis.
- **Java source:** Fiji OrientationJ — `src/main/java/orientation/StructureTensor.java`
- **Repo:** `https://github.com/fiji/OrientationJ`
- **Parameters:**
  - `sigma: Annotated[float, Range(0.5, 20.0)]` — Gaussian window for tensor smoothing (default 2.0)
  - `output: Annotated[str, Options("orientation", "coherence", "energy", "all")]` — Output type
- **Algorithm:** Compute Ix, Iy gradients → form Ixx=Ix², Ixy=Ix·Iy, Iyy=Iy² → smooth each with Gaussian(σ) → eigendecompose 2×2 matrix analytically: λ1,λ2 → orientation = 0.5·atan2(2·Ixy, Ixx-Iyy), coherence = (λ1-λ2)/(λ1+λ2), energy = λ1+λ2.
- **Complexity:** Low-Medium (~120-180 lines)
- **Dependencies:** numpy, scipy.ndimage (gaussian_filter)
- **References:** Jahne, "Digital Image Processing", Springer, Chapter 13. Rezakhaniha et al., Biomechanics and Modeling in Mechanobiology, 11(3-4), 2012.

### T2-04. Frangi Vesselness / Tubeness Filter
- **Category:** `filters` (feature detection)
- **Type:** ImageTransform (BandwiseTransformMixin)
- **Description:** Derives vesselness measure from Hessian matrix eigenvalues. Enhances tubular structures (vessels, ridges, linear features like roads/rivers) while suppressing blobs and background. Multi-scale capable.
- **Java source:** `imagej-ops` — `src/main/java/net/imagej/ops/filter/tubeness/DefaultTubeness.java`, `src/main/java/net/imagej/ops/filter/hessian/`
- **Repo:** `https://github.com/imagej/imagej-ops`
- **Parameters:**
  - `sigmas: list[float]` — Array of Gaussian scales for multi-scale analysis (e.g., [1.0, 2.0, 4.0])
  - `alpha: Annotated[float, Range(0.1, 1.0)]` — Plate-vs-line sensitivity (default 0.5)
  - `beta: Annotated[float, Range(0.1, 1.0)]` — Blob sensitivity (default 0.5)
  - `black_ridges: Annotated[bool, Desc("Detect dark ridges on bright background")]` — (default True)
- **Algorithm:** For each sigma: compute Hessian (Ixx, Ixy, Iyy) using Gaussian 2nd derivatives → eigenvalues λ1, λ2 → vesselness = exp(-R_B²/(2β²)) · (1 - exp(-S²/(2c²))) where R_B = λ1/λ2, S = sqrt(λ1²+λ2²) → take max across scales.
- **Complexity:** Medium (~200-300 lines)
- **Dependencies:** numpy, scipy.ndimage (gaussian_filter)
- **References:** Frangi et al., "Multiscale Vessel Enhancement Filtering", MICCAI, Springer LNCS 1496, 1998.

### T2-05. Marker-Controlled Watershed (MorphoLibJ)
- **Category:** `segmentation`
- **Type:** ImageTransform
- **Description:** Watershed segmentation using provided markers to control region growing, preventing over-segmentation. Operates on gradient magnitude or distance transform images. Extends existing Watershed with marker support.
- **Java source:** MorphoLibJ — `src/main/java/inra/ijpb/watershed/MarkerControlledWatershedTransform2D.java`
- **Repo:** `https://github.com/ijpb/MorphoLibJ`
- **Parameters:**
  - `connectivity: Annotated[int, Options(4, 8)]` — (default 4)
  - `use_gradient: Annotated[bool, Desc("Compute gradient internally")]` — (default True)
- **Algorithm:** Priority-queue-based flooding from markers. Each marker defines a catchment basin. Pixels are processed in order of increasing gradient/intensity. When two basins meet → watershed line. Input: grayscale image + labeled marker image.
- **Complexity:** Medium (~300-400 lines)
- **Dependencies:** numpy, heapq (priority queue)
- **References:** Meyer, "Morphological Segmentation", J. Visual Communication and Image Representation, 1(1), 1990. Legland et al. (2016).

### T2-06. Morphological Reconstruction
- **Category:** `binary` (morphology)
- **Type:** ImageTransform
- **Description:** Geodesic reconstruction by dilation or erosion. Fundamental building block for h-maxima, h-minima, regional maxima/minima, fill holes, and many advanced morphological operations.
- **Java source:** MorphoLibJ — `src/main/java/inra/ijpb/morphology/geodrec/GeodesicReconstructionByDilation.java` and `GeodesicReconstructionByErosion.java`
- **Repo:** `https://github.com/ijpb/MorphoLibJ`
- **Parameters:**
  - `type: Annotated[str, Options("by_dilation", "by_erosion")]` — Reconstruction type
  - `connectivity: Annotated[int, Options(4, 8)]` — (default 4)
- **Algorithm:** Input: marker image and mask image. By dilation: iteratively dilate marker, take pointwise min with mask, until convergence. Efficient hybrid algorithm uses queue-based propagation: raster scan + anti-raster scan + FIFO queue for remaining changes.
- **Complexity:** Medium (~200-300 lines)
- **Dependencies:** numpy
- **References:** Vincent, "Morphological Grayscale Reconstruction in Image Analysis: Applications and Efficient Algorithms", IEEE Trans. Image Processing, 2(2), 1993.

### T2-07. Richardson-Lucy Deconvolution
- **Category:** `fft` (deconvolution)
- **Type:** ImageTransform (BandwiseTransformMixin)
- **Description:** Iterative maximum-likelihood deconvolution restoring images blurred by a known PSF. Each iteration multiplies current estimate by correction factor from observed/re-blurred ratio. Essential for sensor PSF correction.
- **Java source:** `imagej-ops` — `src/main/java/net/imagej/ops/deconvolve/RichardsonLucyC.java`
- **Repo:** `https://github.com/imagej/imagej-ops`
- **Parameters:**
  - `psf: np.ndarray` — Point spread function
  - `n_iterations: Annotated[int, Range(1, 200)]` — Iteration count (default 20)
  - `regularization: Annotated[float, Range(0.0, 0.1)]` — Tikhonov-Miller regularization (default 0.0)
  - `non_circulant: Annotated[bool, Desc("Edge handling mode")]` — (default False)
- **Algorithm:** Initialize estimate = input. Each iteration: `ratio = input / convolve(estimate, PSF)` → `estimate *= correlate(ratio, PSF)`. Optional acceleration: vector extrapolation. FFT-based convolution for speed.
- **Complexity:** Medium (~200-300 lines)
- **Dependencies:** numpy (np.fft), scipy
- **References:** Richardson (1972); Lucy (1974).

### T2-08. Wiener Filter Deconvolution
- **Category:** `fft` (deconvolution)
- **Type:** ImageTransform (BandwiseTransformMixin)
- **Description:** Frequency-domain deconvolution dividing image spectrum by PSF spectrum, regularized by noise-to-signal ratio to avoid noise amplification.
- **Java source:** `imagej-ops` — `src/main/java/net/imagej/ops/filter/ifft/` and deconvolution utilities
- **Repo:** `https://github.com/imagej/imagej-ops`
- **Parameters:**
  - `psf: np.ndarray` — Point spread function
  - `snr: Annotated[float, Range(0.001, 100.0)]` — Signal-to-noise ratio estimate (default 10.0)
  - `clip_negative: Annotated[bool, Desc("Clip negative values to zero")]` — (default True)
- **Algorithm:** `F_restored = conj(H) · F_image / (|H|² + 1/SNR)` where H = FFT(PSF). Single-step, non-iterative.
- **Complexity:** Low (~60-100 lines)
- **Dependencies:** numpy (np.fft)
- **References:** Wiener (1949); Gonzalez & Woods, "Digital Image Processing", Chapter 5.

### T2-09. Hough Transform (Lines and Circles)
- **Category:** `find_maxima` (feature detection)
- **Type:** ImageTransform
- **Description:** Detects lines and circles in edge images by mapping edge points to parameter space and finding peaks corresponding to geometric primitives. Useful for infrastructure detection (roads, buildings).
- **Java source:** Fiji — `src/main/java/Hough_Circle.java` (circles); ImageJ core for lines
- **Repo:** `https://github.com/fiji/Hough_Circle`
- **Parameters:**
  - `mode: Annotated[str, Options("lines", "circles")]` — Detection mode
  - `threshold: Annotated[float, Range(0.0, 1.0)]` — Accumulator threshold (fraction of max)
  - `min_radius: Annotated[int, Range(1, 500)]` — Min circle radius (circles mode)
  - `max_radius: Annotated[int, Range(5, 1000)]` — Max circle radius (circles mode)
  - `rho_resolution: Annotated[float, Range(0.1, 5.0)]` — Rho resolution for lines (default 1.0)
  - `theta_resolution: Annotated[float, Range(0.1, 10.0)]` — Theta resolution in degrees (default 1.0)
- **Algorithm:** Lines: map each edge pixel to sinusoidal curve in (ρ, θ) space → accumulate → find peaks. Circles: for each edge pixel and each radius, vote in (x₀, y₀) space.
- **Complexity:** Medium (~200-350 lines)
- **Dependencies:** numpy
- **References:** Duda & Hart (1972); Yuen et al. (1990).

### T2-10. Ridge Detection (Steger's Algorithm)
- **Category:** `edges` (feature detection)
- **Type:** ImageTransform (BandwiseTransformMixin)
- **Description:** Detects curvilinear structures (ridges, valleys, lines) using Hessian eigenvalues with sub-pixel position extraction. Superior to edge detection for line-like features (roads, rivers, coastlines).
- **Java source:** Fiji — `src/main/java/de/biomedical_imaging/ij/steger/LineDetector.java`
- **Repo:** `https://github.com/fiji/Ridge_Detection`
- **Parameters:**
  - `sigma: Annotated[float, Range(0.5, 10.0)]` — Line width / Gaussian scale
  - `lower_threshold: Annotated[float, Range(0.0, 255.0)]` — Hysteresis low threshold
  - `upper_threshold: Annotated[float, Range(0.0, 255.0)]` — Hysteresis high threshold
  - `min_line_length: Annotated[int, Range(0, 1000)]` — Minimum line length (pixels)
  - `darkline: Annotated[bool, Desc("Detect dark lines on bright background")]`
- **Algorithm:** Compute Hessian → eigenvalues/eigenvectors → sub-pixel localization via Taylor expansion along max curvature direction → hysteresis linking → junction resolution → length filtering.
- **Complexity:** Medium (~400-500 lines)
- **Dependencies:** numpy, scipy.ndimage
- **References:** Steger, "An Unbiased Detector of Curvilinear Structures", IEEE PAMI, 20(2), 1998.

### T2-11. Sliding Paraboloid Background
- **Category:** `background`
- **Type:** ImageTransform (BandwiseTransformMixin)
- **Description:** Background subtraction using sliding paraboloid algorithm. The default method in modern ImageJ (alternative to rolling ball). Slides a parabolic surface under/over the intensity profile in 4 directions.
- **Java source:** ImageJ core — `ij/plugin/filter/BackgroundSubtracter.java` (`slidingParaboloidFloatBackground` method)
- **Repo:** `https://github.com/imagej/ImageJ`
- **Parameters:**
  - `radius: Annotated[float, Range(1.0, 500.0)]` — Paraboloid radius (default 50.0)
  - `light_background: Annotated[bool, Desc("Light background mode")]` — (default False)
- **Algorithm:** 1D parabola fitting in 4 directions (horizontal, vertical, two diagonals). For each direction: slide parabola along the 1D profile, compute geometric envelope as the maximum of all parabola positions. Combine 4 directional results via pointwise min.
- **Complexity:** Medium (~200-280 lines)
- **Dependencies:** numpy
- **References:** Sternberg, "Biomedical Image Processing", IEEE Computer, 16(1), 1983.

### T2-12. Extended Min/Max and H-Minima/H-Maxima
- **Category:** `segmentation` (morphology)
- **Type:** ImageTransform
- **Description:** Finds extended (regional) minima/maxima suppressed to dynamic height h. H-minima suppresses shallow minima (preventing over-segmentation in watershed). Building block for robust segmentation.
- **Java source:** MorphoLibJ — `src/main/java/inra/ijpb/morphology/MinimaAndMaxima.java`, `src/main/java/inra/ijpb/morphology/extrema/RegionalExtremaByFlooding.java`
- **Repo:** `https://github.com/ijpb/MorphoLibJ`
- **Parameters:**
  - `h: Annotated[float, Range(0.1, 255.0)]` — Dynamic height threshold
  - `connectivity: Annotated[int, Options(4, 8)]` — (default 4)
  - `type: Annotated[str, Options("h_minima", "h_maxima", "extended_minima", "extended_maxima", "regional_minima", "regional_maxima")]`
- **Algorithm:** H-minima: `reconstruct_by_erosion(image + h, image)`. Extended minima: regional minima of h-minima result. Built on morphological reconstruction (T2-06).
- **Complexity:** Low-Medium (~100-150 lines, depends on T2-06)
- **Dependencies:** numpy; depends on Morphological Reconstruction (T2-06)
- **References:** Soille, "Morphological Image Analysis", Springer, 2nd ed., 2003.

### T2-13. FFT Custom Filter
- **Category:** `fft`
- **Type:** ImageTransform (BandwiseTransformMixin)
- **Description:** Applies user-defined frequency-domain filter mask via FFT. Allows arbitrary spatial frequency filtering beyond simple bandpass. Hanning window option to reduce edge artifacts.
- **Java source:** ImageJ core — `ij/plugin/FFT.java`, `ij/plugin/filter/FFTFilter.java`
- **Repo:** `https://github.com/imagej/ImageJ`
- **Parameters:**
  - `mask: np.ndarray` — Frequency-domain filter mask (same dimensions as image)
  - `window: Annotated[str, Options("none", "hanning", "hamming", "blackman")]` — Apodization window (default "hanning")
  - `pad_to_power_of_2: Annotated[bool, Desc("Zero-pad to power of 2")]` — (default True)
- **Algorithm:** Apply window → zero-pad → FFT → multiply by mask → IFFT → crop to original size. Builds on existing FFTBandpassFilter infrastructure.
- **Complexity:** Low (~60-100 lines)
- **Dependencies:** numpy (np.fft)
- **References:** ImageJ source code (public domain).

### T2-14. Template Matching (Normalized Cross-Correlation)
- **Category:** `fft` (registration)
- **Type:** ImageTransform
- **Description:** Locates template pattern within larger image using normalized cross-correlation (NCC). Returns correlation map where peaks indicate template locations. Useful for target detection and change detection.
- **Java source:** `imagej-ops` — `src/main/java/net/imagej/ops/filter/correlate/CorrelateFFTC.java`
- **Repo:** `https://github.com/imagej/imagej-ops`
- **Parameters:**
  - `template: np.ndarray` — 2D template array
  - `method: Annotated[str, Options("ncc", "zncc", "ssd")]` — Matching method (default "zncc")
  - `threshold: Annotated[float, Range(0.0, 1.0)]` — Peak detection threshold
- **Algorithm:** FFT-based cross-correlation. ZNCC: subtract means, divide by std devs. `R = IFFT(FFT(img) · conj(FFT(template)))` normalized by local statistics.
- **Complexity:** Low-Medium (~80-150 lines)
- **Dependencies:** numpy (np.fft)
- **References:** Lewis, "Fast Normalized Cross-Correlation", Vision Interface, 1995.

### T2-15. Morphological Gradient (External/Internal/Beucher)
- **Category:** `binary` (morphology)
- **Type:** ImageTransform (BandwiseTransformMixin)
- **Description:** Morphological gradient as difference between dilation and erosion. Internal gradient (original - erosion), external gradient (dilation - original), Beucher gradient (dilation - erosion).
- **Java source:** MorphoLibJ — `src/main/java/inra/ijpb/morphology/Morphology.java` (gradient methods)
- **Repo:** `https://github.com/ijpb/MorphoLibJ`
- **Parameters:**
  - `se_shape: Annotated[str, Options("disk", "square", "diamond")]` — Structuring element shape
  - `se_radius: Annotated[int, Range(1, 15)]` — Structuring element size (default 1)
  - `type: Annotated[str, Options("beucher", "internal", "external")]` — Gradient type
- **Algorithm:** Uses existing MorphologicalFilter for dilation/erosion. Beucher = dilate - erode. Internal = original - erode. External = dilate - original.
- **Complexity:** Low (~60-80 lines, wraps existing MorphologicalFilter)
- **Dependencies:** numpy; reuses existing `MorphologicalFilter`
- **References:** Serra, "Image Analysis and Mathematical Morphology", Academic Press, 1982.

### T2-16. Morphological Laplacian
- **Category:** `binary` (morphology)
- **Type:** ImageTransform (BandwiseTransformMixin)
- **Description:** Morphological analog of Laplacian: (dilation + erosion - 2 × original). Highlights rapid intensity changes using non-linear morphological operations.
- **Java source:** MorphoLibJ — `src/main/java/inra/ijpb/morphology/Morphology.java` (laplacian method)
- **Repo:** `https://github.com/ijpb/MorphoLibJ`
- **Parameters:**
  - `se_shape: Annotated[str, Options("disk", "square", "diamond")]`
  - `se_radius: Annotated[int, Range(1, 15)]` — (default 1)
- **Algorithm:** `output = dilate(image) + erode(image) - 2 × image`. Single formula.
- **Complexity:** Low (~40-60 lines, wraps existing MorphologicalFilter)
- **Dependencies:** numpy; reuses existing `MorphologicalFilter`
- **References:** Serra (1982); Soille (2003).

### T2-17. Directional Filtering (MorphoLibJ)
- **Category:** `binary` (morphology)
- **Type:** ImageTransform (BandwiseTransformMixin)
- **Description:** Morphological operations using oriented line structuring elements at multiple angles. Detects and enhances linear structures at specific orientations. Valuable for road/river detection.
- **Java source:** MorphoLibJ — `src/main/java/inra/ijpb/morphology/directional/DirectionalFilter.java`
- **Repo:** `https://github.com/ijpb/MorphoLibJ`
- **Parameters:**
  - `n_directions: Annotated[int, Range(4, 64)]` — Number of angles (default 12)
  - `line_length: Annotated[int, Range(3, 101)]` — SE length in pixels (default 15)
  - `operation: Annotated[str, Options("opening", "closing", "erosion", "dilation")]`
  - `combination: Annotated[str, Options("max", "mean", "median")]` — How to combine directional results
- **Algorithm:** For each angle θ: generate line SE → apply morphological operation → combine results across directions using max/mean/median.
- **Complexity:** Low-Medium (~120-180 lines)
- **Dependencies:** numpy; reuses existing `MorphologicalFilter`
- **References:** Soille, Breen & Jones, "Recursive Implementation of Erosions and Dilations Along Discrete Lines at Arbitrary Angles", IEEE PAMI, 18(5), 1996.

### T2-18. Kill Borders (MorphoLibJ)
- **Category:** `binary` (morphology)
- **Type:** ImageTransform
- **Description:** Removes all connected components touching the image border. Eliminates partial objects at image edges before measurement. Works on binary and labeled images.
- **Java source:** MorphoLibJ — uses `GeodesicReconstructionByDilation` internally
- **Repo:** `https://github.com/ijpb/MorphoLibJ`
- **Parameters:**
  - `connectivity: Annotated[int, Options(4, 8)]` — (default 8)
- **Algorithm:** Create border marker (copy border pixels, zero interior) → geodesic reconstruction by dilation → subtract from original.
- **Complexity:** Low (~50-80 lines, depends on T2-06)
- **Dependencies:** numpy; depends on Morphological Reconstruction (T2-06)
- **References:** Soille (2003).

### T2-19. Granulometry (MorphoLibJ)
- **Category:** `analyze` (morphology)
- **Type:** ImageTransform
- **Description:** Size distribution analysis by applying morphological openings with increasing SE size and measuring residual. Produces size distribution curve characterizing object sizes.
- **Java source:** MorphoLibJ — `src/main/java/inra/ijpb/measure/Granulometry.java`
- **Repo:** `https://github.com/ijpb/MorphoLibJ`
- **Parameters:**
  - `max_radius: Annotated[int, Range(1, 100)]` — Maximum SE radius
  - `step: Annotated[int, Range(1, 10)]` — Radius increment (default 1)
  - `se_shape: Annotated[str, Options("disk", "square")]` — (default "disk")
  - `type: Annotated[str, Options("opening", "closing")]` — (default "opening")
- **Algorithm:** For r in range(1, max_radius, step): apply morphological opening with SE of radius r → compute sum of residual. Output is the derivative of the volume curve (size distribution).
- **Complexity:** Low (~80-120 lines, uses existing MorphologicalFilter)
- **Dependencies:** numpy; reuses existing `MorphologicalFilter`
- **References:** Matheron, "Random Sets and Integral Geometry", Wiley, 1975.

### T2-20. ROF Total Variation Denoising
- **Category:** `noise` (denoising)
- **Type:** ImageTransform (BandwiseTransformMixin)
- **Description:** Rudin-Osher-Fatemi total variation regularization. Minimizes total variation while maintaining fidelity to noisy input. Piecewise-smooth results with sharp edges. Complementary to AnisotropicDiffusion.
- **Java source:** `imagej-ops` — variational methods in filter package
- **Repo:** `https://github.com/imagej/imagej-ops`
- **Parameters:**
  - `lambda_: Annotated[float, Range(0.01, 10.0)]` — Regularization weight (default 0.1)
  - `n_iterations: Annotated[int, Range(10, 500)]` — Gradient descent iterations (default 100)
  - `dt: Annotated[float, Range(0.01, 0.25)]` — Time step (default 0.125)
- **Algorithm:** Iterative PDE: `u_{n+1} = u_n + dt · (div(∇u/|∇u|) + λ(f - u))`. Similar structure to anisotropic diffusion (already ported) but uses TV norm. Divergence of normalized gradient.
- **Complexity:** Medium (~120-180 lines)
- **Dependencies:** numpy
- **References:** Rudin, Osher & Fatemi, "Nonlinear Total Variation Based Noise Removal Algorithms", Physica D, 60(1-4), 1992.

### T2-21. Color Deconvolution
- **Category:** `enhance` (spectral)
- **Type:** ImageTransform
- **Description:** Separates multi-stain/multi-spectral imagery into individual channel contributions using a known mixing matrix. Based on Beer-Lambert law. Directly applicable to spectral unmixing in remote sensing.
- **Java source:** Fiji — `src/main/java/sc/fiji/colourDeconvolution/Colour_Deconvolution.java`
- **Repo:** `https://github.com/fiji/Colour_Deconvolution`
- **Parameters:**
  - `stain_matrix: np.ndarray` — NxN mixing matrix (stain vectors as rows)
  - `stain_preset: Annotated[str, Options("custom", "H_E", "H_DAB", "FastRed_FastBlue")]` — Predefined matrices
  - `normalize: Annotated[bool, Desc("Normalize stain vectors")]` — (default True)
- **Algorithm:** Convert to OD space: `OD = -log10(I/I_0)`. Invert stain matrix: `M_inv = inv(M)`. Unmix: `channels = M_inv @ OD`. Convert back if needed.
- **Complexity:** Low (~80-120 lines)
- **Dependencies:** numpy (np.linalg.inv)
- **References:** Ruifrok & Johnston, "Quantification of histochemical staining by color deconvolution", Analytical and Quantitative Cytology and Histology, 23(4), 2001.

### T2-22. Tamura Texture Features
- **Category:** `analyze` (texture)
- **Type:** ImageTransform
- **Description:** Perceptually motivated texture features: coarseness, contrast, and directionality. Designed to correspond to human texture perception. Useful for texture-based retrieval and classification.
- **Java source:** `imagej-ops` — `src/main/java/net/imagej/ops/features/tamura/` (DefaultCoarseness, DefaultContrast, DefaultDirectionality)
- **Repo:** `https://github.com/imagej/imagej-ops`
- **Parameters:**
  - `n_scales: Annotated[int, Range(3, 8)]` — Number of scales for coarseness (default 5)
  - `histogram_bins: Annotated[int, Range(16, 128)]` — Bins for directionality (default 64)
- **Algorithm:** Coarseness: average difference at increasing window sizes, select optimal size per pixel. Contrast: kurtosis-based. Directionality: gradient orientation histogram entropy.
- **Complexity:** Low (~150-200 lines)
- **Dependencies:** numpy
- **References:** Tamura, Mori & Yamawaki, "Textural Features Corresponding to Visual Perception", IEEE Trans. SMC, 8(6), 1978.

---

## Tier 3 — Specialized

### T3-01. Level Sets (Active Contours)
- **Category:** `segmentation`
- **Type:** ImageTransform
- **Description:** Evolves contour (zero level set) to segment objects based on edge attraction and/or region statistics. Chan-Vese (region-based) and geodesic active contour (edge-based) models.
- **Java source:** Fiji — `src/main/java/levelsets/algorithm/ChanVese.java`
- **Repo:** `https://github.com/fiji/Level_Sets`
- **Parameters:**
  - `method: Annotated[str, Options("chan_vese", "geodesic")]`
  - `mu: Annotated[float, Range(0.0, 1.0)]` — Contour length regularization
  - `lambda1: Annotated[float, Range(0.1, 10.0)]` — Inside fitting weight
  - `lambda2: Annotated[float, Range(0.1, 10.0)]` — Outside fitting weight
  - `n_iterations: Annotated[int, Range(10, 1000)]` — Evolution steps
  - `dt: Annotated[float, Range(0.01, 0.5)]` — Time step
- **Algorithm:** PDE evolution with upwind schemes, curvature computation, optional reinitialization. Chan-Vese: minimize energy = length + λ1∫(f-c1)² + λ2∫(f-c2)² where c1,c2 are mean intensities inside/outside.
- **Complexity:** Medium (~300-400 lines)
- **References:** Chan & Vese (2001); Caselles, Kimmel & Sapiro (1997).

### T3-02. Morphological Segmentation (MorphoLibJ)
- **Category:** `segmentation`
- **Type:** ImageTransform
- **Description:** Combines morphological gradient with marker-controlled watershed. Auto-generates markers using extended minima of gradient, then applies watershed for clean segmentation.
- **Java source:** MorphoLibJ — `src/main/java/inra/ijpb/plugins/MorphologicalSegmentation.java`
- **Repo:** `https://github.com/ijpb/MorphoLibJ`
- **Parameters:**
  - `tolerance: Annotated[float, Range(0.1, 100.0)]` — Dynamic tolerance for extended minima
  - `connectivity: Annotated[int, Options(4, 8)]`
  - `gradient_radius: Annotated[int, Range(1, 10)]` — SE size for morphological gradient
- **Algorithm:** Compute morphological gradient → extended minima at tolerance → label connected components as markers → marker-controlled watershed on gradient.
- **Complexity:** Medium (~250-350 lines, orchestrates T2-05, T2-06, T2-12, T2-15)
- **Dependencies:** Depends on Morphological Reconstruction, Extended Min/Max, Marker-Controlled Watershed, Morphological Gradient
- **References:** Legland et al. (2016).

### T3-03. Seeded Region Growing
- **Category:** `segmentation`
- **Type:** ImageTransform
- **Description:** Grows regions from seed points by iteratively adding neighboring pixels satisfying intensity similarity criterion. Simple and effective for segmenting homogeneous regions.
- **Java source:** Fiji — `plugins/Seeds/Region_Growing.java`
- **Repo:** `https://github.com/fiji/fiji`
- **Parameters:**
  - `tolerance: Annotated[float, Range(0.1, 255.0)]` — Max intensity difference for inclusion
  - `connectivity: Annotated[int, Options(4, 8)]`
  - `max_iterations: Annotated[int, Range(1, 100000)]` — (default 100000)
- **Algorithm:** Queue-based flood fill with intensity criterion: for each seed, add neighbors whose |intensity - seed_mean| < tolerance. Update region mean as pixels are added.
- **Complexity:** Low (~100-150 lines)
- **References:** Adams & Bischof, "Seeded Region Growing", IEEE PAMI, 16(6), 1994.

### T3-04. FloodFiller
- **Category:** `segmentation`
- **Type:** ImageTransform
- **Description:** Flood fill with configurable 4/8-connectivity and tolerance-based color matching. Returns filled region. Used for magic-wand-like selection operations.
- **Java source:** ImageJ core — `ij/process/FloodFiller.java`
- **Repo:** `https://github.com/imagej/ImageJ`
- **Parameters:**
  - `seed_x: int` — Seed x coordinate
  - `seed_y: int` — Seed y coordinate
  - `tolerance: Annotated[float, Range(0.0, 255.0)]` — Intensity tolerance
  - `connectivity: Annotated[int, Options(4, 8)]`
  - `fill_value: Annotated[float, Desc("Value to fill with")]`
- **Algorithm:** Stack-based scanline flood fill. At each pixel: check if |value - seed_value| ≤ tolerance → fill and push neighbors.
- **Complexity:** Low (~80-120 lines)
- **References:** ImageJ source code (public domain).

### T3-05. EllipseFitter
- **Category:** `analyze`
- **Type:** ImageTransform
- **Description:** Fits ellipse to binary region using second-order central moments. Returns center, axes, angle, eccentricity. Useful for shape characterization of detected objects.
- **Java source:** ImageJ core — `ij/process/EllipseFitter.java`
- **Repo:** `https://github.com/imagej/ImageJ`
- **Parameters:**
  - `output: Annotated[str, Options("parameters", "overlay")]` — Output type
- **Algorithm:** Compute 2nd-order central moments (mu20, mu02, mu11) → eigenvalues of 2×2 moment matrix for axes → eigenvectors for orientation → eccentricity = sqrt(1 - minor²/major²).
- **Complexity:** Low (~80-120 lines)
- **Dependencies:** numpy (np.linalg.eigh)
- **References:** ImageJ source code (public domain).

### T3-06. Shape Descriptors Suite
- **Category:** `analyze`
- **Type:** ImageTransform
- **Description:** Comprehensive shape descriptors for labeled regions: circularity, solidity, aspect ratio, Feret diameters, convexity, roundness, elongation, inertia ellipse.
- **Java source:** MorphoLibJ — `src/main/java/inra/ijpb/measure/region2d/` (Circularity, MaxFeretDiameter, InertiaEllipse, etc.)
- **Repo:** `https://github.com/ijpb/MorphoLibJ`
- **Parameters:**
  - `descriptors: list[str]` — Which descriptors to compute
  - `pixel_size: Annotated[float, Desc("Physical pixel size")]` — For calibrated measurements
- **Algorithm:** Circularity = 4π·area/perimeter². Solidity = area/convex_hull_area. Feret: rotating calipers on convex hull. Aspect ratio from fitted ellipse.
- **Complexity:** Medium (~400-500 lines)
- **References:** Legland et al. (2016).

### T3-07. Image Statistics (Extended Measurements)
- **Category:** `analyze`
- **Type:** ImageTransform
- **Description:** Comprehensive statistics: mean, stddev, min, max, mode, median, skewness, kurtosis, histogram, centroid, spatial moments, Hu moments.
- **Java source:** ImageJ core — `ij/process/ImageStatistics.java`, `ij/process/ByteStatistics.java`, `FloatStatistics.java`
- **Repo:** `https://github.com/imagej/ImageJ`
- **Parameters:**
  - `measurements: list[str]` — Which statistics to compute
  - `n_bins: Annotated[int, Range(16, 65536)]` — Histogram bins
- **Algorithm:** Histogram-based computation. Spatial moments M00, M10, M01, M20, M02, M11 for centroid and orientation. Hu moments for shape invariants.
- **Complexity:** Medium (~300-400 lines)
- **References:** Hu, "Visual pattern recognition by moment invariants", IRE Trans. Information Theory, 8(2), 1962.

### T3-08. Image Correlator (Colocalization Metrics)
- **Category:** `analyze`
- **Type:** ImageTransform
- **Description:** Computes correlation metrics between two images: Pearson's r, Manders coefficients (M1, M2), overlap coefficient. Standard for colocalization analysis and change detection.
- **Java source:** ImageJ core — `ij/plugin/ImageCorrelator.java`
- **Repo:** `https://github.com/imagej/ImageJ`
- **Parameters:**
  - `metrics: list[str]` — ["pearson", "manders_m1", "manders_m2", "overlap", "costes_threshold"]
- **Algorithm:** Pearson's r = cov(A,B)/(std(A)·std(B)). Manders M1 = Σ(Aᵢ where Bᵢ>0)/Σ(Aᵢ). Overlap = Σ(A·B)/√(Σ(A²)·Σ(B²)).
- **Complexity:** Low (~80-120 lines)
- **Dependencies:** numpy (np.corrcoef)
- **References:** Manders et al. (1993).

### T3-09. FractalBoxCount
- **Category:** `analyze`
- **Type:** ImageTransform
- **Description:** Estimates fractal dimension of binary image using box-counting method. Characterizes spatial complexity and texture of patterns. Useful for terrain roughness analysis.
- **Java source:** ImageJ core — `ij/plugin/filter/FractalBoxCounter.java`
- **Repo:** `https://github.com/imagej/ImageJ`
- **Parameters:**
  - `box_sizes: list[int]` — Box sizes (default: [2, 3, 4, 6, 8, 12, 16, 32, 64])
- **Algorithm:** For each box size: tile image with grid → count boxes containing ≥1 foreground pixel. Fractal dimension = negative slope of log(count) vs log(1/box_size) regression. Returns D and R².
- **Complexity:** Low (~60-100 lines)
- **Dependencies:** numpy
- **References:** Standard box-counting fractal dimension.

### T3-10. Histogram (Multi-type)
- **Category:** `analyze`
- **Type:** ImageTransform
- **Description:** Computes histograms with configurable bins, range, and optional log scaling. ROI-masked computation. CDF. Supports 8/16/32-bit images.
- **Java source:** ImageJ core — `ij/process/ImageProcessor.java` (`getHistogram()`), `ij/process/AutoThresholder.java`
- **Repo:** `https://github.com/imagej/ImageJ`
- **Parameters:**
  - `n_bins: Annotated[int, Range(2, 65536)]` — Number of bins
  - `range_min: float` — Minimum value (auto if None)
  - `range_max: float` — Maximum value (auto if None)
  - `cumulative: Annotated[bool, Desc("Compute CDF")]` — (default False)
- **Algorithm:** `np.histogram` with appropriate binning matching ImageJ conventions per image type.
- **Complexity:** Low (~60-100 lines)
- **Dependencies:** numpy
- **References:** ImageJ source code (public domain).

### T3-11. Lookup Table Operations
- **Category:** `enhance`
- **Type:** ImageTransform
- **Description:** Applies, inverts, and manipulates LUTs for intensity remapping. Standard LUTs (fire, ice, spectrum) and custom LUT application. Bakes LUT into pixel values.
- **Java source:** ImageJ core — `ij/plugin/LutLoader.java`, `ij/process/LUT.java`, `ij/process/ImageProcessor.java` (`applyTable()`)
- **Repo:** `https://github.com/imagej/ImageJ`
- **Parameters:**
  - `lut_name: Annotated[str, Options("fire", "ice", "spectrum", "red", "green", "blue", "cyan", "magenta", "yellow", "grays", "custom")]`
  - `custom_lut: np.ndarray` — 256-entry array (for custom)
  - `invert: Annotated[bool, Desc("Invert LUT")]`
- **Algorithm:** `output[i] = lut[input[i]]` — NumPy fancy indexing. Predefined LUT arrays hardcoded.
- **Complexity:** Low (~80-120 lines)
- **Dependencies:** numpy
- **References:** ImageJ source code (public domain).

### T3-12. StackStatistics (Extended Projections)
- **Category:** `stacks`
- **Type:** ImageTransform
- **Description:** Per-pixel statistics across stack slices beyond ZProjection: coefficient of variation, skewness, kurtosis, percentile projections. Extends existing ZProjection.
- **Java source:** ImageJ core — `ij/process/StackStatistics.java`
- **Repo:** `https://github.com/imagej/ImageJ`
- **Parameters:**
  - `method: Annotated[str, Options("cv", "skewness", "kurtosis", "percentile")]`
  - `percentile: Annotated[float, Range(0.0, 100.0)]` — For percentile projection
  - `start_slice: int` — Start slice (default 0)
  - `end_slice: int` — End slice (default -1)
- **Algorithm:** Per-pixel computation along axis=0. CV = std/mean. Skewness/kurtosis via scipy.stats. Percentile via np.percentile.
- **Complexity:** Low (~60-100 lines)
- **Dependencies:** numpy, scipy.stats
- **References:** ImageJ source code (public domain).

### T3-13. SubstackMaker
- **Category:** `stacks`
- **Type:** ImageTransform
- **Description:** Extracts subsets of slices from stacks using range specs, step sizes, or explicit slice lists.
- **Java source:** ImageJ core — `ij/plugin/SubstackMaker.java`
- **Repo:** `https://github.com/imagej/ImageJ`
- **Parameters:**
  - `spec: str` — Range specification: "start-end-step" or comma-separated list
- **Algorithm:** Parse range spec → NumPy array slicing `[start:end:step]`.
- **Complexity:** Low (~40-60 lines)
- **Dependencies:** numpy
- **References:** ImageJ source code (public domain).

### T3-14. Stack Montage
- **Category:** `stacks`
- **Type:** ImageTransform
- **Description:** Arranges stack slices into NxM tiled montage with configurable border width, labels, and scale factor.
- **Java source:** ImageJ core — `ij/plugin/MontageMaker.java`
- **Repo:** `https://github.com/imagej/ImageJ`
- **Parameters:**
  - `columns: Annotated[int, Range(1, 50)]`
  - `rows: Annotated[int, Range(1, 50)]`
  - `scale: Annotated[float, Range(0.1, 2.0)]` — Scale factor (default 1.0)
  - `border_width: Annotated[int, Range(0, 10)]` — Border between tiles (default 0)
- **Algorithm:** Grid layout arithmetic + np.pad for borders + array placement.
- **Complexity:** Low (~60-100 lines)
- **Dependencies:** numpy
- **References:** ImageJ source code (public domain).

### T3-15. Stack Reslice
- **Category:** `stacks`
- **Type:** ImageTransform
- **Description:** Reslices 3D stack along arbitrary axis, producing orthogonal views (XZ, YZ) or oblique cross-sections.
- **Java source:** ImageJ core — `ij/plugin/Slicer.java`
- **Repo:** `https://github.com/imagej/ImageJ`
- **Parameters:**
  - `axis: Annotated[str, Options("XZ", "YZ", "oblique")]`
  - `interpolation: Annotated[str, Options("nearest", "bilinear")]`
  - `angle: float` — For oblique reslicing
- **Algorithm:** Orthogonal: axis transposition `np.swapaxes`. Oblique: interpolation along line normal using `scipy.ndimage.map_coordinates`.
- **Complexity:** Medium (~150-250 lines)
- **Dependencies:** numpy, scipy.ndimage
- **References:** ImageJ source code (public domain).

### T3-16. Chamfer Distance Transform
- **Category:** `binary`
- **Type:** ImageTransform
- **Description:** Distance transform using chamfer (weighted) masks. Various weight configurations (3-4, 5-7-11, Borgefors) for different accuracy/speed tradeoffs. Faster than Euclidean EDT.
- **Java source:** MorphoLibJ — `src/main/java/inra/ijpb/binary/distmap/ChamferDistanceTransform2DFloat.java`
- **Repo:** `https://github.com/ijpb/MorphoLibJ`
- **Parameters:**
  - `weights: Annotated[str, Options("borgefors_3_4", "chebyshev", "city_block", "quasi_euclidean_5_7_11")]`
  - `normalize: Annotated[bool, Desc("Normalize by weight")]` — (default True)
- **Algorithm:** Two-pass raster scan with weighted mask. Forward pass (top-left → bottom-right) + backward pass (bottom-right → top-left).
- **Complexity:** Low (~100-150 lines)
- **Dependencies:** numpy
- **References:** Borgefors, "Distance Transformations in Digital Images", CVGIP, 34(3), 1986.

### T3-17. Geodesic Distance Transform
- **Category:** `binary` (morphology)
- **Type:** ImageTransform
- **Description:** Distance from each foreground pixel to nearest marker, constrained to travel within foreground mask. Unlike Euclidean distance, respects image topology.
- **Java source:** MorphoLibJ — `src/main/java/inra/ijpb/binary/geodesic/GeodesicDistanceTransformFloat.java`
- **Repo:** `https://github.com/ijpb/MorphoLibJ`
- **Parameters:**
  - `weights: Annotated[str, Options("borgefors_3_4", "chebyshev", "city_block", "quasi_euclidean")]`
  - `normalize: Annotated[bool, Desc("Normalize by weight")]`
- **Algorithm:** Chamfer distance propagation constrained to binary mask. Two-pass raster scan. Input: binary mask + marker image.
- **Complexity:** Medium (~200-250 lines)
- **Dependencies:** numpy
- **References:** Soille (2003); Borgefors (1986).

### T3-18. Area Opening / Area Closing
- **Category:** `binary` (morphology)
- **Type:** ImageTransform
- **Description:** Removes connected components smaller than area threshold. Unlike standard opening with fixed SE, adapts to actual component geometry. Excellent for noise removal by size.
- **Java source:** MorphoLibJ — `src/main/java/inra/ijpb/morphology/attrfilt/AreaOpeningQueue.java`
- **Repo:** `https://github.com/ijpb/MorphoLibJ`
- **Parameters:**
  - `min_area: Annotated[int, Range(1, 10000)]` — Minimum area to retain (pixels)
  - `connectivity: Annotated[int, Options(4, 8)]`
  - `type: Annotated[str, Options("opening", "closing")]`
- **Algorithm:** Component tree (max-tree) construction or queue-based attribute filtering. Remove nodes with area < threshold.
- **Complexity:** Medium (~250-350 lines)
- **Dependencies:** numpy
- **References:** Vincent, "Morphological Area Openings and Closings for Grey-scale Images", NATO Shape in Picture Workshop, 1994.

### T3-19. Fill Holes (Morphological, Grayscale)
- **Category:** `binary` (morphology)
- **Type:** ImageTransform
- **Description:** Fills holes in binary or grayscale images using morphological reconstruction by erosion. For grayscale: fills regional minima not connected to border. Extends BinaryFillHoles (T1-13).
- **Java source:** MorphoLibJ — `src/main/java/inra/ijpb/morphology/reconstruct/ReconstructionByErosion2DGray8.java`
- **Repo:** `https://github.com/ijpb/MorphoLibJ`
- **Parameters:**
  - `connectivity: Annotated[int, Options(4, 8)]`
- **Algorithm:** Create border marker at max intensity → reconstruction by erosion from border → original - reconstructed gives filled holes.
- **Complexity:** Low (~40-60 lines, depends on T2-06)
- **Dependencies:** numpy; depends on Morphological Reconstruction (T2-06)
- **References:** Soille (2003); Vincent (1993).

### T3-20. Label Image Utilities
- **Category:** `analyze` (morphology)
- **Type:** Various utilities
- **Description:** Suite of operations for labeled images: replace labels, keep/remove largest, merge labels, dilate/erode labels, compute boundaries, measure region properties.
- **Java source:** MorphoLibJ — `src/main/java/inra/ijpb/label/LabelImages.java`, `src/main/java/inra/ijpb/measure/region2d/`
- **Repo:** `https://github.com/ijpb/MorphoLibJ`
- **Parameters:** Operation-specific.
- **Algorithm:** Collection of independent utilities. Each is individually simple but numerous.
- **Complexity:** Medium (~400-500 lines for full set)
- **References:** Legland et al. (2016).

### T3-21. Mean Shift Filter
- **Category:** `noise` (denoising/segmentation)
- **Type:** ImageTransform (BandwiseTransformMixin)
- **Description:** Iterative mode-seeking that shifts each pixel toward densest region of its local intensity neighborhood. Piecewise-constant output with sharp boundaries. Pre-segmentation step.
- **Java source:** `imagej-ops` — `src/main/java/net/imagej/ops/filter/meanshift/`
- **Repo:** `https://github.com/imagej/imagej-ops`
- **Parameters:**
  - `spatial_radius: Annotated[int, Range(1, 50)]` — Pixel neighborhood radius
  - `range_radius: Annotated[float, Range(1.0, 100.0)]` — Intensity bandwidth
  - `max_iterations: Annotated[int, Range(1, 100)]` — Convergence limit (default 20)
  - `convergence_threshold: Annotated[float, Range(0.01, 5.0)]` — Shift cutoff (default 0.5)
- **Algorithm:** For each pixel: iteratively shift position toward weighted mean of nearby pixels (spatial and range kernels) until convergence.
- **Complexity:** Medium (~150-250 lines)
- **References:** Comaniciu & Meer, "Mean Shift: A Robust Approach Toward Feature Space Analysis", IEEE PAMI, 24(5), 2002.

### T3-22. BinaryInterpolator
- **Category:** `binary`
- **Type:** ImageTransform
- **Description:** Morphologically interpolates between binary slices in a stack, generating smooth intermediate shapes by distance-map blending. For 3D reconstruction from sparse labels.
- **Java source:** ImageJ core — `ij/plugin/filter/Binary.java` (`interpolate()` method)
- **Repo:** `https://github.com/imagej/ImageJ`
- **Parameters:**
  - `n_intermediate: Annotated[int, Range(1, 20)]` — Number of intermediate slices
- **Algorithm:** Compute signed EDT on each binary slice → linearly interpolate distance maps between slices → threshold at zero.
- **Complexity:** Medium (~120-180 lines)
- **Dependencies:** numpy, scipy.ndimage (distance_transform_edt)
- **References:** ImageJ source code (public domain).

### T3-23. Spectral Unmixing (Linear)
- **Category:** `enhance` (spectral)
- **Type:** ImageTransform
- **Description:** Decomposes multi-spectral/hyperspectral imagery into fractional abundance maps using linear mixing model. Constrained least-squares at each pixel. Core remote sensing algorithm.
- **Java source:** `imagej-ops` — `src/main/java/net/imagej/ops/linalg/`
- **Repo:** `https://github.com/imagej/imagej-ops`
- **Parameters:**
  - `endmembers: np.ndarray` — NxM endmember spectra matrix
  - `method: Annotated[str, Options("unconstrained_ls", "nnls", "fully_constrained")]`
  - `regularization: Annotated[float, Range(0.0, 1.0)]` — Tikhonov lambda (default 0.0)
- **Algorithm:** Unconstrained: `x = (E^T E)^{-1} E^T y`. NNLS: scipy.optimize.nnls. Fully constrained: NNLS + sum-to-one constraint.
- **Complexity:** Medium (~150-250 lines)
- **Dependencies:** numpy, scipy.optimize
- **References:** Keshava & Mustard, "Spectral Unmixing", IEEE Signal Processing Magazine, 19(1), 2002.

### T3-24. MedianCut Color Quantization
- **Category:** `enhance` (color)
- **Type:** ImageTransform
- **Description:** Reduces number of colors using median-cut algorithm. Converts 24-bit color to optimized indexed palette. Optional Floyd-Steinberg dithering.
- **Java source:** ImageJ core — `ij/process/MedianCut.java`
- **Repo:** `https://github.com/imagej/ImageJ`
- **Parameters:**
  - `n_colors: Annotated[int, Range(2, 256)]` — Number of palette colors (default 256)
  - `dither: Annotated[bool, Desc("Apply Floyd-Steinberg dithering")]` — (default False)
- **Algorithm:** Recursively split RGB cube along axis of greatest range at median → produce N-color palette → map each pixel to nearest palette entry. Optional error diffusion dithering.
- **Complexity:** Medium (~200-280 lines)
- **Dependencies:** numpy
- **References:** Heckbert, "Color Image Quantization for Frame Buffer Display", SIGGRAPH, 1982.

### T3-25. PSF Generator
- **Category:** `fft` (deconvolution utility)
- **Type:** Utility function
- **Description:** Generates synthetic Point Spread Functions: Gaussian, Airy disk (diffraction-limited). For use with deconvolution algorithms (T2-07, T2-08).
- **Java source:** Fiji — `src/main/java/psf/GaussianPSF.java`, `src/main/java/psf/AiryPSF.java`
- **Repo:** `https://github.com/fiji/PSF_Generator`
- **Parameters:**
  - `type: Annotated[str, Options("gaussian", "airy")]`
  - `sigma: float` — For Gaussian
  - `wavelength: float` — For Airy (nm)
  - `numerical_aperture: float` — NA
  - `pixel_size: float` — nm/pixel
  - `size: tuple[int, int]` — Output dimensions
- **Algorithm:** Gaussian: analytical formula. Airy: `J₁(x)/x` pattern using scipy.special.j1.
- **Complexity:** Low-Medium (~100-180 lines)
- **Dependencies:** numpy, scipy.special (for Airy)
- **References:** Gibson & Lanni (1989).

### T3-26. Viscous Watershed
- **Category:** `segmentation`
- **Type:** ImageTransform
- **Description:** Watershed variant using viscous flooding simulation. Thicker fluid smooths relief, merging shallow basins to reduce over-segmentation.
- **Java source:** MorphoLibJ — `src/main/java/inra/ijpb/watershed/ViscousWatershed2D.java`
- **Repo:** `https://github.com/ijpb/MorphoLibJ`
- **Parameters:**
  - `viscosity: Annotated[float, Range(0.0, 100.0)]` — Fluid thickness (higher = fewer regions)
  - `connectivity: Annotated[int, Options(4, 8)]`
- **Algorithm:** Apply morphological operations to smooth the input relief based on viscosity → standard watershed on smoothed relief.
- **Complexity:** Medium (~250-350 lines)
- **References:** Vachier & Meyer, "The Viscous Watershed Transform", J. Mathematical Imaging and Vision, 22(2-3), 2005.

### T3-27. Lipschitz Background Leveling
- **Category:** `background`
- **Type:** ImageTransform (BandwiseTransformMixin)
- **Description:** Corrects uneven background by fitting Lipschitz lower/upper envelope. Maximum slope constraint produces smooth, physically plausible background model.
- **Java source:** Various Fiji implementations
- **Repo:** `https://github.com/fiji/fiji`
- **Parameters:**
  - `slope: Annotated[float, Range(0.1, 10.0)]` — Maximum Lipschitz slope
  - `top_hat: Annotated[bool, Desc("Subtract background (True) or return background (False)")]`
  - `direction: Annotated[str, Options("down", "up")]` — Lower/upper envelope
- **Algorithm:** Iterative envelope computation using morphological-like propagation with slope constraint.
- **Complexity:** Medium (~150-200 lines)
- **Dependencies:** numpy
- **References:** Lezoray & Grady, "Image Processing and Analysis with Graphs", CRC Press, 2012.

### T3-28. Line Profile (Plot Profile)
- **Category:** `analyze`
- **Type:** Utility
- **Description:** Extracts intensity profiles along arbitrary lines/polylines. Averaging across line width. Returns distance/intensity arrays.
- **Java source:** ImageJ core — `ij/gui/ProfilePlot.java`
- **Repo:** `https://github.com/imagej/ImageJ`
- **Parameters:**
  - `start: tuple[int, int]` — Start point
  - `end: tuple[int, int]` — End point
  - `line_width: Annotated[int, Range(1, 50)]` — Averaging width (default 1)
- **Algorithm:** Bresenham line walk or interpolated sampling. For width > 1: sample perpendicular to line and average. Uses `scipy.ndimage.map_coordinates` for subpixel.
- **Complexity:** Low (~80-120 lines)
- **Dependencies:** numpy, scipy.ndimage
- **References:** ImageJ source code (public domain).

### T3-29. Rotate Image
- **Category:** `stacks` (geometric)
- **Type:** ImageTransform (BandwiseTransformMixin)
- **Description:** Rotates image by arbitrary angle with configurable interpolation and optional canvas enlargement.
- **Java source:** ImageJ core — `ij/plugin/filter/Rotator.java`
- **Repo:** `https://github.com/imagej/ImageJ`
- **Parameters:**
  - `angle: Annotated[float, Range(-360.0, 360.0)]` — Rotation angle (degrees)
  - `interpolation: Annotated[str, Options("nearest", "bilinear", "bicubic")]`
  - `enlarge: Annotated[bool, Desc("Enlarge canvas to fit")]` — (default False)
  - `fill_value: Annotated[float, Desc("Background fill value")]` — (default 0.0)
- **Algorithm:** Affine rotation around image center. scipy.ndimage.rotate provides core; matching ImageJ's exact behavior is the work.
- **Complexity:** Low (~60-100 lines)
- **Dependencies:** scipy.ndimage
- **References:** ImageJ source code (public domain).

### T3-30. Resize Image
- **Category:** `stacks` (geometric)
- **Type:** ImageTransform (BandwiseTransformMixin)
- **Description:** Resizes images with bilinear/bicubic interpolation and area-averaging for downscale.
- **Java source:** ImageJ core — `ij/plugin/filter/Resizer.java`
- **Repo:** `https://github.com/imagej/ImageJ`
- **Parameters:**
  - `scale_x: Annotated[float, Range(0.01, 10.0)]` — X scale factor
  - `scale_y: Annotated[float, Range(0.01, 10.0)]` — Y scale factor
  - `interpolation: Annotated[str, Options("nearest", "bilinear", "bicubic", "area_average")]`
- **Algorithm:** scipy.ndimage.zoom or similar. Area-averaging for downscale is the specific ImageJ detail.
- **Complexity:** Low (~60-100 lines)
- **Dependencies:** scipy.ndimage
- **References:** ImageJ source code (public domain).

---

## Tier 4 — Advanced / High Effort

### T4-01. SIFT (Scale-Invariant Feature Transform)
- **Category:** `edges` (feature detection/registration)
- **Type:** ImageTransform
- **Description:** Scale- and rotation-invariant keypoint detection and description. Scale-space pyramid, DoG extrema, orientation histograms, 128-dim descriptors. Foundation for image matching and registration.
- **Java source:** Fiji MPICBG — `src/main/java/mpicbg/imagefeatures/FloatArray2DSIFT.java`
- **Repo:** `https://github.com/fiji/mpicbg`
- **Parameters:**
  - `n_octaves: Annotated[int, Range(1, 8)]` — Scale octaves (default 4)
  - `n_scales_per_octave: Annotated[int, Range(1, 6)]` — Scales within octave (default 3)
  - `initial_sigma: Annotated[float, Range(0.5, 3.0)]` — Initial blur (default 1.6)
  - `contrast_threshold: Annotated[float, Range(0.001, 0.1)]` — DoG response threshold
  - `edge_threshold: Annotated[float, Range(2.0, 20.0)]` — Edge rejection ratio (default 10.0)
- **Algorithm:** Multi-stage: Gaussian scale-space → DoG → keypoint localization with sub-pixel refinement → orientation assignment → 4×4×8 gradient histogram descriptor.
- **Complexity:** High (~800-1200 lines)
- **References:** Lowe, "Distinctive Image Features from Scale-Invariant Keypoints", IJCV, 60(2), 2004.

### T4-02. BM3D (Block-Matching and 3D Filtering)
- **Category:** `noise` (denoising)
- **Type:** ImageTransform (BandwiseTransformMixin)
- **Description:** State-of-the-art denoising: groups similar patches into 3D stacks, applies collaborative filtering in transform domain. Near-optimal denoising performance.
- **Java source:** Fiji — `src/main/java/de/fzj/jungle/denoise/BM3D.java`
- **Repo:** `https://github.com/fiji/BM3D`
- **Parameters:**
  - `sigma: Annotated[float, Range(1.0, 100.0)]` — Noise std dev
  - `block_size: Annotated[int, Options(4, 8, 16)]` — Patch size (default 8)
  - `search_window: Annotated[int, Range(11, 65)]` — Search area radius (default 39)
  - `n_similar: Annotated[int, Range(8, 64)]` — Max similar blocks per group (default 16)
- **Algorithm:** Two stages: (1) Hard thresholding — group similar blocks → 3D transform → threshold → inverse → aggregate. (2) Wiener — use stage 1 result as pilot, re-group → Wiener filter in 3D → inverse → aggregate.
- **Complexity:** High (~600-1000 lines)
- **References:** Dabov et al., "Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering", IEEE Trans. Image Processing, 16(8), 2007.

### T4-03. SIFT-based Registration (with RANSAC)
- **Category:** `fft` (registration)
- **Type:** ImageTransform
- **Description:** Uses SIFT matching with RANSAC outlier rejection to compute rigid, affine, or perspective transforms between image pairs.
- **Java source:** Fiji MPICBG — `src/main/java/mpicbg/trakem2/transform/`
- **Repo:** `https://github.com/fiji/mpicbg`
- **Parameters:**
  - `transform_type: Annotated[str, Options("rigid", "similarity", "affine", "perspective")]`
  - `max_epsilon: Annotated[float, Range(0.5, 50.0)]` — RANSAC inlier threshold (pixels)
  - `min_inlier_ratio: Annotated[float, Range(0.01, 0.5)]` — Min fraction of inliers
  - `n_iterations: Annotated[int, Range(100, 10000)]` — RANSAC iterations
- **Algorithm:** Depends on SIFT (T4-01) plus RANSAC model fitting for geometric transform estimation.
- **Complexity:** High (~400-600 lines, plus SIFT dependency)
- **References:** Saalfeld et al. (2010).

### T4-04. Graph Cut Segmentation
- **Category:** `segmentation`
- **Type:** ImageTransform
- **Description:** Min-cut/max-flow binary segmentation. Globally optimal partition given foreground/background seeds. Pixels as graph nodes, edges encode similarity.
- **Java source:** Fiji — `src/main/java/graphcut/GraphCut.java`
- **Repo:** `https://github.com/fiji/Graph_Cut`
- **Parameters:**
  - `lambda_: Annotated[float, Range(0.1, 100.0)]` — Smoothness weight
  - `sigma: Annotated[float, Range(1.0, 50.0)]` — Intensity similarity bandwidth
  - `connectivity: Annotated[int, Options(4, 8)]`
- **Algorithm:** Build graph: each pixel = node, edges to neighbors weighted by similarity. Source/sink edges from seed annotations. Solve max-flow via Boykov-Kolmogorov algorithm.
- **Complexity:** High (~400-600 lines for max-flow)
- **References:** Boykov & Jolly, "Interactive Graph Cuts for Optimal Boundary & Region Segmentation", ICCV, 2001.

### T4-05. MSER (Maximally Stable Extremal Regions)
- **Category:** `find_maxima` (feature detection)
- **Type:** ImageTransform
- **Description:** Detects blob-like regions stable across intensity thresholds. Component tree construction, stability analysis, region extraction. Complementary to DoG blob detection.
- **Java source:** Fiji MPICBG — `src/main/java/mpicbg/imagefeatures/FloatArray2DMSER.java`
- **Repo:** `https://github.com/fiji/mpicbg`
- **Parameters:**
  - `delta: Annotated[int, Range(1, 20)]` — Threshold step size (default 5)
  - `min_area: Annotated[float, Range(0.0001, 0.1)]` — Min region area (fraction of image)
  - `max_area: Annotated[float, Range(0.1, 1.0)]` — Max region area
  - `max_variation: Annotated[float, Range(0.1, 1.0)]` — Max area variation for stability
- **Algorithm:** Build component tree by sweeping thresholds → compute area stability = |area(t+δ) - area(t)| / area(t) → select regions with minimal variation.
- **Complexity:** High (~400-500 lines)
- **References:** Matas et al., "Robust Wide-Baseline Stereo from Maximally Stable Extremal Regions", Image and Vision Computing, 22(10), 2004.

### T4-06. bUnwarpJ (B-spline Elastic Registration)
- **Category:** `fft` (registration)
- **Type:** ImageTransform
- **Description:** Elastic non-rigid registration using B-spline deformations with consistency constraint. Symmetric forward/backward optimization.
- **Java source:** Fiji — `src/main/java/bunwarpj/Transformation.java`
- **Repo:** `https://github.com/fiji/bUnwarpJ`
- **Parameters:**
  - `initial_deformation: Annotated[str, Options("very_coarse", "coarse", "fine", "very_fine")]`
  - `final_deformation: Annotated[str, Options("very_coarse", "coarse", "fine", "very_fine")]`
  - `divergence_weight: Annotated[float, Range(0.0, 1.0)]`
  - `curl_weight: Annotated[float, Range(0.0, 1.0)]`
  - `consistency_weight: Annotated[float, Range(0.0, 10.0)]`
- **Algorithm:** Multi-resolution B-spline optimization with regularization (divergence, curl, landmark, image similarity). Alternates forward/backward transform optimization.
- **Complexity:** High (~1500-2000+ lines)
- **References:** Arganda-Carreras et al. (2006).

### T4-07. Trainable Segmentation Feature Stack
- **Category:** `analyze` (feature extraction)
- **Type:** ImageTransform
- **Description:** Extracts comprehensive multi-scale feature stack (Gaussian, derivatives, Hessian, structure tensor, Gabor, DoG, membrane, Laplacian, edges) for pixel classification. The portable feature extraction component.
- **Java source:** Fiji — `src/main/java/trainableSegmentation/FeatureStack.java`
- **Repo:** `https://github.com/fiji/Trainable_Segmentation`
- **Parameters:**
  - `min_sigma: Annotated[float, Range(0.5, 5.0)]` — Min scale (default 1.0)
  - `max_sigma: Annotated[float, Range(2.0, 32.0)]` — Max scale (default 16.0)
  - `features: list[str]` — ["gaussian", "hessian", "structure_tensor", "gabor", "dog", "laplacian", "edges", "membrane"]
- **Algorithm:** For each scale σ in geometric series from min to max: compute selected features → stack all into multi-channel feature image.
- **Complexity:** High (~500-700 lines, orchestrates many sub-filters)
- **References:** Arganda-Carreras et al. (2017).

### T4-08. FeatureJ Suite (Multi-scale Differential Geometry)
- **Category:** `filters` (feature detection)
- **Type:** ImageTransform
- **Description:** Gaussian derivatives at any order, Hessian eigenvalues, Laplacian of Gaussian, structure tensor eigenvalues at arbitrary scales. Unified multi-scale differential geometry toolkit.
- **Java source:** Various FeatureJ plugins in Fiji
- **Repo:** `https://github.com/imagej/imagej`
- **Parameters:**
  - `sigma: Annotated[float, Range(0.5, 20.0)]` — Gaussian scale
  - `operation: Annotated[str, Options("derivatives", "hessian", "laplacian", "structure")]`
  - `order: int` — Derivative order (for derivatives op)
  - `output: Annotated[str, Options("largest_eigenvalue", "smallest_eigenvalue", "magnitude", "orientation")]`
- **Algorithm:** Gaussian derivative computation via separable convolution with Gaussian derivative kernels. Eigendecomposition for Hessian/structure tensor.
- **Complexity:** Medium-High (~500+ lines for full suite)
- **References:** Lindeberg, "Scale-Space Theory in Computer Vision", Springer, 1994.

---

## Summary Statistics

| Tier | Count | Complexity Distribution |
|------|-------|------------------------|
| Tier 1 (High Value / Low Effort) | 20 | All Low |
| Tier 2 (High Value / Medium Effort) | 22 | 8 Low, 14 Medium |
| Tier 3 (Specialized) | 30 | 17 Low, 13 Medium |
| Tier 4 (Advanced) | 8 | All High |
| **Total** | **80** | **25 Low, 27 Medium, 8 High** |

## Category Distribution

| Category | Count | Key Candidates |
|----------|-------|---------------|
| Filters / Texture | 16 | Gabor, LBP, Kuwahara, DoG, Variance, Entropy, Shadows, Frangi |
| Binary / Morphology | 16 | MorphRecon, Gradient, Laplacian, Directional, Area Opening, Chamfer DT, Geodesic DT |
| Segmentation | 9 | MarkerWatershed, LevelSets, RegionGrowing, GraphCut, MorphSeg, Viscous |
| Analyze / Measure | 12 | GLCM, Tamura, ShapeDescriptors, FractalDim, Correlation, Statistics |
| Enhancement / Color | 8 | ColorSpace, WhiteBalance, ColorDeconv, SpectralUnmix, LUT, Quantization |
| Noise / Denoising | 6 | Bilateral, NLM, BM3D, MeanShift, TVDenoise, NoiseGen |
| FFT / Deconvolution | 7 | PhaseCorr, RL, Wiener, CustomFFT, Template, PSF |
| Feature Detection | 6 | Harris, Ridge, SIFT, Hough, MSER, Orientation |
| Stacks / Geometric | 8 | StackStats, Reslice, Montage, Rotate, Resize, Substack |
| Registration | 3 | PhaseCorrelation, SIFT+RANSAC, bUnwarpJ |

---

## Recommended Porting Order

**Phase 1 — Foundation (Tier 1, weeks 1-2):**
Start with T1-01 through T1-20. These are all Low complexity, directly useful, and
many build on existing infrastructure (GaussianBlur, Convolver, MorphologicalFilter).

**Phase 2 — Core Analytics (Tier 2, weeks 3-5):**
Priority order: T2-06 (MorphRecon — enables many others) → T2-01 (GLCM) → T2-02 (NLM) →
T2-03 (StructureTensor) → T2-04 (Frangi) → T2-05 (MarkerWatershed) → T2-07/08 (Deconv).

**Phase 3 — Specialization (Tier 3, weeks 6-8):**
Pick based on project needs. Segmentation chain (T3-01, T3-02, T3-03) or
morphology suite (T3-16–T3-19) are natural groups.

**Phase 4 — Advanced (Tier 4, as needed):**
T4-01 (SIFT) and T4-02 (BM3D) are the highest-impact. Only attempt when
Phase 1-2 are solid with full test coverage.
