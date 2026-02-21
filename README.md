# grdl-imagej

Pure-NumPy reimplementations of 22 classic ImageJ/Fiji image processing algorithms,
built as an extension of the [GRDL](https://github.com/geoint-org/GRDL) framework.

Selected for relevance to remotely sensed imagery (PAN, MSI, HSI, SAR, thermal).
Each class inherits from `grdl.image_processing.base.ImageTransform`, carries
`@processor_tags` metadata for capability discovery, and declares `__gpu_compatible__`
for downstream GPU dispatch.

## Installation

```bash
pip install grdl-imagej
```

Or install from source:

```bash
git clone https://github.com/geoint-org/grdl-imagej.git
cd grdl-imagej
pip install -e ".[dev]"
```

## Dependencies

- **grdl** >= 0.1.0 (base classes, versioning, `Annotated` parameter system, vocabulary enums)
- **numpy** >= 1.20.0
- **scipy** >= 1.7.0

## Components (22 processors)

All processors inherit from `ImageTransform` and follow the same pattern:
instantiate with parameters, call `.apply(image)`. Every processor carries
`@processor_version` and `@processor_tags` metadata decorators, and declares
its tunable parameters via `Annotated` type hints with `Range`, `Options`,
and `Desc` constraint markers from `grdl.image_processing.params`. This
metadata enables grdl-runtime catalog discovery and grdk GUI auto-configuration.

### Process > Filters

| Class | Description |
|-------|-------------|
| `RankFilters` | Median, Min, Max, Mean, Variance, Despeckle |
| `UnsharpMask` | Gaussian-based sharpening |
| `GaussianBlur` | Isotropic/anisotropic Gaussian smoothing |
| `Convolver` | Arbitrary 2D kernel convolution |

### Process > Subtract Background

| Class | Description |
|-------|-------------|
| `RollingBallBackground` | Sternberg's rolling-ball background subtraction |

### Process > Binary

| Class | Description |
|-------|-------------|
| `MorphologicalFilter` | Erode, Dilate, Open, Close, TopHat, BlackHat, Gradient |
| `DistanceTransform` | Euclidean Distance Map (EDM) |
| `Skeletonize` | Zhang-Suen binary thinning |

### Process > Enhance Contrast

| Class | Description |
|-------|-------------|
| `CLAHE` | Contrast Limited Adaptive Histogram Equalization |
| `GammaCorrection` | Power-law intensity transform |
| `ContrastEnhancer` | Linear histogram stretching with saturation |

### Process > Find Edges

| Class | Description |
|-------|-------------|
| `EdgeDetector` | Sobel, Prewitt, Roberts, LoG, Scharr |

### Process > FFT

| Class | Description |
|-------|-------------|
| `FFTBandpassFilter` | Frequency-domain bandpass and stripe suppression |

### Process > Find Maxima

| Class | Description |
|-------|-------------|
| `FindMaxima` | Prominence-based peak/target detection |

### Process > Image Calculator

| Class | Description |
|-------|-------------|
| `ImageCalculator` | Pixel-wise arithmetic and logical operations |

### Image > Adjust > Threshold

| Class | Description |
|-------|-------------|
| `AutoThreshold` | Global thresholding (Otsu, Triangle, Huang, Li, Yen, etc.) |
| `AutoLocalThreshold` | Local thresholding (Bernsen, Niblack, Sauvola, Phansalkar, etc.) |

### Plugins > Segmentation

| Class | Description |
|-------|-------------|
| `StatisticalRegionMerging` | SRM region-based segmentation |
| `Watershed` | EDT-based watershed for splitting touching objects |

### Image > Stacks

| Class | Description |
|-------|-------------|
| `ZProjection` | Stack projection (max, mean, median, min, sum, std) |

### Analyze > Analyze Particles

| Class | Description |
|-------|-------------|
| `AnalyzeParticles` | Connected component analysis with measurements |

### Plugins > Anisotropic Diffusion

| Class | Description |
|-------|-------------|
| `AnisotropicDiffusion` | Perona-Malik edge-preserving smoothing |

## Quick Start

```python
from grdl_imagej import CLAHE, RankFilters, UnsharpMask

# Enhance contrast
clahe = CLAHE(block_size=127, n_bins=256, max_slope=3.0)
enhanced = clahe.apply(image)

# Denoise with median filter
median = RankFilters(method='median', radius=2)
denoised = median.apply(image)

# Sharpen
usm = UnsharpMask(sigma=2.0, weight=0.6)
sharpened = usm.apply(image)
```

### Pipeline Composition

```python
from grdl_imagej import GammaCorrection, UnsharpMask, EdgeDetector
from grdl.image_processing import Pipeline

pipe = Pipeline([
    GammaCorrection(gamma=0.5),
    UnsharpMask(sigma=2.0, weight=0.6),
    EdgeDetector(method='sobel'),
])
result = pipe.apply(image)
```

### Target Detection

```python
from grdl_imagej import FFTBandpassFilter, FindMaxima

# Remove background drift and high-frequency noise
bp = FFTBandpassFilter(filter_large=40, filter_small=3)
filtered = bp.apply(image)

# Find peaks
fm = FindMaxima(prominence=50.0)
peaks = fm.apply(filtered)  # (N, 2) array of [row, col]
```

### Segmentation

```python
from grdl_imagej import StatisticalRegionMerging, AutoThreshold

# Region segmentation
srm = StatisticalRegionMerging(Q=25)
labels = srm.apply(image)

# Binary thresholding
thresh = AutoThreshold(method='otsu')
binary = thresh.apply(image)
```

## Project Structure

```
grdl-imagej/
├── grdl_imagej/
│   ├── __init__.py              # Barrel export of all 22 components
│   ├── _taxonomy.py             # ProcessorCategory → ImageJ menu label mapping
│   ├── filters/                 # RankFilters, UnsharpMask, GaussianBlur, Convolver
│   ├── background/              # RollingBallBackground
│   ├── binary/                  # MorphologicalFilter, DistanceTransform, Skeletonize
│   ├── enhance/                 # CLAHE, GammaCorrection, ContrastEnhancer
│   ├── edges/                   # EdgeDetector
│   ├── fft/                     # FFTBandpassFilter
│   ├── find_maxima/             # FindMaxima
│   ├── math/                    # ImageCalculator
│   ├── threshold/               # AutoThreshold, AutoLocalThreshold
│   ├── segmentation/            # StatisticalRegionMerging, Watershed
│   ├── stacks/                  # ZProjection
│   ├── analyze/                 # AnalyzeParticles
│   └── noise/                   # AnisotropicDiffusion
└── tests/
    ├── conftest.py              # Shared fixtures
    ├── test_imagej.py           # Unit tests for all 22 components
    └── test_benchmarks.py       # Performance benchmarks
```

## Testing

```bash
# Run all tests
pytest tests/ -v --benchmark-disable

# Run with coverage
pytest tests/ -v --cov=grdl_imagej --cov-report=term-missing --benchmark-disable

# Run benchmarks only
pytest tests/test_benchmarks.py --benchmark-only
```

## Publishing to PyPI

### Dependency Management

All dependencies are defined in `pyproject.toml`. Keep these files synchronized:

- **`pyproject.toml`** — source of truth for versions and dependencies
- **`requirements.txt`** — regenerate with `pip freeze > requirements.txt` after updating `pyproject.toml`
- **`.github/workflows/publish.yml`** — automated PyPI publication (do not edit manually)

### Releasing a New Version

1. Update the `version` field in `pyproject.toml` (semantic versioning: `major.minor.patch`)
2. Update `requirements.txt` if dependencies changed: `pip install -e ".[all,dev]" && pip freeze > requirements.txt`
3. Commit both files
4. Create a git tag: `git tag v0.2.0` (matches version in `pyproject.toml`)
5. Push to GitHub: `git push && git push --tags`
6. Create a GitHub Release from the tag — this triggers the publish workflow automatically

The workflow:
- Builds wheels and source distribution using `python -m build`
- Publishes to PyPI with OIDC authentication (secure, no API keys)
- Artifacts are available at [pypi.org/p/grdl-imagej](https://pypi.org/p/grdl-imagej)

See [CLAUDE.md](CLAUDE.md#dependency-management) for detailed dependency management guidelines.

## Attribution

ImageJ is developed by Wayne Rasband at the U.S. National Institutes of Health.
ImageJ 1.x source code is in the public domain.

Fiji plugins (CLAHE, Auto Local Threshold, Statistical Region Merging,
Auto Threshold, Anisotropic Diffusion) are distributed under GPL-2.
This package provides independent reimplementations in NumPy, not
derivative works of the GPL source, but follows the same published
algorithms and cites the original authors.

## License

MIT License. See [LICENSE](LICENSE) for full text.
