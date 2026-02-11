# grdl-imagej Development Guide

## Project Overview

Pure-NumPy reimplementations of 22 ImageJ/Fiji image processing algorithms,
packaged as a GRDL extension. All processors inherit from
`grdl.image_processing.base.ImageTransform`.

## Architecture

- **Base classes** come from `grdl` (dependency): `ImageTransform`, `BandwiseTransformMixin`
- **Versioning** via `@processor_version('x.y.z')` from `grdl.image_processing.versioning`
- **Tags** via `@processor_tags(modalities=[...], category=...)` from `grdl.image_processing.versioning`
- **Parameters** via `typing.Annotated` with `Range`, `Options`, `Desc` from `grdl.image_processing.params`
- **Categories** from `grdl.vocabulary.ProcessorCategory`

## Directory Layout

```
grdl_imagej/
├── __init__.py          # Barrel re-export of all 22 classes
├── _taxonomy.py         # Category → ImageJ menu label mapping
├── filters/             # RankFilters, UnsharpMask, GaussianBlur, Convolver
├── background/          # RollingBallBackground
├── binary/              # MorphologicalFilter, DistanceTransform, Skeletonize
├── enhance/             # CLAHE, GammaCorrection, ContrastEnhancer
├── edges/               # EdgeDetector
├── fft/                 # FFTBandpassFilter
├── find_maxima/         # FindMaxima
├── math/                # ImageCalculator
├── threshold/           # AutoThreshold, AutoLocalThreshold
├── segmentation/        # StatisticalRegionMerging, Watershed
├── stacks/              # ZProjection
├── analyze/             # AnalyzeParticles
└── noise/               # AnisotropicDiffusion
```

## Code Style

- **Line length**: 100 characters (black, ruff)
- **Python**: >= 3.11
- **Type hints** on all public methods
- **NumPy-style docstrings** (Parameters, Returns, Raises, Examples)
- **Imports**: fail-fast at module level (no lazy imports in hot paths)

### File Headers

Every Python file includes:

```python
# -*- coding: utf-8 -*-
"""
Title - one line.

Purpose and description.

Attribution
-----------
Original source/author info.

Dependencies
------------
List of imports with usage notes.

Author
------
Name

License
-------
MIT License
Copyright (c) 2024 geoint.org

Created
-------
YYYY-MM-DD

Modified
--------
YYYY-MM-DD
"""
```

### Processor Pattern

```python
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ProcessorCategory as PC, ImageModality as IM

@processor_tags(modalities=[IM.PAN, IM.SAR, IM.MSI], category=PC.FILTERS)
@processor_version('1.54j')
class MyFilter(ImageTransform):
    param: Annotated[float, Range(min=0.1), Desc('...')] = 2.0

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        params = self._resolve_params(kwargs)
        # implementation
        return result
```

## Testing

- Framework: **pytest**
- Fixtures in `tests/conftest.py`
- Benchmarks with `pytest-benchmark` (mark with `@pytest.mark.benchmark`)
- Run: `pytest tests/ -v --benchmark-disable`
- Coverage: `pytest tests/ -v --cov=grdl_imagej --cov-report=term-missing --benchmark-disable`
- Target: >75% line coverage

### Test Conventions

- One test class per processor: `TestProcessorName`
- Test algorithmic correctness, not implementation details
- Use synthetic images from fixtures (deterministic, seed=42)
- Verify: output shape, dtype, value ranges, edge cases, parameter validation

## Dependencies

- `grdl` — base classes, versioning, parameter system, vocabulary
- `numpy` — array operations
- `scipy` — ndimage filters (background, binary, edges, filters, find_maxima, threshold)

## Git Practices

- Conventional commit messages
- One logical change per commit
- Never commit `.pyc`, `__pycache__/`, or IDE files
