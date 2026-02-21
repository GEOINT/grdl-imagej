# grdl-imagej Development Guide

## Project Overview

Pure-NumPy reimplementations of 64 ImageJ/Fiji image processing algorithms,
packaged as a GRDL extension. All processors inherit from
`grdl.image_processing.base.ImageTransform`.

## Architecture

- **Base classes** come from `grdl` (dependency): `ImageTransform`, `BandwiseTransformMixin`
- **Versioning** via `@processor_version('x.y.z')` from `grdl.image_processing.versioning`
- **Tags** via `@processor_tags(modalities=[...], category=...)` from `grdl.image_processing.versioning`
- **Parameters** via `typing.Annotated` with `Range`, `Options`, `Desc` from `grdl.image_processing.params`
- **Categories** from `grdl.vocabulary.ProcessorCategory`

**Every processor in grdl-imagej must have all three metadata annotations** (`@processor_version`, `@processor_tags`, and `Annotated` parameter declarations). This metadata drives grdl-runtime catalog discovery and grdk widget UI generation. See the Processor Pattern below.

## Directory Layout

```
grdl_imagej/
├── __init__.py          # Barrel re-export of all 64 classes
├── _taxonomy.py         # Category → ImageJ menu label mapping
├── filters/             # RankFilters, UnsharpMask, GaussianBlur, Convolver,
│                        #   DifferenceOfGaussians, Shadows, Smooth, Sharpen,
│                        #   VarianceFilter, EntropyFilter, KuwaharaFilter,
│                        #   LocalBinaryPatterns, GaborFilterBank, FrangiVesselness
├── background/          # RollingBallBackground, PseudoFlatField, SlidingParaboloid
├── binary/              # MorphologicalFilter, DistanceTransform, Skeletonize,
│                        #   BinaryOutline, BinaryFillHoles, MorphologicalReconstruction,
│                        #   MorphologicalGradient, MorphologicalLaplacian,
│                        #   DirectionalFilter, KillBorders
├── enhance/             # CLAHE, GammaCorrection, ContrastEnhancer,
│                        #   ColorSpaceConverter, WhiteBalance, ColorDeconvolution
├── edges/               # EdgeDetector, HarrisCornerDetector, RidgeDetection
├── fft/                 # FFTBandpassFilter, PhaseCorrelation, RichardsonLucy,
│                        #   WienerFilter, FFTCustomFilter, TemplateMatching
├── find_maxima/         # FindMaxima, HoughTransform
├── math/                # ImageCalculator, MathOperations, TypeConverter
├── threshold/           # AutoThreshold, AutoLocalThreshold
├── segmentation/        # StatisticalRegionMerging, Watershed,
│                        #   MarkerControlledWatershed, ExtendedMinMax
├── stacks/              # ZProjection
├── analyze/             # AnalyzeParticles, GLCMHaralick, StructureTensor,
│                        #   Granulometry, TamuraTexture
└── noise/               # AnisotropicDiffusion, BilateralFilter, NoiseGenerator,
                         #   NonLocalMeans, ROFDenoise
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

All processors follow this template. The three metadata annotations are **required** — they make
the processor visible to grdl-runtime's catalog and configurable in grdk's widget UI:

```python
from typing import Annotated, Any
import numpy as np
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.image_processing.params import Range, Options, Desc
from grdl.vocabulary import ProcessorCategory as PC, ImageModality as IM

@processor_tags(modalities=[IM.PAN, IM.SAR, IM.MSI], category=PC.FILTERS)
@processor_version('1.54j')
class MyFilter(ImageTransform):
    # Annotated params → auto-generated __init__, introspectable by grdk
    param: Annotated[float, Range(min=0.1), Desc('...')] = 2.0

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        params = self._resolve_params(kwargs)  # merge + validate
        # implementation using params['param']
        return result
```

- `@processor_version`: stamps `__processor_version__` — bump when algorithm changes.
- `@processor_tags`: stamps `__processor_tags__` — modalities, category, description for catalog filtering.
- `Annotated[type, Range/Options, Desc]`: declares tunable parameters with constraints that are validated at init and at runtime via `_resolve_params()`.

### Catalog Registration

**Every new processor must be registered in `grdl_imagej/catalog/components.yaml`.**  This YAML catalog is the static manifest consumed by grdl-runtime's `YamlArtifactCatalog`. A processor that is not listed here will not appear in the runtime catalog or grdk UI.

When adding a new entry:

1. Increment `catalog_meta.next_id` and use the previous value as the new artifact's `id`.
2. Follow the existing entry format exactly — required fields: `id`, `name`, `version`, `artifact_type` (`grdl_processor`), `description`, `author`, `license`, `processor_class` (fully-qualified dotted path), `processor_version`, `processor_type` (`transform` | `detector` | `segmentation` | `analyzer`), `gpu_compatible`, `imagej_menu`, `tags` (with `modality` and `category` lists), and `parameters` (list of param dicts or `[]`).
3. Place the entry under the correct `imagej_menu` section header (add a new section if needed).
4. Update `tests/test_catalog.py`: add the new name to `TestAllComponentsPresent.EXPECTED_NAMES` and bump all `== 22` counts to match the new total.

## Testing

- Framework: **pytest**
- Fixtures in `tests/conftest.py`
- Benchmarks with `pytest-benchmark` (mark with `@pytest.mark.benchmark`)
- Run: `pytest tests/ -v -p no:napari --benchmark-disable`
- Coverage: `pytest tests/ -v -p no:napari --cov=grdl_imagej --cov-report=term-missing --benchmark-disable`
- Target: >75% line coverage

### Test Conventions

- All tests in a single file: `tests/test_imagej.py`
- One test class per processor: `TestProcessorName`
- Test algorithmic correctness, not implementation details
- Use synthetic images from fixtures (deterministic, seed=42)
- Verify: output shape, dtype, value ranges, edge cases, parameter validation
- Processors needing extra inputs (markers, PSF, template, stain_matrix, mask) receive them via `**kwargs`

## Dependency Management

### Source of Truth: `pyproject.toml`

**`pyproject.toml` is the single source of truth** for all dependencies. All package metadata, dependencies, and optional extras are defined here. This file drives PyPI publication and is read by build tools.

### Keeping Files in Sync

Three files must be kept synchronized:

| File | Purpose | How to Update |
|------|---------|---------------|
| `pyproject.toml` | **Source of truth** — package metadata, all dependencies, extras | Edit directly; this is the authoritative definition |
| `requirements.txt` (if it exists) | Development convenience — pinned versions for reproducible environments | `pip freeze > requirements.txt` after updating dependencies in `pyproject.toml` and installing |
| `.github/workflows/publish.yml` | PyPI publication — **DO NOT EDIT this file manually** (it extracts version from `pyproject.toml` automatically) | No action needed; the workflow reads `version` from `pyproject.toml` |

**Workflow:**
1. Update dependencies in `pyproject.toml` (add new packages, change versions, create/rename extras)
2. Install dependencies: `pip install -e ".[all,dev]"` (or appropriate extras for your work)
3. If `requirements.txt` exists in this project, regenerate it: `pip freeze > requirements.txt`
4. Commit both files
5. When creating a release, bump the `version` field in `pyproject.toml` (semantic versioning: `major.minor.patch`)
6. Create a git tag (e.g., `v0.2.0`) and push — the publish workflow triggers automatically

### Versioning for PyPI

- Versions follow **semantic versioning**: `major.minor.patch` (e.g., `0.1.0`, `1.2.3`)
- Update `version = "X.Y.Z"` in `pyproject.toml` before creating a release
- The publish workflow extracts the version automatically — no manual version extraction needed

## Dependencies

- `grdl` — base classes, versioning, parameter system, vocabulary
- `numpy` — array operations
- `scipy` — ndimage filters (background, binary, edges, filters, find_maxima, threshold)

## Git Practices

- Conventional commit messages
- One logical change per commit
- Never commit `.pyc`, `__pycache__/`, or IDE files
