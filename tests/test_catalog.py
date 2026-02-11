# -*- coding: utf-8 -*-
"""
Tests for grdl-imagej Component Catalog.

Verifies the YAML catalog can be loaded by grdl-runtime's
YamlArtifactCatalog, that all 22 processors are present with
correct metadata, and that search/filter operations work.

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
"""

import pytest

from grdl_imagej.catalog import CATALOG_PATH
from grdl_rt.catalog import YamlArtifactCatalog, Artifact


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def catalog():
    """Catalog loaded from bundled components.yaml via YamlArtifactCatalog."""
    return YamlArtifactCatalog(file_path=CATALOG_PATH)


# ---------------------------------------------------------------------------
# Loading & basic structure
# ---------------------------------------------------------------------------

class TestCatalogLoading:

    def test_catalog_path_exists(self):
        assert CATALOG_PATH.exists()
        assert CATALOG_PATH.name == "components.yaml"

    def test_loads_successfully(self, catalog):
        artifacts = catalog.list_artifacts()
        assert len(artifacts) == 22

    def test_all_are_grdl_processor(self, catalog):
        artifacts = catalog.list_artifacts()
        assert all(a.artifact_type == "grdl_processor" for a in artifacts)

    def test_list_filters_by_type(self, catalog):
        processors = catalog.list_artifacts(artifact_type="grdl_processor")
        assert len(processors) == 22
        # No workflows in this catalog
        workflows = catalog.list_artifacts(artifact_type="grdk_workflow")
        assert len(workflows) == 0


# ---------------------------------------------------------------------------
# get_artifact() lookup
# ---------------------------------------------------------------------------

class TestGetArtifact:

    def test_get_by_name_and_version(self, catalog):
        a = catalog.get_artifact("gaussian-blur", "1.54j")
        assert a is not None
        assert a.name == "gaussian-blur"
        assert a.version == "1.54j"
        assert a.id == 1

    def test_get_clahe(self, catalog):
        a = catalog.get_artifact("clahe", "0.5.0")
        assert a is not None
        assert a.processor_class == "grdl_imagej.enhance.clahe.CLAHE"

    def test_get_nonexistent(self, catalog):
        assert catalog.get_artifact("nonexistent", "1.0") is None

    def test_get_wrong_version(self, catalog):
        assert catalog.get_artifact("gaussian-blur", "9.9.9") is None


# ---------------------------------------------------------------------------
# Artifact fields
# ---------------------------------------------------------------------------

class TestArtifactFields:

    def test_core_fields(self, catalog):
        a = catalog.get_artifact("gaussian-blur", "1.54j")
        assert a.artifact_type == "grdl_processor"
        assert a.processor_type == "transform"
        assert a.processor_class == "grdl_imagej.filters.gaussian_blur.GaussianBlur"
        assert "Gaussian" in a.description

    def test_detector_type(self, catalog):
        a = catalog.get_artifact("find-maxima", "1.54j")
        assert a.processor_type == "detector"

    def test_segmentation_type(self, catalog):
        a = catalog.get_artifact("statistical-region-merging", "1.0")
        assert a.processor_type == "segmentation"

    def test_analyzer_type(self, catalog):
        a = catalog.get_artifact("analyze-particles", "1.54j")
        assert a.processor_type == "analyzer"

    def test_tags_modalities(self, catalog):
        a = catalog.get_artifact("gaussian-blur", "1.54j")
        assert "SAR" in a.tags.get("modality", [])
        assert "PAN" in a.tags.get("modality", [])
        assert "EO" in a.tags.get("modality", [])

    def test_tags_category(self, catalog):
        a = catalog.get_artifact("gaussian-blur", "1.54j")
        assert "FILTERS" in a.tags.get("category", [])

    def test_author(self, catalog):
        a = catalog.get_artifact("gaussian-blur", "1.54j")
        assert "Rasband" in a.author

    def test_license(self, catalog):
        a = catalog.get_artifact("gaussian-blur", "1.54j")
        assert a.license == "MIT"

    def test_repr(self, catalog):
        a = catalog.get_artifact("gaussian-blur", "1.54j")
        r = repr(a)
        assert "gaussian-blur" in r
        assert "1.54j" in r


# ---------------------------------------------------------------------------
# search()
# ---------------------------------------------------------------------------

class TestSearch:

    def test_search_by_name(self, catalog):
        results = catalog.search("gaussian")
        assert any(a.name == "gaussian-blur" for a in results)

    def test_search_by_description(self, catalog):
        results = catalog.search("morphological")
        assert any(a.name == "morphological-filter" for a in results)

    def test_search_case_insensitive(self, catalog):
        results = catalog.search("GAUSSIAN")
        assert any(a.name == "gaussian-blur" for a in results)

    def test_search_no_results(self, catalog):
        results = catalog.search("xyznonexistent")
        assert len(results) == 0

    def test_search_partial(self, catalog):
        results = catalog.search("thresh")
        names = {a.name for a in results}
        assert "auto-threshold" in names
        assert "auto-local-threshold" in names


# ---------------------------------------------------------------------------
# search_by_tags()
# ---------------------------------------------------------------------------

class TestSearchByTags:

    def test_filter_by_modality(self, catalog):
        results = catalog.search_by_tags({"modality": "SAR"})
        assert len(results) == 22  # all support SAR

    def test_filter_by_category(self, catalog):
        results = catalog.search_by_tags({"category": "FILTERS"})
        names = {a.name for a in results}
        assert "gaussian-blur" in names
        assert "rank-filters" in names
        assert "unsharp-mask" in names
        assert "convolver" in names
        assert len(results) == 4

    def test_filter_by_multiple_tags(self, catalog):
        results = catalog.search_by_tags({
            "modality": "HSI",
            "category": "SEGMENTATION",
        })
        names = {a.name for a in results}
        assert "statistical-region-merging" in names

    def test_empty_tags(self, catalog):
        results = catalog.search_by_tags({})
        assert len(results) == 22

    def test_no_match(self, catalog):
        results = catalog.search_by_tags({"modality": "NONEXISTENT"})
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Mutation (add / remove) on a copy
# ---------------------------------------------------------------------------

class TestMutation:

    def test_add_and_remove(self, tmp_path, catalog):
        """Verify the catalog supports add/remove via YamlArtifactCatalog."""
        import shutil
        tmp_yaml = tmp_path / "components.yaml"
        shutil.copy(CATALOG_PATH, tmp_yaml)

        cat = YamlArtifactCatalog(file_path=tmp_yaml)
        assert len(cat.list_artifacts()) == 22

        new = Artifact(
            name="test-proc", version="1.0.0",
            artifact_type="grdl_processor",
            description="Test processor",
        )
        aid = cat.add_artifact(new)
        assert aid == 23
        assert len(cat.list_artifacts()) == 23

        removed = cat.remove_artifact("test-proc", "1.0.0")
        assert removed is True
        assert len(cat.list_artifacts()) == 22


# ---------------------------------------------------------------------------
# All 22 components present
# ---------------------------------------------------------------------------

class TestAllComponentsPresent:
    """Verify every expected processor appears in the catalog."""

    EXPECTED_NAMES = [
        "analyze-particles",
        "anisotropic-diffusion",
        "auto-local-threshold",
        "auto-threshold",
        "clahe",
        "contrast-enhancer",
        "convolver",
        "distance-transform",
        "edge-detector",
        "fft-bandpass-filter",
        "find-maxima",
        "gamma-correction",
        "gaussian-blur",
        "image-calculator",
        "morphological-filter",
        "rank-filters",
        "rolling-ball-background",
        "skeletonize",
        "statistical-region-merging",
        "unsharp-mask",
        "watershed",
        "z-projection",
    ]

    def test_all_present(self, catalog):
        artifacts = catalog.list_artifacts()
        names = [a.name for a in artifacts]
        for expected in self.EXPECTED_NAMES:
            assert expected in names, f"Missing component: {expected}"

    def test_count_matches(self, catalog):
        assert len(catalog.list_artifacts()) == len(self.EXPECTED_NAMES)

    def test_all_have_processor_class(self, catalog):
        for a in catalog.list_artifacts():
            assert a.processor_class.startswith("grdl_imagej.")

    def test_all_have_version(self, catalog):
        for a in catalog.list_artifacts():
            assert a.version, f"{a.name} missing version"

    def test_all_have_ids(self, catalog):
        for a in catalog.list_artifacts():
            assert a.id is not None and a.id > 0, f"{a.name} missing id"
