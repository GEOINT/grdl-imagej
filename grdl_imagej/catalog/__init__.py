# -*- coding: utf-8 -*-
"""
Component Catalog - Static YAML catalog of grdl-imagej processors.

Ships a ``components.yaml`` file containing metadata for all grdl-imagej
image processing components. The format is fully compatible with
grdl-runtime's ``YamlArtifactCatalog`` and can be opened directly::

    from grdl_rt.catalog import YamlArtifactCatalog
    from grdl_imagej.catalog import CATALOG_PATH

    catalog = YamlArtifactCatalog(file_path=CATALOG_PATH)

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

from pathlib import Path

CATALOG_PATH: Path = Path(__file__).parent / "components.yaml"
"""Absolute path to the bundled ``components.yaml`` file."""

__all__ = ["CATALOG_PATH"]
