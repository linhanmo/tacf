"""Preprocessing utilities: TSLib-compatible preprocessor for TACF.

Public entry points (imported eagerly; the CLI entry-point ``main`` is kept
lazy-available via ``src.preprocess.preprocess_tslib.main`` so that
``python -m src.preprocess.preprocess_tslib`` does not trigger a duplicate
sys.modules conflict).
"""
from .preprocess_tslib import (
    DATASET_CONFIG,
    DATASET_ROOT,
    OUTPUT_ROOT,
    get_borders_custom,
    get_borders_ett_hour,
    get_borders_ett_minute,
    process_single_dataset,
)

__all__ = [
    "DATASET_CONFIG",
    "DATASET_ROOT",
    "OUTPUT_ROOT",
    "get_borders_custom",
    "get_borders_ett_hour",
    "get_borders_ett_minute",
    "process_single_dataset",
]
