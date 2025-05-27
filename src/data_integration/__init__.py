"""
Phase 4 Data Integration Module

This module provides comprehensive data integration and validation functionality
for the banking marketing dataset, focusing on Phase 3 → Phase 4 data flow continuity.

Components:
- CSVLoader: Efficient CSV-based data loading with validation
- DataValidator: Comprehensive data integrity and quality validation
- PipelineUtils: Pipeline integration utilities for ML workflow
- Data Access Functions: High-level data access and validation functions

Key Features:
- Maintains 100% data quality score from Phase 3
- Supports >97K records/second processing performance
- Provides comprehensive error handling and validation
- Ensures seamless Phase 3 → Phase 4 → Phase 5 data flow continuity
"""

from .csv_loader import CSVLoader
from .data_validator import DataValidator
from .pipeline_utils import PipelineUtils
from .data_access import (
    load_and_validate_data,
    prepare_data_for_ml,
    validate_phase3_continuity,
    get_data_summary,
    quick_data_check,
    load_sample_data,
    load_phase3_output,
    validate_phase3_output,
    prepare_ml_pipeline,
)

__version__ = "1.0.0"
__author__ = "AI-Vive Banking Team"

# Module-level constants
PHASE3_OUTPUT_PATH = "data/processed/cleaned-db.csv"
EXPECTED_RECORD_COUNT = 41188
EXPECTED_FEATURE_COUNT = 33
PERFORMANCE_STANDARD = 97000  # records per second

# Quality metrics thresholds
QUALITY_SCORE_THRESHOLD = 100
MISSING_VALUES_THRESHOLD = 0
AGE_MIN = 18
AGE_MAX = 100

__all__ = [
    "CSVLoader",
    "DataValidator",
    "PipelineUtils",
    "load_and_validate_data",
    "prepare_data_for_ml",
    "validate_phase3_continuity",
    "get_data_summary",
    "quick_data_check",
    "load_sample_data",
    "load_phase3_output",
    "validate_phase3_output",
    "prepare_ml_pipeline",
    "PHASE3_OUTPUT_PATH",
    "EXPECTED_RECORD_COUNT",
    "EXPECTED_FEATURE_COUNT",
    "PERFORMANCE_STANDARD",
    "QUALITY_SCORE_THRESHOLD",
    "MISSING_VALUES_THRESHOLD",
    "AGE_MIN",
    "AGE_MAX",
]
