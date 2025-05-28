# Phase 6 Validation Scripts

This directory contains validation and development scripts used during Phase 6 Model Preparation implementation.

## Files

### `simple_phase6_test.py`
- **Purpose:** Basic functionality validation for Phase 6 core requirements
- **Usage:** `python tests/validation/simple_phase6_test.py`
- **Tests:** 7 core functionality tests (data loading, feature compatibility, splitting, etc.)
- **Status:** Development/validation script

### `test_phase6_implementation.py`
- **Purpose:** Step 2 implementation validation with business metrics
- **Usage:** `python tests/validation/test_phase6_implementation.py`
- **Tests:** 6 implementation tests including model training and ROI calculations
- **Status:** Development/validation script

### `phase6_final_validation.py`
- **Purpose:** Comprehensive final validation before Phase 7 handoff
- **Usage:** `python tests/validation/phase6_final_validation.py`
- **Tests:** 5 comprehensive validation tests with performance benchmarking
- **Status:** Final validation script

## Note

These scripts were used during Phase 6 development and validation. The official test suite is located in:
- `tests/smoke/test_phase6_model_preparation_smoke.py`
- `tests/unit/test_phase6_model_preparation_critical.py`
- `tests/integration/test_phase6_model_preparation_integration.py`

Run the official test suite with: `python tests/run_phase6_tests.py`
