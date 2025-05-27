"""
Phase 3 Smoke Tests

Quick validation tests for Phase 3 data cleaning and preprocessing pipeline.
These tests provide fast feedback on basic functionality and critical path verification.

Smoke Test Philosophy: Quick verification, not exhaustive validation
Focus: Ensure Phase 3 pipeline works without errors and produces expected output format
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from preprocessing.data_cleaner import BankingDataCleaner
from preprocessing.validation import DataValidator
from preprocessing.feature_engineering import FeatureEngineer


class TestPhase3SmokeTests:
    """
    Smoke tests for Phase 3 pipeline components.

    Quick tests to ensure basic functionality works without errors.
    """

    def test_data_cleaner_initialization(self):
        """Smoke test: Data cleaner initializes without errors."""
        try:
            cleaner = BankingDataCleaner()
            initialization_success = True
        except Exception as e:
            initialization_success = False
            pytest.fail(f"Data cleaner initialization failed: {str(e)}")

        assert initialization_success, "Data cleaner should initialize successfully"
        assert hasattr(
            cleaner, "clean_banking_data"
        ), "Should have main cleaning method"

    def test_data_validator_initialization(self):
        """Smoke test: Data validator initializes without errors."""
        try:
            validator = DataValidator()
            initialization_success = True
        except Exception as e:
            initialization_success = False
            pytest.fail(f"Data validator initialization failed: {str(e)}")

        assert initialization_success, "Data validator should initialize successfully"
        assert hasattr(
            validator, "generate_validation_report"
        ), "Should have main validation method"

    def test_feature_engineer_initialization(self):
        """Smoke test: Feature engineer initializes without errors."""
        try:
            engineer = FeatureEngineer()
            initialization_success = True
        except Exception as e:
            initialization_success = False
            pytest.fail(f"Feature engineer initialization failed: {str(e)}")

        assert initialization_success, "Feature engineer should initialize successfully"
        assert hasattr(
            engineer, "engineer_features"
        ), "Should have main feature engineering method"

    def test_basic_data_cleaning_pipeline(self, phase3_raw_sample_data):
        """Smoke test: Basic data cleaning pipeline executes without errors."""
        cleaner = BankingDataCleaner()

        # Use small sample for quick testing
        sample_data = phase3_raw_sample_data.head(10).copy()

        try:
            cleaned_data = cleaner.clean_banking_data(sample_data)
            pipeline_success = True
        except Exception as e:
            pipeline_success = False
            pytest.fail(f"Basic cleaning pipeline failed: {str(e)}")

        assert pipeline_success, "Basic cleaning pipeline should execute successfully"
        assert cleaned_data is not None, "Should return cleaned data"
        assert len(cleaned_data) > 0, "Cleaned data should not be empty"
        assert len(cleaned_data) == len(sample_data), "Should preserve record count"

    def test_basic_data_validation(self, phase3_raw_sample_data):
        """Smoke test: Basic data validation executes without errors."""
        validator = DataValidator()

        # Use small sample for quick testing
        sample_data = phase3_raw_sample_data.head(10).copy()

        try:
            validation_report = validator.generate_validation_report(sample_data)
            validation_success = True
        except Exception as e:
            validation_success = False
            pytest.fail(f"Basic validation failed: {str(e)}")

        assert validation_success, "Basic validation should execute successfully"
        assert validation_report is not None, "Should return validation report"
        assert isinstance(
            validation_report, dict
        ), "Validation report should be a dictionary"

    def test_basic_feature_engineering(self, phase3_raw_sample_data):
        """Smoke test: Basic feature engineering executes without errors."""
        cleaner = BankingDataCleaner()
        engineer = FeatureEngineer()

        # Use small sample and clean it first
        sample_data = phase3_raw_sample_data.head(10).copy()
        cleaned_data = cleaner.clean_banking_data(sample_data)

        try:
            engineered_data = engineer.engineer_features(cleaned_data)
            engineering_success = True
        except Exception as e:
            engineering_success = False
            pytest.fail(f"Basic feature engineering failed: {str(e)}")

        assert (
            engineering_success
        ), "Basic feature engineering should execute successfully"
        assert engineered_data is not None, "Should return engineered data"
        assert len(engineered_data) > 0, "Engineered data should not be empty"
        assert len(engineered_data) == len(cleaned_data), "Should preserve record count"


class TestPhase3CriticalPathSmoke:
    """
    Smoke tests for critical path functionality.

    Tests the most important transformations identified in EDA.
    """

    def test_age_conversion_smoke(self):
        """Smoke test: Age conversion from text to numeric."""
        cleaner = BankingDataCleaner()

        # Minimal test data
        test_data = pd.DataFrame({"Age": ["25 years", "45 years", "67 years"]})

        try:
            cleaned_data = cleaner.clean_age_column(test_data)
            conversion_success = True
        except Exception as e:
            conversion_success = False
            pytest.fail(f"Age conversion failed: {str(e)}")

        assert conversion_success, "Age conversion should work"
        assert pd.api.types.is_numeric_dtype(
            cleaned_data["Age"]
        ), "Age should be numeric after conversion"

    def test_missing_value_handling_smoke(self):
        """Smoke test: Missing value handling."""
        cleaner = BankingDataCleaner()

        # Minimal test data with missing values
        test_data = pd.DataFrame(
            {
                "Housing Loan": ["yes", np.nan, "no"],
                "Personal Loan": [np.nan, "yes", "no"],
            }
        )

        try:
            cleaned_data = cleaner.handle_missing_values(test_data)
            handling_success = True
        except Exception as e:
            handling_success = False
            pytest.fail(f"Missing value handling failed: {str(e)}")

        assert handling_success, "Missing value handling should work"
        assert cleaned_data.isna().sum().sum() == 0, "Should eliminate missing values"

    def test_target_encoding_smoke(self):
        """Smoke test: Target variable encoding."""
        cleaner = BankingDataCleaner()

        # Minimal test data
        test_data = pd.DataFrame({"Subscription Status": ["yes", "no", "yes", "no"]})

        try:
            cleaned_data = cleaner.encode_target_variable(test_data)
            encoding_success = True
        except Exception as e:
            encoding_success = False
            pytest.fail(f"Target encoding failed: {str(e)}")

        assert encoding_success, "Target encoding should work"
        assert pd.api.types.is_numeric_dtype(
            cleaned_data["Subscription Status"]
        ), "Target should be numeric"
        assert set(cleaned_data["Subscription Status"].unique()).issubset(
            {0, 1}
        ), "Should be binary encoded"

    def test_contact_method_standardization_smoke(self):
        """Smoke test: Contact method standardization."""
        cleaner = BankingDataCleaner()

        # Minimal test data
        test_data = pd.DataFrame(
            {"Contact Method": ["Cell", "cellular", "Telephone", "telephone"]}
        )

        try:
            cleaned_data = cleaner.standardize_contact_methods(test_data)
            standardization_success = True
        except Exception as e:
            standardization_success = False
            pytest.fail(f"Contact method standardization failed: {str(e)}")

        assert standardization_success, "Contact method standardization should work"
        assert (
            len(cleaned_data["Contact Method"].unique()) <= 2
        ), "Should standardize to fewer unique values"


class TestPhase3OutputFormatSmoke:
    """
    Smoke tests for output format validation.

    Ensures output is in expected format for Phase 4.
    """

    def test_output_has_required_columns(self, phase3_raw_sample_data):
        """Smoke test: Output has required columns."""
        cleaner = BankingDataCleaner()
        engineer = FeatureEngineer()

        # Use small sample
        sample_data = phase3_raw_sample_data.head(5).copy()

        # Execute pipeline
        cleaned_data = cleaner.clean_banking_data(sample_data)
        final_data = engineer.engineer_features(cleaned_data)

        # Check for essential columns
        essential_columns = ["Client ID", "Age", "Subscription Status"]

        for col in essential_columns:
            assert (
                col in final_data.columns
            ), f"Essential column '{col}' should be present"

    def test_output_data_types_smoke(self, phase3_raw_sample_data):
        """Smoke test: Output has correct data types."""
        cleaner = BankingDataCleaner()

        # Use small sample
        sample_data = phase3_raw_sample_data.head(5).copy()

        # Execute cleaning
        cleaned_data = cleaner.clean_banking_data(sample_data)

        # Check key data types
        assert pd.api.types.is_numeric_dtype(
            cleaned_data["Age"]
        ), "Age should be numeric"
        assert pd.api.types.is_numeric_dtype(
            cleaned_data["Subscription Status"]
        ), "Target should be numeric"
        assert pd.api.types.is_numeric_dtype(
            cleaned_data["Campaign Calls"]
        ), "Campaign Calls should be numeric"

    def test_output_no_missing_values_smoke(self, phase3_raw_sample_data):
        """Smoke test: Output has no missing values."""
        cleaner = BankingDataCleaner()

        # Use small sample
        sample_data = phase3_raw_sample_data.head(5).copy()

        # Execute cleaning
        cleaned_data = cleaner.clean_banking_data(sample_data)

        # Check for missing values
        missing_count = cleaned_data.isna().sum().sum()
        assert (
            missing_count == 0
        ), f"Should have no missing values, found {missing_count}"

    def test_output_record_preservation_smoke(self, phase3_raw_sample_data):
        """Smoke test: Output preserves all records."""
        cleaner = BankingDataCleaner()
        engineer = FeatureEngineer()

        # Use small sample
        sample_data = phase3_raw_sample_data.head(5).copy()
        initial_count = len(sample_data)

        # Execute pipeline
        cleaned_data = cleaner.clean_banking_data(sample_data)
        final_data = engineer.engineer_features(cleaned_data)

        # Check record preservation
        assert (
            len(cleaned_data) == initial_count
        ), "Cleaning should preserve all records"
        assert (
            len(final_data) == initial_count
        ), "Feature engineering should preserve all records"


class TestPhase3PerformanceSmoke:
    """
    Smoke tests for performance validation.

    Quick checks to ensure reasonable performance.
    """

    def test_cleaning_performance_smoke(self, phase3_raw_sample_data):
        """Smoke test: Cleaning completes in reasonable time."""
        import time

        cleaner = BankingDataCleaner()

        # Use moderate sample size
        sample_data = phase3_raw_sample_data.head(50).copy()

        # Measure performance
        start_time = time.time()
        cleaned_data = cleaner.clean_banking_data(sample_data)
        end_time = time.time()

        processing_time = end_time - start_time

        # Should complete quickly for small dataset
        assert (
            processing_time < 5.0
        ), f"Cleaning should complete quickly, took {processing_time:.2f}s"
        assert cleaned_data is not None, "Should produce output"

    def test_validation_performance_smoke(self, phase3_raw_sample_data):
        """Smoke test: Validation completes in reasonable time."""
        import time

        validator = DataValidator()

        # Use moderate sample size
        sample_data = phase3_raw_sample_data.head(50).copy()

        # Measure performance
        start_time = time.time()
        validation_report = validator.generate_validation_report(sample_data)
        end_time = time.time()

        processing_time = end_time - start_time

        # Should complete quickly for small dataset
        assert (
            processing_time < 3.0
        ), f"Validation should complete quickly, took {processing_time:.2f}s"
        assert validation_report is not None, "Should produce validation report"

    def test_memory_usage_smoke(self, phase3_raw_sample_data):
        """Smoke test: Memory usage is reasonable."""
        cleaner = BankingDataCleaner()

        # Use moderate sample size
        sample_data = phase3_raw_sample_data.head(50).copy()

        # Execute cleaning
        cleaned_data = cleaner.clean_banking_data(sample_data)

        # Check memory usage
        memory_usage_mb = cleaned_data.memory_usage(deep=True).sum() / 1024**2

        # Should use reasonable memory for small dataset
        assert (
            memory_usage_mb < 10.0
        ), f"Memory usage should be reasonable, used {memory_usage_mb:.2f}MB"
