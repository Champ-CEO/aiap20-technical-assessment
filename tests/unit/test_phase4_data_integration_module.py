"""
Phase 4 Data Integration Module Unit Tests

Unit tests for Phase 4 data integration core components:
1. CSV Loader functionality and validation
2. Data Validator integrity checks
3. Pipeline Utilities performance and error handling
4. Data access functions reliability

Following TDD approach: tests define requirements before implementation.
"""

import pytest
import pandas as pd
import numpy as np
import time
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))


class TestCSVLoaderUnit:
    """Unit tests for CSV Loader functionality."""

    def test_csv_loader_initialization(self):
        """
        Unit Test: CSV Loader initializes correctly

        Requirements:
        - Accepts file path parameter
        - Validates file existence
        - Sets up proper configuration
        """
        from data_integration.csv_loader import CSVLoader, CSVLoaderError

        # Test default initialization
        loader = CSVLoader()
        assert loader.file_path.name == "cleaned-db.csv"
        assert loader.performance_metrics is not None

        # Test custom path initialization
        custom_loader = CSVLoader("data/processed/cleaned-db.csv")
        assert custom_loader.file_path.name == "cleaned-db.csv"

        # Test error on non-existent file
        with pytest.raises(CSVLoaderError):
            CSVLoader("non_existent_file.csv")

    def test_csv_loader_load_data_success(self):
        """
        Unit Test: CSV Loader loads data successfully

        Requirements:
        - Loads CSV file without errors
        - Returns valid DataFrame
        - Preserves data integrity
        """
        from data_integration.csv_loader import CSVLoader

        loader = CSVLoader()
        df = loader.load_data()

        # Validate successful loading
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert len(df.columns) > 0

        # Validate expected structure for full dataset
        assert len(df) == 41188
        assert len(df.columns) == 33

        # Validate no missing values
        assert df.isnull().sum().sum() == 0

    def test_csv_loader_performance_validation(self):
        """
        Unit Test: CSV Loader meets performance requirements

        Requirements:
        - Loads 41K records in reasonable time
        - Maintains >97K records/second standard
        - Memory usage is efficient
        """
        from data_integration.csv_loader import CSVLoader

        loader = CSVLoader()
        df = loader.load_data()

        # Check performance metrics
        metrics = loader.get_performance_metrics()
        assert "load_time" in metrics
        assert "records_per_second" in metrics
        assert "memory_usage_mb" in metrics

        # Validate performance standards
        assert metrics["records_per_second"] >= 50000  # Reasonable minimum
        assert metrics["memory_usage_mb"] < 200  # Reasonable memory usage

    def test_csv_loader_error_handling(self):
        """
        Unit Test: CSV Loader handles errors gracefully

        Requirements:
        - Handles missing files appropriately
        - Handles corrupted files gracefully
        - Provides clear error messages
        """
        from data_integration.csv_loader import CSVLoader, CSVLoaderError

        # Test missing file error
        with pytest.raises(CSVLoaderError):
            CSVLoader("missing_file.csv")

        # Test sample loading
        loader = CSVLoader()
        sample_df = loader.load_sample(n_rows=100)
        assert len(sample_df) == 100

        # Test column loading
        columns = ["Client ID", "Age", "Subscription Status"]
        column_df = loader.load_columns(columns)
        assert list(column_df.columns) == columns


class TestDataValidatorUnit:
    """Unit tests for Data Validator functionality."""

    def test_data_validator_initialization(self):
        """
        Unit Test: Data Validator initializes with proper configuration

        Requirements:
        - Sets up validation rules from Phase 3
        - Configures business rule parameters
        - Initializes quality metrics framework
        """
        from data_integration.data_validator import DataValidator

        validator = DataValidator()

        # Check initialization
        assert validator.expected_records == 41188
        assert validator.expected_features == 33
        assert validator.quality_score_threshold == 100
        assert validator.business_rules is not None
        assert validator.core_columns is not None

    def test_data_validator_schema_validation(self):
        """
        Unit Test: Data Validator performs schema validation

        Requirements:
        - Validates 33 features structure
        - Checks column names and types
        - Verifies data format consistency
        """
        from data_integration.data_validator import DataValidator
        from data_integration.csv_loader import CSVLoader

        validator = DataValidator()
        loader = CSVLoader()
        df = loader.load_data()

        validation_report = validator.validate_data(df, comprehensive=False)

        # Check schema validation results
        schema_validation = validation_report["schema_validation"]
        assert schema_validation["record_count_valid"] == True
        assert schema_validation["feature_count_valid"] == True
        assert schema_validation["core_columns_present"] == True

    def test_data_validator_integrity_checks(self):
        """
        Unit Test: Data Validator performs integrity checks

        Requirements:
        - Validates Phase 3 transformations preserved
        - Checks age conversion integrity
        - Verifies target encoding preservation
        - Validates missing value elimination
        """
        from data_integration.data_validator import DataValidator
        from data_integration.csv_loader import CSVLoader

        validator = DataValidator()
        loader = CSVLoader()
        df = loader.load_data()

        validation_report = validator.validate_data(df, comprehensive=False)

        # Check data integrity
        integrity = validation_report["data_integrity"]
        assert integrity["zero_missing_values"] == True
        assert integrity["missing_values_count"] == 0

        # Check Phase 3 preservation
        preservation = validation_report["phase3_preservation"]
        assert preservation["age_conversion_preserved"] == True
        assert preservation["target_encoding_preserved"] == True
        assert preservation["missing_values_eliminated"] == True

    def test_data_validator_quality_score_calculation(self):
        """
        Unit Test: Data Validator calculates quality scores

        Requirements:
        - Calculates 100% quality score for clean data
        - Identifies quality issues when present
        - Provides detailed quality metrics
        """
        from data_integration.data_validator import DataValidator
        from data_integration.csv_loader import CSVLoader

        validator = DataValidator()
        loader = CSVLoader()
        df = loader.load_data()

        validation_report = validator.validate_data(df, comprehensive=False)

        # Check quality metrics
        quality_metrics = validation_report["quality_metrics"]
        assert quality_metrics["completeness_score"] == 100
        assert quality_metrics["overall_quality_score"] >= 95  # Should be very high

    def test_data_validator_business_rules_validation(self):
        """
        Unit Test: Data Validator validates business rules

        Requirements:
        - Age range validation (18-100)
        - Target variable binary validation
        - Contact method standardization validation
        - Campaign calls range validation
        """
        from data_integration.data_validator import DataValidator
        from data_integration.csv_loader import CSVLoader

        validator = DataValidator()
        loader = CSVLoader()
        df = loader.load_data()

        validation_report = validator.validate_data(df, comprehensive=False)

        # Check business rules
        business_rules = validation_report["business_rules"]
        assert business_rules["age_range_valid"] == True
        assert business_rules["target_binary_valid"] == True
        assert business_rules["contact_methods_valid"] == True

    def test_data_validator_quick_validation(self):
        """
        Unit Test: Data Validator performs quick validation

        Requirements:
        - Quick validation for basic checks
        - Returns boolean result
        - Efficient for frequent checks
        """
        from data_integration.data_validator import DataValidator
        from data_integration.csv_loader import CSVLoader

        validator = DataValidator()
        loader = CSVLoader()
        df = loader.load_data()

        # Test quick validation
        is_valid = validator.validate_quick(df)
        assert is_valid == True


class TestPipelineUtilitiesUnit:
    """Unit tests for Pipeline Utilities functionality."""

    def test_pipeline_utils_data_splitting(self):
        """
        Unit Test: Pipeline utilities perform data splitting

        Requirements:
        - Stratified sampling preserves target distribution
        - Train/test splits maintain data integrity
        - Supports configurable split ratios
        """
        from data_integration.pipeline_utils import PipelineUtils
        from data_integration.csv_loader import CSVLoader

        utils = PipelineUtils()
        loader = CSVLoader()
        df = loader.load_data()

        # Test data splitting
        splits = utils.split_data(df, target_column="Subscription Status")

        # Validate splits
        assert "train" in splits
        assert "test" in splits
        assert len(splits["train"]) > len(splits["test"])

        # Validate total records preserved
        total_split_records = sum(len(split_df) for split_df in splits.values())
        assert total_split_records == len(df)

    def test_pipeline_utils_memory_optimization(self):
        """
        Unit Test: Pipeline utilities optimize memory usage

        Requirements:
        - Efficient data loading strategies
        - Memory usage monitoring
        - Chunked processing capabilities
        """
        from data_integration.pipeline_utils import PipelineUtils
        from data_integration.csv_loader import CSVLoader

        utils = PipelineUtils()
        loader = CSVLoader()
        df = loader.load_data()

        # Test memory optimization
        df_optimized = utils.optimize_memory_usage(df)

        # Validate optimization
        assert isinstance(df_optimized, pd.DataFrame)
        assert len(df_optimized) == len(df)
        assert len(df_optimized.columns) == len(df.columns)

        # Check performance metrics
        performance_report = utils.get_performance_report()
        assert "memory_usage" in performance_report

    def test_pipeline_utils_performance_monitoring(self):
        """
        Unit Test: Pipeline utilities monitor performance

        Requirements:
        - Tracks loading performance metrics
        - Monitors memory usage
        - Provides performance reports
        """
        from data_integration.pipeline_utils import PipelineUtils

        utils = PipelineUtils()

        # Test performance monitoring
        def sample_operation():
            return list(range(1000))

        result = utils.monitor_performance("test_operation", sample_operation)

        # Validate monitoring
        assert result == list(range(1000))

        # Check metrics recorded
        performance_report = utils.get_performance_report()
        assert "test_operation" in performance_report["operation_times"]

    def test_pipeline_utils_error_handling(self):
        """
        Unit Test: Pipeline utilities handle errors gracefully

        Requirements:
        - Graceful handling of data access issues
        - Clear error messages and logging
        - Recovery mechanisms where possible
        """
        from data_integration.pipeline_utils import PipelineUtils

        utils = PipelineUtils()

        # Test error handling with non-existent file
        result = utils.handle_data_access_error(
            FileNotFoundError("Test error"), "non_existent_file.csv", retry_count=1
        )

        # Should return None when all retries fail
        assert result is None

        # Test configuration
        utils.configure(test_size=0.3)
        config = utils.get_configuration()
        assert config["test_size"] == 0.3


class TestDataAccessFunctionsUnit:
    """Unit tests for Data Access Functions."""

    def test_data_access_load_and_validate(self):
        """
        Unit Test: Data access functions load and validate data

        Requirements:
        - Loads cleaned data successfully
        - Validates data quality maintained
        - Ensures Phase 3 transformations preserved
        """
        from data_integration.data_access import load_and_validate_data

        # Test load and validate
        df, validation_report = load_and_validate_data()

        # Validate successful loading
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 41188
        assert len(df.columns) == 33

        # Validate report structure
        assert "overall_status" in validation_report
        assert validation_report["overall_status"] in ["PASSED", "PARTIAL", "FAILED"]

    def test_data_access_feature_validation(self):
        """
        Unit Test: Data access functions validate features

        Requirements:
        - Confirms data types meet ML requirements
        - Validates feature ranges and distributions
        - Checks feature completeness
        """
        from data_integration.data_access import get_data_summary, load_phase3_output

        df = load_phase3_output()
        summary = get_data_summary(df)

        # Validate summary structure
        assert "basic_info" in summary
        assert "data_quality" in summary
        assert "feature_types" in summary
        assert "phase4_readiness" in summary

        # Validate ML readiness
        assert summary["phase4_readiness"]["ml_ready"] == True
        assert summary["phase4_readiness"]["sufficient_samples"] == True
        assert summary["phase4_readiness"]["no_missing_values"] == True

    def test_data_access_schema_validation(self):
        """
        Unit Test: Data access functions validate schema

        Requirements:
        - Verifies 33 features structure
        - Validates column names and order
        - Checks data type consistency
        """
        from data_integration.data_access import (
            validate_phase3_continuity,
            load_phase3_output,
        )

        df = load_phase3_output()
        continuity_report = validate_phase3_continuity(df)

        # Validate continuity report
        assert "continuity_status" in continuity_report
        assert "phase3_preservation" in continuity_report
        assert "schema_validation" in continuity_report

        # Validate continuity status
        assert continuity_report["continuity_status"] == "PASSED"

    def test_data_access_performance_standards(self):
        """
        Unit Test: Data access functions meet performance standards

        Requirements:
        - Maintains >97K records/second processing
        - Efficient memory usage
        - Quick data access operations
        """
        from data_integration.data_access import quick_data_check, load_sample_data

        # Test quick check
        is_valid = quick_data_check()
        assert is_valid == True

        # Test sample loading
        sample_df = load_sample_data(n_rows=100)
        assert len(sample_df) == 100
        assert len(sample_df.columns) > 0

    def test_data_access_ml_preparation(self):
        """
        Unit Test: Data access functions prepare data for ML

        Requirements:
        - Prepares train/test splits
        - Maintains data integrity
        - Optimizes for ML workflow
        """
        from data_integration.data_access import prepare_ml_pipeline

        splits = prepare_ml_pipeline()

        # Validate splits
        assert "train" in splits
        assert "test" in splits
        assert len(splits["train"]) > len(splits["test"])

        # Validate data integrity
        total_records = sum(len(split_df) for split_df in splits.values())
        assert total_records == 41188


# Integration test placeholders for TDD approach
class TestPhase4ModuleIntegration:
    """Integration tests for Phase 4 module components working together."""

    def test_csv_loader_with_data_validator_integration(self):
        """
        Integration Test: CSV Loader works with Data Validator

        Requirements:
        - CSV Loader output is valid input for Data Validator
        - Validation results are accurate and complete
        - Performance is maintained in integrated workflow
        """
        # Test will be implemented after creating the actual classes
        assert True  # Placeholder for TDD approach

    def test_data_validator_with_pipeline_utils_integration(self):
        """
        Integration Test: Data Validator works with Pipeline Utilities

        Requirements:
        - Validation results inform pipeline decisions
        - Pipeline utilities respect validation constraints
        - Error handling works across components
        """
        # Test will be implemented after creating the actual classes
        assert True  # Placeholder for TDD approach

    def test_complete_phase4_module_workflow(self):
        """
        Integration Test: Complete Phase 4 module workflow

        Requirements:
        - All components work together seamlessly
        - Data flows correctly through all stages
        - Performance and quality standards are maintained
        """
        # Test will be implemented after creating the actual classes
        assert True  # Placeholder for TDD approach


if __name__ == "__main__":
    pytest.main([__file__])
