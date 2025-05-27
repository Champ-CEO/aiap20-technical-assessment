"""
Phase 4 Critical Tests - Data Quality Requirements

Critical tests for Phase 4 data quality requirements:
1. Data integrity tests: All Phase 3 transformations preserved
2. Quality score validation: Maintain 100% data quality score from Phase 3
3. Schema consistency: Verify 41,188 records with 33 features structure
4. Performance requirements: Maintain >97K records/second processing standard
5. Error handling requirements: Graceful handling of missing or corrupted files

Following TDD approach: tests define requirements before implementation.
"""

import pytest
import pandas as pd
import numpy as np
import time
import sys
import tempfile
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))


class TestPhase4DataIntegrityValidation:
    """Critical tests for data integrity - all Phase 3 transformations preserved."""

    def test_age_numeric_conversion_preserved(self):
        """
        Critical Test: Age numeric conversion from Phase 3 is preserved

        Requirements:
        - Age column is numeric (not text)
        - Age values are within business range (18-100)
        - No missing values in Age column
        """
        # Arrange
        df = pd.read_csv("data/processed/cleaned-db.csv")

        # Act & Assert
        # 1. Age should be numeric
        assert pd.api.types.is_numeric_dtype(
            df["Age"]
        ), f"Age should be numeric, got {df['Age'].dtype}"

        # 2. Age range validation (Phase 3 business rules)
        age_min, age_max = df["Age"].min(), df["Age"].max()
        assert 18 <= age_min, f"Minimum age should be >= 18, got {age_min}"
        assert age_max <= 100, f"Maximum age should be <= 100, got {age_max}"

        # 3. No missing values in Age
        age_missing = df["Age"].isnull().sum()
        assert (
            age_missing == 0
        ), f"Age should have no missing values, found {age_missing}"

        print(
            f"✅ Age numeric conversion preserved: range {age_min}-{age_max}, no missing values"
        )

    def test_target_binary_encoding_preserved(self):
        """
        Critical Test: Target variable binary encoding from Phase 3 is preserved

        Requirements:
        - Subscription Status is binary encoded (0, 1)
        - No missing values in target
        - Proper distribution maintained
        """
        # Arrange
        df = pd.read_csv("data/processed/cleaned-db.csv")

        # Act & Assert
        # 1. Target should be numeric
        assert pd.api.types.is_numeric_dtype(
            df["Subscription Status"]
        ), f"Target should be numeric, got {df['Subscription Status'].dtype}"

        # 2. Binary values only
        unique_values = set(df["Subscription Status"].unique())
        expected_values = {0, 1}
        assert (
            unique_values == expected_values
        ), f"Target should be binary {expected_values}, got {unique_values}"

        # 3. No missing values
        target_missing = df["Subscription Status"].isnull().sum()
        assert (
            target_missing == 0
        ), f"Target should have no missing values, found {target_missing}"

        # 4. Reasonable distribution (not all 0s or all 1s)
        value_counts = df["Subscription Status"].value_counts()
        assert len(value_counts) == 2, "Should have both 0 and 1 values"
        assert value_counts.min() > 0, "Both classes should be present"

        print(f"✅ Target binary encoding preserved: {value_counts.to_dict()}")

    def test_missing_values_elimination_preserved(self):
        """
        Critical Test: Zero missing values from Phase 3 is preserved

        Requirements:
        - No missing values in any column
        - 100% data completeness maintained
        """
        # Arrange
        df = pd.read_csv("data/processed/cleaned-db.csv")

        # Act
        missing_values_per_column = df.isnull().sum()
        total_missing_values = missing_values_per_column.sum()

        # Assert
        assert (
            total_missing_values == 0
        ), f"Should have 0 missing values, found {total_missing_values}"

        # Verify no column has missing values
        columns_with_missing = missing_values_per_column[missing_values_per_column > 0]
        assert (
            len(columns_with_missing) == 0
        ), f"No columns should have missing values, found: {columns_with_missing.to_dict()}"

        print(
            f"✅ Missing values elimination preserved: 0 missing values across {len(df.columns)} columns"
        )

    def test_contact_method_standardization_preserved(self):
        """
        Critical Test: Contact method standardization from Phase 3 is preserved

        Requirements:
        - Contact methods are standardized (cellular, telephone)
        - No inconsistent values (Cell, Telephone, etc.)
        """
        # Arrange
        df = pd.read_csv("data/processed/cleaned-db.csv")

        # Act
        unique_contact_methods = set(df["Contact Method"].unique())

        # Assert
        expected_methods = {"cellular", "telephone"}
        assert (
            unique_contact_methods == expected_methods
        ), f"Contact methods should be standardized to {expected_methods}, got {unique_contact_methods}"

        print(f"✅ Contact method standardization preserved: {unique_contact_methods}")

    def test_previous_contact_999_handling_preserved(self):
        """
        Critical Test: Previous contact 999 handling from Phase 3 is preserved

        Requirements:
        - No_Previous_Contact binary flag exists
        - Previous Contact Days 999 values handled appropriately
        """
        # Arrange
        df = pd.read_csv("data/processed/cleaned-db.csv")

        # Act & Assert
        # 1. No_Previous_Contact column should exist
        assert (
            "No_Previous_Contact" in df.columns
        ), "No_Previous_Contact binary flag should exist from Phase 3"

        # 2. Should be binary
        unique_flag_values = set(df["No_Previous_Contact"].unique())
        expected_flag_values = {0, 1}
        assert (
            unique_flag_values == expected_flag_values
        ), f"No_Previous_Contact should be binary {expected_flag_values}, got {unique_flag_values}"

        # 3. Previous Contact Days 999 values should be properly flagged
        contact_999_count = (df["Previous Contact Days"] == 999).sum()
        no_previous_contact_count = (df["No_Previous_Contact"] == 1).sum()

        # The 999 values should correspond to No_Previous_Contact flag
        assert (
            contact_999_count == no_previous_contact_count
        ), f"999 values ({contact_999_count}) should match No_Previous_Contact flag ({no_previous_contact_count})"

        # Verify the binary flag correctly identifies 999 values
        assert (df["Previous Contact Days"] == 999).equals(
            df["No_Previous_Contact"] == 1
        ), "No_Previous_Contact flag should correctly identify 999 values"

        print(
            f"✅ Previous contact 999 handling preserved: {contact_999_count} values properly flagged"
        )


class TestPhase4QualityScoreValidation:
    """Critical tests for maintaining 100% data quality score from Phase 3."""

    def test_data_quality_score_100_percent(self):
        """
        Critical Test: Maintain 100% data quality score from Phase 3

        Requirements:
        - No missing values (contributes to quality score)
        - All data types are appropriate
        - All business rules are satisfied
        """
        # Arrange
        df = pd.read_csv("data/processed/cleaned-db.csv")

        # Act - Calculate quality score components
        quality_metrics = {
            "missing_values_score": 100 if df.isnull().sum().sum() == 0 else 0,
            "age_type_score": 100 if pd.api.types.is_numeric_dtype(df["Age"]) else 0,
            "target_type_score": (
                100 if pd.api.types.is_numeric_dtype(df["Subscription Status"]) else 0
            ),
            "age_range_score": (
                100 if (df["Age"].min() >= 18 and df["Age"].max() <= 100) else 0
            ),
            "target_binary_score": (
                100 if set(df["Subscription Status"].unique()) == {0, 1} else 0
            ),
        }

        # Assert
        for metric_name, score in quality_metrics.items():
            assert (
                score == 100
            ), f"Quality metric {metric_name} should be 100%, got {score}%"

        # Overall quality score
        overall_quality_score = sum(quality_metrics.values()) / len(quality_metrics)
        assert (
            overall_quality_score == 100
        ), f"Overall data quality score should be 100%, got {overall_quality_score}%"

        print(f"✅ Data quality score maintained: {overall_quality_score}%")
        print(f"   Quality metrics: {quality_metrics}")


class TestPhase4SchemaConsistency:
    """Critical tests for schema consistency - 41,188 records with 33 features."""

    def test_record_count_consistency(self):
        """
        Critical Test: Verify 41,188 records structure is maintained

        Requirements:
        - Exactly 41,188 records (no data loss)
        - Record count matches Phase 3 output
        """
        # Arrange
        expected_record_count = 41188
        df = pd.read_csv("data/processed/cleaned-db.csv")

        # Act & Assert
        actual_record_count = len(df)
        assert (
            actual_record_count == expected_record_count
        ), f"Should have {expected_record_count} records, got {actual_record_count}"

        print(f"✅ Record count consistency verified: {actual_record_count} records")

    def test_feature_count_consistency(self):
        """
        Critical Test: Verify 33 features structure is maintained

        Requirements:
        - Exactly 33 features (no feature loss)
        - Feature count matches Phase 3 output
        """
        # Arrange
        expected_feature_count = 33
        df = pd.read_csv("data/processed/cleaned-db.csv")

        # Act & Assert
        actual_feature_count = len(df.columns)
        assert (
            actual_feature_count == expected_feature_count
        ), f"Should have {expected_feature_count} features, got {actual_feature_count}"

        print(f"✅ Feature count consistency verified: {actual_feature_count} features")

    def test_core_schema_structure(self):
        """
        Critical Test: Core schema structure from Phase 3 is preserved

        Requirements:
        - All core columns from Phase 3 are present
        - Data types are consistent
        - Column names are preserved
        """
        # Arrange
        df = pd.read_csv("data/processed/cleaned-db.csv")

        # Core columns that must be present
        core_columns = [
            "Client ID",
            "Age",
            "Occupation",
            "Marital Status",
            "Education Level",
            "Credit Default",
            "Housing Loan",
            "Personal Loan",
            "Contact Method",
            "Campaign Calls",
            "Previous Contact Days",
            "Subscription Status",
        ]

        # Act & Assert
        missing_columns = [col for col in core_columns if col not in df.columns]
        assert len(missing_columns) == 0, f"Missing core columns: {missing_columns}"

        # Verify key data types
        type_checks = {
            "Client ID": pd.api.types.is_integer_dtype(df["Client ID"]),
            "Age": pd.api.types.is_numeric_dtype(df["Age"]),
            "Campaign Calls": pd.api.types.is_integer_dtype(df["Campaign Calls"]),
            "Subscription Status": pd.api.types.is_integer_dtype(
                df["Subscription Status"]
            ),
        }

        failed_type_checks = [col for col, passed in type_checks.items() if not passed]
        assert (
            len(failed_type_checks) == 0
        ), f"Data type validation failed for columns: {failed_type_checks}"

        print(
            f"✅ Core schema structure preserved: {len(core_columns)} core columns validated"
        )


class TestPhase4PerformanceRequirements:
    """Critical tests for performance requirements - maintain >97K records/second processing."""

    def test_data_loading_performance_standard(self):
        """
        Critical Test: Maintain >97K records/second processing standard

        Requirements:
        - Loading performance meets Phase 3 standard (97,481 records/second)
        - Performance is consistent across multiple runs
        - Memory usage is reasonable
        """
        # Arrange
        min_records_per_second = (
            97000  # Slightly below Phase 3 achievement for tolerance
        )
        test_runs = 3
        performance_results = []

        # Act - Multiple performance runs
        for run in range(test_runs):
            start_time = time.time()
            df = pd.read_csv("data/processed/cleaned-db.csv")
            load_time = time.time() - start_time

            records_per_second = len(df) / load_time if load_time > 0 else float("inf")
            performance_results.append(records_per_second)

        # Assert
        avg_performance = sum(performance_results) / len(performance_results)
        assert (
            avg_performance >= min_records_per_second
        ), f"Average performance {avg_performance:,.0f} records/second should be >= {min_records_per_second:,.0f}"

        # Verify consistency (no performance should be drastically different)
        min_performance = min(performance_results)
        assert (
            min_performance >= min_records_per_second * 0.8
        ), f"Minimum performance {min_performance:,.0f} should be within 80% of standard"

        print(
            f"✅ Performance standard maintained: {avg_performance:,.0f} records/second average"
        )
        print(
            f"   Performance range: {min(performance_results):,.0f} - {max(performance_results):,.0f} records/second"
        )

    def test_memory_efficiency_validation(self):
        """
        Critical Test: Memory usage is reasonable for 41K records

        Requirements:
        - Memory usage is reasonable for dataset size
        - No memory leaks during loading
        """
        # Arrange
        import psutil
        import gc

        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Act
        df = pd.read_csv("data/processed/cleaned-db.csv")
        after_load_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Clean up
        del df
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Assert
        memory_used = after_load_memory - initial_memory
        memory_per_record = memory_used / 41188 * 1024  # KB per record

        # Reasonable memory usage (should be < 100MB for 41K records)
        assert (
            memory_used < 100
        ), f"Memory usage {memory_used:.1f}MB should be < 100MB for 41K records"

        # Memory should be released after cleanup
        memory_retained = final_memory - initial_memory
        assert (
            memory_retained < memory_used * 0.5
        ), f"Memory retention {memory_retained:.1f}MB should be < 50% of used memory"

        print(
            f"✅ Memory efficiency validated: {memory_used:.1f}MB used, {memory_per_record:.2f}KB per record"
        )


class TestPhase4ErrorHandling:
    """Critical tests for error handling - graceful handling of missing or corrupted files."""

    def test_missing_file_error_handling(self):
        """
        Critical Test: Graceful handling of missing files

        Requirements:
        - Appropriate error when file doesn't exist
        - Clear error message for debugging
        """
        # Arrange
        non_existent_file = "data/processed/non_existent_file.csv"

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            pd.read_csv(non_existent_file)

        print(
            "✅ Missing file error handling validated: FileNotFoundError raised appropriately"
        )

    def test_corrupted_file_error_handling(self):
        """
        Critical Test: Graceful handling of corrupted files

        Requirements:
        - Appropriate error when file is corrupted
        - System doesn't crash on bad data
        """
        # Arrange - Create a corrupted CSV file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmp_file:
            tmp_file.write("This is not a valid CSV file\n")
            tmp_file.write("Random text without proper CSV structure\n")
            tmp_file.write("More invalid content\n")
            corrupted_file_path = tmp_file.name

        try:
            # Act & Assert - pandas might still try to parse it, so check for any error or unexpected result
            try:
                df = pd.read_csv(corrupted_file_path)
                # If it loads, it should be obviously wrong (no proper columns)
                assert (
                    len(df.columns) <= 1
                    or df.empty
                    or "This is not a valid CSV file" in str(df.iloc[0, 0])
                )
                print(
                    "✅ Corrupted file handled: Loaded with obvious corruption indicators"
                )
            except (
                pd.errors.EmptyDataError,
                pd.errors.ParserError,
                ValueError,
                UnicodeDecodeError,
            ):
                print(
                    "✅ Corrupted file error handling validated: Appropriate error raised"
                )

        finally:
            # Cleanup
            try:
                os.unlink(corrupted_file_path)
            except (OSError, PermissionError):
                pass  # Handle Windows file locking

    def test_empty_file_error_handling(self):
        """
        Critical Test: Graceful handling of empty files

        Requirements:
        - Appropriate handling when file is empty
        - Clear error or empty DataFrame response
        """
        # Arrange - Create an empty CSV file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmp_file:
            empty_file_path = tmp_file.name

        try:
            # Act & Assert
            with pytest.raises(pd.errors.EmptyDataError):
                df = pd.read_csv(empty_file_path)

            print(
                "✅ Empty file error handling validated: EmptyDataError raised appropriately"
            )

        finally:
            # Cleanup
            try:
                os.unlink(empty_file_path)
            except (OSError, PermissionError):
                pass  # Handle Windows file locking
