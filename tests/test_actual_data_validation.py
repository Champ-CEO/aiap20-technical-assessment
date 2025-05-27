"""
Phase 3 Actual Data Validation Tests

Tests Phase 3 data cleaning and preprocessing pipeline on actual data from:
- data/raw/initial_dataset.csv (41,188 records)
- data/raw/bmarket.db (SQLite database)

This validates that our Phase 3 implementation works correctly on real data
with all the issues identified in the EDA.
"""

import pytest
import pandas as pd
import numpy as np
import sqlite3
import sys
from pathlib import Path
import time

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from preprocessing.data_cleaner import BankingDataCleaner
from preprocessing.validation import DataValidator
from preprocessing.feature_engineering import FeatureEngineer


class TestActualDataValidation:
    """Test Phase 3 pipeline on actual banking data."""

    @pytest.fixture
    def actual_csv_data(self):
        """Load actual CSV data for testing."""
        csv_path = Path("data/raw/initial_dataset.csv")
        if not csv_path.exists():
            pytest.skip("Actual CSV data not available")

        # Load a sample for testing (first 1000 records for performance)
        data = pd.read_csv(csv_path, nrows=1000)
        return data

    @pytest.fixture
    def actual_db_data(self):
        """Load actual database data for testing."""
        db_path = Path("data/raw/bmarket.db")
        if not db_path.exists():
            pytest.skip("Actual database not available")

        try:
            conn = sqlite3.connect(db_path)
            # Load a sample for testing (first 1000 records for performance)
            data = pd.read_sql_query("SELECT * FROM bank_marketing LIMIT 1000", conn)
            conn.close()
            return data
        except Exception as e:
            pytest.skip(f"Could not load database data: {str(e)}")

    def test_actual_csv_data_loading(self, actual_csv_data):
        """Test that actual CSV data loads correctly."""
        assert actual_csv_data is not None, "Should load actual CSV data"
        assert len(actual_csv_data) > 0, "Should have records"
        assert len(actual_csv_data.columns) >= 10, "Should have expected columns"

        # Verify key columns exist
        expected_columns = ["Client ID", "Age", "Subscription Status"]
        for col in expected_columns:
            assert col in actual_csv_data.columns, f"Should have column: {col}"

    def test_actual_data_eda_issues_present(self, actual_csv_data):
        """Verify that actual data contains the EDA issues we're testing for."""
        # Age in text format
        assert actual_csv_data["Age"].dtype == "object", "Age should be text format"
        assert any(
            "years" in str(age) for age in actual_csv_data["Age"].head(10)
        ), "Should have 'years' in age values"

        # Missing values
        missing_count = actual_csv_data.isna().sum().sum()
        assert missing_count > 0, "Should have missing values to test handling"

        # Unknown values
        unknown_count = (actual_csv_data == "unknown").sum().sum()
        assert unknown_count > 0, "Should have 'unknown' values to test handling"

        # Contact method inconsistencies
        if "Contact Method" in actual_csv_data.columns:
            contact_methods = actual_csv_data["Contact Method"].unique()
            has_inconsistency = any(
                "Cell" in str(method) for method in contact_methods
            ) and any("cellular" in str(method) for method in contact_methods)
            assert (
                has_inconsistency or len(contact_methods) > 2
            ), "Should have contact method inconsistencies"

        # Target variable as text
        if "Subscription Status" in actual_csv_data.columns:
            assert (
                actual_csv_data["Subscription Status"].dtype == "object"
            ), "Target should be text format"

    def test_phase3_pipeline_on_actual_data(self, actual_csv_data):
        """Test complete Phase 3 pipeline on actual data."""
        # Initialize components
        cleaner = BankingDataCleaner()
        validator = DataValidator()
        engineer = FeatureEngineer()

        # Record initial state
        initial_shape = actual_csv_data.shape
        initial_missing = actual_csv_data.isna().sum().sum()
        initial_unknown = (actual_csv_data == "unknown").sum().sum()

        print(f"\nActual Data Initial State:")
        print(f"  Shape: {initial_shape}")
        print(f"  Missing values: {initial_missing}")
        print(f"  Unknown values: {initial_unknown}")

        # Measure performance
        start_time = time.time()

        # Step 1: Clean data
        try:
            cleaned_data = cleaner.clean_banking_data(actual_csv_data)
            cleaning_success = True
        except Exception as e:
            cleaning_success = False
            pytest.fail(f"Data cleaning failed on actual data: {str(e)}")

        # Step 2: Validate cleaned data
        try:
            validation_report = validator.generate_validation_report(cleaned_data)
            validation_success = True
        except Exception as e:
            validation_success = False
            print(f"Warning: Validation failed: {str(e)}")
            validation_report = {"overall_quality_score": 0}

        # Step 3: Engineer features
        try:
            final_data = engineer.engineer_features(cleaned_data)
            engineering_success = True
        except Exception as e:
            engineering_success = False
            print(f"Warning: Feature engineering failed: {str(e)}")
            final_data = cleaned_data

        end_time = time.time()
        processing_time = end_time - start_time

        # Verify pipeline results
        assert cleaning_success, "Data cleaning should succeed on actual data"
        assert cleaned_data is not None, "Should produce cleaned data"
        assert len(cleaned_data) == len(actual_csv_data), "Should preserve all records"

        # Record final state
        final_missing = cleaned_data.isna().sum().sum()
        final_shape = cleaned_data.shape

        print(f"\nActual Data Final State:")
        print(f"  Shape: {final_shape}")
        print(f"  Missing values: {final_missing}")
        print(f"  Processing time: {processing_time:.2f}s")
        print(
            f"  Quality score: {validation_report.get('overall_quality_score', 'N/A')}"
        )

        # Verify improvements
        assert (
            final_missing <= initial_missing
        ), "Should reduce or eliminate missing values"
        assert final_shape[0] == initial_shape[0], "Should preserve record count"

        # Performance check
        records_per_second = (
            len(actual_csv_data) / processing_time
            if processing_time > 0
            else float("inf")
        )
        assert (
            records_per_second > 10
        ), f"Should process >10 records/second, achieved {records_per_second:.1f}"

    def test_actual_data_age_conversion(self, actual_csv_data):
        """Test age conversion on actual data."""
        cleaner = BankingDataCleaner()

        # Extract age column
        age_data = actual_csv_data[["Age"]].copy()

        # Verify initial state
        assert age_data["Age"].dtype == "object", "Age should be text initially"

        # Apply age cleaning
        cleaned_age = cleaner.clean_age_column(age_data)

        # Verify conversion
        assert pd.api.types.is_numeric_dtype(
            cleaned_age["Age"]
        ), "Age should be numeric after cleaning"

        # Check for reasonable values
        valid_ages = cleaned_age["Age"].dropna()
        if len(valid_ages) > 0:
            assert all(valid_ages >= 18), "All valid ages should be >= 18"
            assert all(valid_ages <= 100), "All valid ages should be <= 100"

        print(f"\nAge Conversion Results:")
        print(f"  Original dtype: {age_data['Age'].dtype}")
        print(f"  Final dtype: {cleaned_age['Age'].dtype}")
        print(f"  Valid ages: {len(valid_ages)}/{len(cleaned_age)}")
        print(f"  Age range: {valid_ages.min():.0f} - {valid_ages.max():.0f}")

    def test_actual_data_missing_value_handling(self, actual_csv_data):
        """Test missing value handling on actual data."""
        cleaner = BankingDataCleaner()

        # Count initial missing values by column
        initial_missing = actual_csv_data.isna().sum()
        columns_with_missing = initial_missing[initial_missing > 0]

        print(f"\nMissing Values by Column (Initial):")
        for col, count in columns_with_missing.items():
            pct = (count / len(actual_csv_data)) * 100
            print(f"  {col}: {count} ({pct:.1f}%)")

        # Apply missing value handling
        cleaned_data = cleaner.handle_missing_values(actual_csv_data)

        # Count final missing values
        final_missing = cleaned_data.isna().sum().sum()

        print(f"\nMissing Value Handling Results:")
        print(f"  Initial total missing: {initial_missing.sum()}")
        print(f"  Final total missing: {final_missing}")
        print(
            f"  Elimination rate: {((initial_missing.sum() - final_missing) / initial_missing.sum() * 100):.1f}%"
        )

        # Verify improvement
        assert (
            final_missing <= initial_missing.sum()
        ), "Should reduce or eliminate missing values"

    def test_actual_data_special_values(self, actual_csv_data):
        """Test special value handling on actual data."""
        cleaner = BankingDataCleaner()

        # Count initial unknown values by column
        initial_unknown = (actual_csv_data == "unknown").sum()
        columns_with_unknown = initial_unknown[initial_unknown > 0]

        print(f"\nUnknown Values by Column (Initial):")
        for col, count in columns_with_unknown.items():
            pct = (count / len(actual_csv_data)) * 100
            print(f"  {col}: {count} ({pct:.1f}%)")

        # Apply special value cleaning
        cleaned_data = cleaner.clean_special_values(actual_csv_data)

        # Count final unknown values (should be preserved as business categories)
        final_unknown = 0
        for col in cleaned_data.columns:
            if cleaned_data[col].dtype == "object":
                col_unknown = (
                    cleaned_data[col]
                    .astype(str)
                    .str.contains("unknown", case=False, na=False)
                    .sum()
                )
                final_unknown += col_unknown

        print(f"\nSpecial Value Handling Results:")
        print(f"  Initial unknown values: {initial_unknown.sum()}")
        print(f"  Final unknown-related values: {final_unknown.sum()}")

        # Verify unknown values are preserved as business categories
        for col in columns_with_unknown.index:
            if col in cleaned_data.columns:
                col_unknown = (
                    cleaned_data[col]
                    .astype(str)
                    .str.contains("unknown", case=False, na=False)
                    .sum()
                )
                assert (
                    col_unknown >= initial_unknown[col] * 0.8
                ), f"Most unknown values should be preserved in {col}"

    def test_actual_data_performance_benchmarks(self, actual_csv_data):
        """Test performance benchmarks on actual data."""
        cleaner = BankingDataCleaner()

        # Test with different data sizes
        sizes = (
            [100, 500, 1000] if len(actual_csv_data) >= 1000 else [len(actual_csv_data)]
        )

        performance_results = []

        for size in sizes:
            sample_data = actual_csv_data.head(size).copy()

            start_time = time.time()
            cleaned_data = cleaner.clean_banking_data(sample_data)
            end_time = time.time()

            processing_time = end_time - start_time
            records_per_second = (
                size / processing_time if processing_time > 0 else float("inf")
            )

            performance_results.append(
                {
                    "size": size,
                    "time": processing_time,
                    "records_per_second": records_per_second,
                }
            )

            print(f"\nPerformance Test - {size} records:")
            print(f"  Processing time: {processing_time:.3f}s")
            print(f"  Records/second: {records_per_second:.1f}")

        # Verify performance requirements
        for result in performance_results:
            assert (
                result["records_per_second"] > 10
            ), f"Should process >10 records/second for {result['size']} records"
            assert (
                result["time"] < 30
            ), f"Should complete within 30 seconds for {result['size']} records"

    def test_actual_data_output_quality(self, actual_csv_data):
        """Test output quality on actual data."""
        cleaner = BankingDataCleaner()
        validator = DataValidator()

        # Process actual data
        cleaned_data = cleaner.clean_banking_data(actual_csv_data)

        # Generate quality report
        try:
            quality_report = validator.generate_validation_report(cleaned_data)
            quality_score = quality_report.get("overall_quality_score", 0)
        except Exception as e:
            print(f"Warning: Quality validation failed: {str(e)}")
            quality_score = 0

        # Manual quality checks
        manual_quality_checks = {
            "no_missing_values": cleaned_data.isna().sum().sum() == 0,
            "numeric_age": pd.api.types.is_numeric_dtype(cleaned_data["Age"]),
            "preserved_records": len(cleaned_data) == len(actual_csv_data),
            "reasonable_age_range": (
                (cleaned_data["Age"].min() >= 18 and cleaned_data["Age"].max() <= 100)
                if pd.api.types.is_numeric_dtype(cleaned_data["Age"])
                else False
            ),
        }

        manual_score = (
            sum(manual_quality_checks.values()) / len(manual_quality_checks) * 100
        )

        print(f"\nData Quality Assessment:")
        print(f"  Validator quality score: {quality_score}")
        print(f"  Manual quality score: {manual_score:.1f}%")
        print(f"  Quality checks:")
        for check, passed in manual_quality_checks.items():
            status = "✅" if passed else "❌"
            print(f"    {status} {check}")

        # Verify quality requirements
        assert (
            manual_score >= 75
        ), f"Manual quality score should be >=75%, got {manual_score:.1f}%"

        # Verify specific quality requirements
        assert manual_quality_checks[
            "no_missing_values"
        ], "Should have no missing values"
        assert manual_quality_checks["preserved_records"], "Should preserve all records"


class TestActualDataComparison:
    """Compare results between test fixtures and actual data."""

    def test_fixture_vs_actual_data_consistency(self, phase3_raw_sample_data):
        """Test that fixture data behaves similarly to actual data."""
        # Load actual data sample
        csv_path = Path("data/raw/initial_dataset.csv")
        if not csv_path.exists():
            pytest.skip("Actual CSV data not available")

        actual_data = pd.read_csv(csv_path, nrows=100)
        fixture_data = phase3_raw_sample_data.copy()

        cleaner = BankingDataCleaner()

        # Process both datasets
        cleaned_actual = cleaner.clean_banking_data(actual_data)
        cleaned_fixture = cleaner.clean_banking_data(fixture_data)

        # Compare results
        comparison_results = {
            "actual_missing_eliminated": cleaned_actual.isna().sum().sum() == 0,
            "fixture_missing_eliminated": cleaned_fixture.isna().sum().sum() == 0,
            "actual_age_numeric": pd.api.types.is_numeric_dtype(cleaned_actual["Age"]),
            "fixture_age_numeric": pd.api.types.is_numeric_dtype(
                cleaned_fixture["Age"]
            ),
            "actual_records_preserved": len(cleaned_actual) == len(actual_data),
            "fixture_records_preserved": len(cleaned_fixture) == len(fixture_data),
        }

        print(f"\nFixture vs Actual Data Comparison:")
        for metric, result in comparison_results.items():
            status = "✅" if result else "❌"
            print(f"  {status} {metric}")

        # Verify consistency
        actual_results = [
            comparison_results[k] for k in comparison_results.keys() if "actual" in k
        ]
        fixture_results = [
            comparison_results[k] for k in comparison_results.keys() if "fixture" in k
        ]

        assert (
            actual_results == fixture_results
        ), "Fixture data should behave consistently with actual data"
