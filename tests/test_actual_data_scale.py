"""
Phase 3 Actual Data Scale Testing

Tests Phase 3 pipeline performance and reliability on larger samples of actual data.
Validates scalability and identifies any issues that emerge with realistic data volumes.
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


class TestActualDataScale:
    """Test Phase 3 pipeline scalability on actual data."""

    @pytest.fixture
    def large_actual_data(self):
        """Load larger sample of actual data for scale testing."""
        csv_path = Path("data/raw/initial_dataset.csv")
        if not csv_path.exists():
            pytest.skip("Actual CSV data not available")

        # Load larger sample (10,000 records for scale testing)
        try:
            data = pd.read_csv(csv_path, nrows=10000)
            return data
        except Exception as e:
            pytest.skip(f"Could not load large data sample: {str(e)}")

    def test_large_scale_pipeline_performance(self, large_actual_data):
        """Test pipeline performance on larger data sample."""
        cleaner = BankingDataCleaner()
        validator = DataValidator()
        engineer = FeatureEngineer()

        print(f"\nLarge Scale Test - {len(large_actual_data)} records")

        # Record initial state
        initial_shape = large_actual_data.shape
        initial_missing = large_actual_data.isna().sum().sum()
        initial_unknown = (large_actual_data == "unknown").sum().sum()

        print(f"Initial State:")
        print(f"  Shape: {initial_shape}")
        print(f"  Missing values: {initial_missing}")
        print(f"  Unknown values: {initial_unknown}")

        # Measure performance
        start_time = time.time()

        # Execute pipeline
        cleaned_data = cleaner.clean_banking_data(large_actual_data)
        validation_report = validator.generate_validation_report(cleaned_data)
        final_data = engineer.engineer_features(cleaned_data)

        end_time = time.time()
        processing_time = end_time - start_time
        records_per_second = (
            len(large_actual_data) / processing_time
            if processing_time > 0
            else float("inf")
        )

        # Record final state
        final_missing = final_data.isna().sum().sum()
        final_shape = final_data.shape
        quality_score = validation_report.get("overall_quality_score", 0)

        print(f"Final State:")
        print(f"  Shape: {final_shape}")
        print(f"  Missing values: {final_missing}")
        print(f"  Quality score: {quality_score}")
        print(f"  Processing time: {processing_time:.2f}s")
        print(f"  Records/second: {records_per_second:.1f}")

        # Verify performance requirements
        assert (
            processing_time < 60
        ), f"Should complete within 60 seconds, took {processing_time:.2f}s"
        assert (
            records_per_second > 100
        ), f"Should process >100 records/second, achieved {records_per_second:.1f}"

        # Verify quality requirements
        assert final_missing == 0, "Should eliminate all missing values"
        assert final_shape[0] == initial_shape[0], "Should preserve all records"
        assert (
            quality_score >= 90
        ), f"Should achieve high quality score, got {quality_score}"

    def test_memory_efficiency_large_data(self, large_actual_data):
        """Test memory efficiency on larger data sample."""
        cleaner = BankingDataCleaner()

        # Measure memory usage
        initial_memory = large_actual_data.memory_usage(deep=True).sum() / 1024**2

        # Process data
        cleaned_data = cleaner.clean_banking_data(large_actual_data)

        final_memory = cleaned_data.memory_usage(deep=True).sum() / 1024**2
        memory_increase = final_memory - initial_memory
        memory_increase_pct = (memory_increase / initial_memory) * 100

        print(f"\nMemory Efficiency Test:")
        print(f"  Initial memory: {initial_memory:.2f} MB")
        print(f"  Final memory: {final_memory:.2f} MB")
        print(
            f"  Memory increase: {memory_increase:.2f} MB ({memory_increase_pct:.1f}%)"
        )

        # Verify memory efficiency
        assert (
            final_memory < 100
        ), f"Memory usage should be reasonable (<100MB), used {final_memory:.2f}MB"
        assert (
            memory_increase_pct < 200
        ), f"Memory increase should be <200%, was {memory_increase_pct:.1f}%"

    def test_data_quality_consistency_at_scale(self, large_actual_data):
        """Test that data quality remains consistent at larger scale."""
        cleaner = BankingDataCleaner()

        # Test with different sample sizes
        sample_sizes = (
            [1000, 5000, 10000]
            if len(large_actual_data) >= 10000
            else [len(large_actual_data)]
        )

        quality_results = []

        for size in sample_sizes:
            sample_data = large_actual_data.head(size).copy()

            # Process sample
            cleaned_sample = cleaner.clean_banking_data(sample_data)

            # Calculate quality metrics
            missing_elimination_rate = (
                1.0
                - (cleaned_sample.isna().sum().sum() / sample_data.isna().sum().sum())
                if sample_data.isna().sum().sum() > 0
                else 1.0
            )
            record_preservation_rate = len(cleaned_sample) / len(sample_data)
            age_conversion_success = pd.api.types.is_numeric_dtype(
                cleaned_sample["Age"]
            )

            quality_results.append(
                {
                    "size": size,
                    "missing_elimination_rate": missing_elimination_rate,
                    "record_preservation_rate": record_preservation_rate,
                    "age_conversion_success": age_conversion_success,
                }
            )

            print(f"\nQuality Test - {size} records:")
            print(f"  Missing elimination: {missing_elimination_rate:.1%}")
            print(f"  Record preservation: {record_preservation_rate:.1%}")
            print(f"  Age conversion: {'✅' if age_conversion_success else '❌'}")

        # Verify quality consistency across scales
        for result in quality_results:
            assert (
                result["missing_elimination_rate"] >= 0.95
            ), f"Should eliminate >=95% missing values at {result['size']} records"
            assert (
                result["record_preservation_rate"] >= 0.99
            ), f"Should preserve >=99% records at {result['size']} records"
            assert result[
                "age_conversion_success"
            ], f"Age conversion should succeed at {result['size']} records"

    def test_error_resilience_large_data(self, large_actual_data):
        """Test error resilience with larger data containing edge cases."""
        cleaner = BankingDataCleaner()

        # Introduce additional edge cases to test resilience
        test_data = large_actual_data.copy()

        # Add some extreme edge cases
        if len(test_data) > 100:
            test_data.loc[50, "Age"] = "999 years"  # Extreme age
            test_data.loc[51, "Campaign Calls"] = -999  # Extreme negative
            test_data.loc[52, "Contact Method"] = (
                "InvalidMethod"  # Invalid contact method
            )
            test_data.loc[53, "Subscription Status"] = "maybe"  # Invalid target value

        print(f"\nError Resilience Test - {len(test_data)} records with edge cases")

        # Process data with edge cases
        try:
            cleaned_data = cleaner.clean_banking_data(test_data)
            processing_success = True
            error_message = None
        except Exception as e:
            processing_success = False
            error_message = str(e)

        print(f"Processing success: {'✅' if processing_success else '❌'}")
        if error_message:
            print(f"Error message: {error_message}")

        # Verify error resilience
        assert (
            processing_success
        ), f"Pipeline should handle edge cases gracefully: {error_message}"

        if processing_success:
            # Verify edge cases were handled appropriately
            assert len(cleaned_data) == len(
                test_data
            ), "Should preserve all records despite edge cases"

            # Check if edge cases were handled (some invalid values might become NaN)
            final_missing = cleaned_data.isna().sum().sum()
            if final_missing > 0:
                print(
                    f"  Edge cases resulted in {final_missing} NaN values (acceptable for invalid data)"
                )
                # Allow some NaN values for truly invalid edge cases
                assert (
                    final_missing <= 5
                ), "Should handle most edge cases, allowing few NaN for invalid data"
            else:
                print("  All edge cases handled without creating NaN values")

    def test_full_dataset_sample_validation(self):
        """Test pipeline on a representative sample of the full dataset."""
        csv_path = Path("data/raw/initial_dataset.csv")
        if not csv_path.exists():
            pytest.skip("Actual CSV data not available")

        # Load a representative sample (every 10th record for diversity)
        try:
            full_data = pd.read_csv(csv_path)
            sample_data = full_data.iloc[::10].copy()  # Every 10th record
            sample_data = sample_data.reset_index(drop=True)
        except Exception as e:
            pytest.skip(f"Could not load full dataset sample: {str(e)}")

        print(
            f"\nFull Dataset Sample Test - {len(sample_data)} records (every 10th from {len(full_data)})"
        )

        cleaner = BankingDataCleaner()
        validator = DataValidator()

        # Process representative sample
        start_time = time.time()
        cleaned_data = cleaner.clean_banking_data(sample_data)
        validation_report = validator.generate_validation_report(cleaned_data)
        end_time = time.time()

        processing_time = end_time - start_time
        quality_score = validation_report.get("overall_quality_score", 0)

        print(f"Results:")
        print(f"  Processing time: {processing_time:.2f}s")
        print(f"  Quality score: {quality_score}")
        print(f"  Records preserved: {len(cleaned_data)}/{len(sample_data)}")
        print(
            f"  Missing values eliminated: {sample_data.isna().sum().sum()} → {cleaned_data.isna().sum().sum()}"
        )

        # Verify representative sample results
        assert len(cleaned_data) == len(
            sample_data
        ), "Should preserve all records in representative sample"
        assert (
            cleaned_data.isna().sum().sum() == 0
        ), "Should eliminate all missing values in representative sample"
        assert (
            quality_score >= 90
        ), f"Should achieve high quality on representative sample, got {quality_score}"

        # Estimate full dataset processing time
        estimated_full_time = (processing_time / len(sample_data)) * len(full_data)
        print(f"  Estimated full dataset processing time: {estimated_full_time:.1f}s")

        assert (
            estimated_full_time < 300
        ), f"Estimated full dataset processing should be <5 minutes, estimated {estimated_full_time:.1f}s"


class TestActualDataEdgeCases:
    """Test Phase 3 pipeline on actual data edge cases."""

    def test_actual_data_extreme_values(self):
        """Test handling of extreme values found in actual data."""
        csv_path = Path("data/raw/initial_dataset.csv")
        if not csv_path.exists():
            pytest.skip("Actual CSV data not available")

        # Load data and find extreme values
        data = pd.read_csv(csv_path, nrows=5000)

        # Find records with extreme ages (150 years from EDA)
        extreme_age_records = data[data["Age"].str.contains("150", na=False)]

        # Find records with extreme campaign calls
        extreme_campaign_records = data[
            (data["Campaign Calls"] < -10) | (data["Campaign Calls"] > 50)
        ]

        print(f"\nExtreme Values Test:")
        print(f"  Records with 150 years age: {len(extreme_age_records)}")
        print(f"  Records with extreme campaign calls: {len(extreme_campaign_records)}")

        if len(extreme_age_records) > 0 or len(extreme_campaign_records) > 0:
            # Create test dataset with extreme values
            extreme_data = pd.concat(
                [extreme_age_records.head(10), extreme_campaign_records.head(10)]
            ).drop_duplicates()

            cleaner = BankingDataCleaner()

            # Process extreme values
            cleaned_extreme = cleaner.clean_banking_data(extreme_data)

            print(
                f"  Extreme records processed: {len(cleaned_extreme)}/{len(extreme_data)}"
            )

            # Verify extreme values are handled
            if "Age" in cleaned_extreme.columns:
                age_range = f"{cleaned_extreme['Age'].min():.0f} - {cleaned_extreme['Age'].max():.0f}"
                print(f"  Age range after cleaning: {age_range}")
                assert (
                    cleaned_extreme["Age"].max() <= 100
                ), "Extreme ages should be capped"
                assert (
                    cleaned_extreme["Age"].min() >= 18
                ), "Extreme ages should be floored"

            if "Campaign Calls" in cleaned_extreme.columns:
                campaign_range = f"{cleaned_extreme['Campaign Calls'].min()} - {cleaned_extreme['Campaign Calls'].max()}"
                print(f"  Campaign calls range after cleaning: {campaign_range}")
                assert (
                    cleaned_extreme["Campaign Calls"].min() >= 0
                ), "Negative campaign calls should be handled"
                assert (
                    cleaned_extreme["Campaign Calls"].max() <= 50
                ), "Extreme campaign calls should be capped"
        else:
            pytest.skip("No extreme values found in sample data")

    def test_actual_data_missing_patterns(self):
        """Test handling of actual missing value patterns."""
        csv_path = Path("data/raw/initial_dataset.csv")
        if not csv_path.exists():
            pytest.skip("Actual CSV data not available")

        # Load data and analyze missing patterns
        data = pd.read_csv(csv_path, nrows=2000)

        # Find records with high missing value density
        missing_per_record = data.isna().sum(axis=1)
        high_missing_records = data[
            missing_per_record >= 3
        ]  # Records with 3+ missing values

        print(f"\nMissing Patterns Test:")
        print(f"  Records with 3+ missing values: {len(high_missing_records)}")

        if len(high_missing_records) > 0:
            cleaner = BankingDataCleaner()

            # Process high missing value records
            cleaned_missing = cleaner.clean_banking_data(high_missing_records)

            print(
                f"  High missing records processed: {len(cleaned_missing)}/{len(high_missing_records)}"
            )
            print(
                f"  Missing values eliminated: {high_missing_records.isna().sum().sum()} → {cleaned_missing.isna().sum().sum()}"
            )

            # Verify missing patterns are handled
            assert len(cleaned_missing) == len(
                high_missing_records
            ), "Should preserve records with high missing values"
            assert (
                cleaned_missing.isna().sum().sum() == 0
            ), "Should eliminate all missing values"
        else:
            pytest.skip("No high missing value patterns found in sample data")
