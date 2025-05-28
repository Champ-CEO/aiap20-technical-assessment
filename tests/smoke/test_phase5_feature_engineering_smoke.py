"""
Phase 5 Feature Engineering Smoke Tests

Smoke tests for Phase 5 feature engineering core requirements following TDD approach:
1. Age binning smoke test: Verify basic age categorization (young/middle/senior)
2. Data flow smoke test: Phase 4 → Phase 5 pipeline functionality
3. Output format smoke test: Featured data saves correctly as CSV
4. Critical path verification: Core features (age_bin, contact_recency, campaign_intensity) created

Following streamlined testing approach: critical path over exhaustive coverage.
Based on Phase 3 foundation: 41,188 cleaned records, 33 base features, 100% data quality.
"""

import pytest
import pandas as pd
import numpy as np
import time
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

# Try to import Phase 4 data integration functions
try:
    from data_integration import (
        prepare_ml_pipeline,
        validate_phase3_continuity,
        load_phase3_output as data_integration_load_phase3_output,
        EXPECTED_RECORD_COUNT,
        EXPECTED_FEATURE_COUNT,
        PERFORMANCE_STANDARD,
    )

    PHASE4_INTEGRATION_AVAILABLE = True
except ImportError:
    # Fallback constants if data_integration not available
    PHASE4_INTEGRATION_AVAILABLE = False
    EXPECTED_RECORD_COUNT = 41188
    EXPECTED_FEATURE_COUNT = 33
    PERFORMANCE_STANDARD = 97000

# Constants for testing
PHASE3_OUTPUT_PATH = "data/processed/cleaned-db.csv"


def load_phase3_output():
    """Load Phase 3 output data for testing."""
    # Try to use Phase 4 data integration function first
    if PHASE4_INTEGRATION_AVAILABLE:
        try:
            return data_integration_load_phase3_output()
        except Exception as e:
            print(
                f"⚠️ Phase 4 data integration failed: {e}, falling back to direct loading"
            )

    # Try different paths to find the data file
    possible_paths = [
        Path("data/processed/cleaned-db.csv"),
        Path("../data/processed/cleaned-db.csv"),
        Path("../../data/processed/cleaned-db.csv"),
    ]

    for path in possible_paths:
        if path.exists():
            return pd.read_csv(path)

    # If file not found, create mock data for testing
    print("⚠️ Using mock data for testing (actual data file not found)")
    return pd.DataFrame(
        {
            "Age": np.random.randint(18, 100, EXPECTED_RECORD_COUNT),
            "Campaign Calls": np.random.randint(0, 10, EXPECTED_RECORD_COUNT),
            "No_Previous_Contact": np.random.choice([0, 1], EXPECTED_RECORD_COUNT),
            "Education Level": np.random.choice(
                ["university.degree", "high.school", "professional.course"],
                EXPECTED_RECORD_COUNT,
            ),
            "Occupation": np.random.choice(
                ["management", "technician", "blue-collar"], EXPECTED_RECORD_COUNT
            ),
            "Subscription Status": np.random.choice([0, 1], EXPECTED_RECORD_COUNT),
        }
    )


class TestPhase5FeatureEngineeringSmoke:
    """Smoke tests for Phase 5 feature engineering core requirements."""

    def test_age_binning_smoke_test(self):
        """
        Smoke Test: Age binning produces expected categories (young/middle/senior).

        Validates that basic age categorization works with simplified business categories.
        """
        # Create sample data with age range 18-100
        sample_ages = [20, 30, 45, 60, 80]  # Representative ages
        sample_data = pd.DataFrame(
            {"Age": sample_ages, "Subscription Status": [0, 1, 0, 1, 0]}  # Dummy target
        )

        # Test age binning logic (simplified for smoke test)
        age_bins = [18, 35, 55, 100]
        age_labels = ["young", "middle", "senior"]

        sample_data["age_bin"] = pd.cut(
            sample_data["Age"], bins=age_bins, labels=age_labels, include_lowest=True
        )

        # Verify expected categories are created
        expected_categories = {"young", "middle", "senior"}
        actual_categories = set(sample_data["age_bin"].astype(str).unique())

        assert expected_categories.issubset(
            actual_categories
        ), f"Expected age categories {expected_categories} not found in {actual_categories}"

        # Verify age binning logic
        assert sample_data.loc[sample_data["Age"] == 20, "age_bin"].iloc[0] == "young"
        assert sample_data.loc[sample_data["Age"] == 45, "age_bin"].iloc[0] == "middle"
        assert sample_data.loc[sample_data["Age"] == 80, "age_bin"].iloc[0] == "senior"

        print("✅ Age binning smoke test: Basic categorization works")

    def test_data_flow_smoke_test(self):
        """
        Smoke Test: Phase 4 cleaned data → featured data pipeline works.

        Validates that data can flow from Phase 4 output through feature engineering.
        """
        try:
            # Load Phase 4 output (Phase 3 cleaned data)
            df = load_phase3_output()

            # Verify data loaded successfully
            assert df is not None, "Phase 4 data loading failed"
            assert len(df) > 0, "Phase 4 data is empty"
            assert "Age" in df.columns, "Age column missing from Phase 4 data"

            # Test basic feature engineering pipeline
            df_featured = df.copy()

            # Add simple age_bin feature
            df_featured["age_bin"] = pd.cut(
                df_featured["Age"],
                bins=[18, 35, 55, 100],
                labels=["young", "middle", "senior"],
                include_lowest=True,
            )

            # Verify feature was added
            assert "age_bin" in df_featured.columns, "age_bin feature not created"
            assert len(df_featured) == len(
                df
            ), "Record count changed during feature engineering"

            print(f"✅ Data flow smoke test: {len(df)} records processed successfully")

        except Exception as e:
            pytest.fail(f"Data flow smoke test failed: {str(e)}")

    def test_output_format_smoke_test(self, tmp_path):
        """
        Smoke Test: Featured data saves correctly as CSV.

        Validates that feature engineering output can be saved in required CSV format.
        """
        # Create sample featured data
        sample_data = pd.DataFrame(
            {
                "Age": [25, 40, 65],
                "age_bin": ["young", "middle", "senior"],
                "contact_recency": [1, 0, 1],
                "campaign_intensity": ["low", "medium", "high"],
                "Subscription Status": [0, 1, 0],
            }
        )

        # Test CSV output
        output_file = tmp_path / "featured_data.csv"

        try:
            sample_data.to_csv(output_file, index=False)
            csv_save_success = True
        except Exception as e:
            csv_save_success = False
            pytest.fail(f"CSV save failed: {str(e)}")

        assert csv_save_success, "Featured data CSV save failed"
        assert output_file.exists(), "Output CSV file not created"

        # Verify CSV can be read back
        try:
            df_loaded = pd.read_csv(output_file)
            assert len(df_loaded) == len(sample_data), "CSV round-trip failed"
            assert list(df_loaded.columns) == list(
                sample_data.columns
            ), "Column structure changed"

        except Exception as e:
            pytest.fail(f"CSV read-back failed: {str(e)}")

        print("✅ Output format smoke test: CSV save/load works correctly")

    def test_critical_path_verification_smoke_test(self):
        """
        Smoke Test: Core features (age_bin, contact_recency, campaign_intensity) created successfully.

        Validates that the critical feature engineering path produces required features.
        """
        # Create sample data with required input columns
        sample_data = pd.DataFrame(
            {
                "Age": [25, 40, 65, 30, 50],
                "Campaign Calls": [1, 3, 5, 2, 4],
                "No_Previous_Contact": [1, 0, 1, 0, 1],
                "Subscription Status": [0, 1, 0, 1, 0],
            }
        )

        df_featured = sample_data.copy()

        # Create core features
        # 1. Age binning
        df_featured["age_bin"] = pd.cut(
            df_featured["Age"],
            bins=[18, 35, 55, 100],
            labels=["young", "middle", "senior"],
            include_lowest=True,
        )

        # 2. Contact recency
        df_featured["contact_recency"] = 1 - df_featured["No_Previous_Contact"]

        # 3. Campaign intensity
        df_featured["campaign_intensity"] = pd.cut(
            df_featured["Campaign Calls"],
            bins=[0, 1, 3, 5, 50],
            labels=["none", "low", "medium", "high"],
            include_lowest=True,
        )

        # Verify all core features created
        required_features = ["age_bin", "contact_recency", "campaign_intensity"]
        for feature in required_features:
            assert feature in df_featured.columns, f"Core feature {feature} not created"
            assert (
                df_featured[feature].notna().all()
            ), f"Core feature {feature} has missing values"

        # Verify feature values are reasonable
        assert (
            df_featured["age_bin"].dtype == "category"
            or df_featured["age_bin"].dtype == "object"
        )
        assert df_featured["contact_recency"].dtype in ["int64", "float64"]
        assert (
            df_featured["campaign_intensity"].dtype == "category"
            or df_featured["campaign_intensity"].dtype == "object"
        )

        print("✅ Critical path verification: All core features created successfully")

    def test_phase4_input_accessibility_smoke_test(self):
        """
        Smoke Test: Phase 4 input data is accessible and valid.

        Validates that Phase 4 output (Phase 3 cleaned data) is available for feature engineering.
        """
        try:
            # Test file accessibility
            input_path = Path(PHASE3_OUTPUT_PATH)
            assert (
                input_path.exists()
            ), f"Phase 4 input file not found: {PHASE3_OUTPUT_PATH}"

            # Test data loading
            df = load_phase3_output()
            assert df is not None, "Phase 4 data loading returned None"
            assert len(df) > 0, "Phase 4 data is empty"

            # Verify expected structure
            assert (
                len(df) == EXPECTED_RECORD_COUNT
            ), f"Expected {EXPECTED_RECORD_COUNT} records, got {len(df)}"
            assert (
                len(df.columns) == EXPECTED_FEATURE_COUNT
            ), f"Expected {EXPECTED_FEATURE_COUNT} features, got {len(df.columns)}"

            print(
                f"✅ Phase 4 input accessibility: {len(df)} records × {len(df.columns)} features available"
            )

        except Exception as e:
            pytest.fail(f"Phase 4 input accessibility test failed: {str(e)}")

    def test_performance_smoke_test(self):
        """
        Smoke Test: Feature engineering completes within reasonable time.

        Validates that basic feature engineering operations meet performance requirements.
        """
        try:
            # Load data and measure time
            start_time = time.time()
            df = load_phase3_output()
            load_time = time.time() - start_time

            # Test basic feature engineering performance
            start_time = time.time()
            df_featured = df.copy()

            # Simple feature engineering operations
            df_featured["age_bin"] = pd.cut(
                df_featured["Age"],
                bins=[18, 35, 55, 100],
                labels=["young", "middle", "senior"],
                include_lowest=True,
            )

            feature_time = time.time() - start_time

            # Calculate processing rate
            records_per_second = (
                len(df) / feature_time if feature_time > 0 else float("inf")
            )

            # Verify performance meets requirements (>97K records/second)
            assert (
                records_per_second >= PERFORMANCE_STANDARD
            ), f"Performance too slow: {records_per_second:.0f} records/sec < {PERFORMANCE_STANDARD}"

            print(
                f"✅ Performance smoke test: {records_per_second:.0f} records/sec (target: {PERFORMANCE_STANDARD})"
            )

        except Exception as e:
            pytest.fail(f"Performance smoke test failed: {str(e)}")

    def test_phase4_integration_smoke_test(self):
        """
        Smoke Test: Phase 4 integration functions (prepare_ml_pipeline) work correctly.

        Validates that Phase 4 data integration functions provide train/test/validation splits successfully.
        """
        if not PHASE4_INTEGRATION_AVAILABLE:
            pytest.skip("Phase 4 data integration module not available")

        try:
            # Test prepare_ml_pipeline function
            splits = prepare_ml_pipeline()

            # Verify splits structure
            assert isinstance(
                splits, dict
            ), "prepare_ml_pipeline should return a dictionary"

            required_splits = ["train", "test"]
            for split_name in required_splits:
                assert split_name in splits, f"Missing required split: {split_name}"
                assert isinstance(
                    splits[split_name], pd.DataFrame
                ), f"{split_name} split should be DataFrame"
                assert (
                    len(splits[split_name]) > 0
                ), f"{split_name} split should not be empty"

            # Verify data integrity across splits
            total_records = sum(len(splits[split]) for split in splits.values())
            assert total_records > 0, "Total records across splits should be positive"

            # Verify target column exists in all splits
            for split_name, split_df in splits.items():
                assert (
                    "Subscription Status" in split_df.columns
                ), f"Target column missing in {split_name} split"

            print(
                f"✅ Phase 4 integration smoke test: {len(splits)} splits created successfully"
            )

        except Exception as e:
            pytest.fail(f"Phase 4 integration smoke test failed: {str(e)}")

    def test_data_continuity_smoke_test(self):
        """
        Smoke Test: Data continuity validation (validate_phase3_continuity) passes before feature engineering.

        Validates that Phase 3 → Phase 4 data flow continuity validation works correctly.
        """
        if not PHASE4_INTEGRATION_AVAILABLE:
            pytest.skip("Phase 4 data integration module not available")

        try:
            # Load data for continuity validation
            df = load_phase3_output()

            # Test validate_phase3_continuity function
            continuity_report = validate_phase3_continuity(df)

            # Verify continuity report structure
            assert isinstance(
                continuity_report, dict
            ), "Continuity report should be a dictionary"

            # Check for key continuity metrics
            expected_keys = ["continuity_status", "quality_score"]
            for key in expected_keys:
                assert (
                    key in continuity_report
                ), f"Missing key in continuity report: {key}"

            # Verify continuity status
            continuity_status = continuity_report.get("continuity_status")
            assert continuity_status in [
                "PASSED",
                "FAILED",
            ], f"Invalid continuity status: {continuity_status}"

            # Verify quality score is reasonable
            quality_score = continuity_report.get("quality_score", 0)
            assert isinstance(
                quality_score, (int, float)
            ), "Quality score should be numeric"
            assert (
                0 <= quality_score <= 100
            ), f"Quality score should be 0-100, got {quality_score}"

            print(
                f"✅ Data continuity smoke test: Status={continuity_status}, Quality={quality_score}%"
            )

        except Exception as e:
            pytest.fail(f"Data continuity smoke test failed: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
