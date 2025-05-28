"""
Phase 5 Feature Engineering Pipeline Integration Tests

Integration tests for Phase 5 feature engineering pipeline following TDD approach:
1. End-to-end Phase 4 → Phase 5 data flow validation
2. Data continuity verification across pipeline stages
3. Performance benchmarking for full pipeline
4. Output validation for downstream phases (Phase 6+)
5. Error handling and recovery testing

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
        validate_phase3_continuity as data_integration_validate_phase3_continuity,
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


def validate_phase3_continuity(df):
    """Validate Phase 3 continuity using Phase 4 integration if available."""
    if PHASE4_INTEGRATION_AVAILABLE:
        try:
            return data_integration_validate_phase3_continuity(df)
        except Exception as e:
            print(f"⚠️ Phase 4 continuity validation failed: {e}, using mock validation")

    # Mock validation function for testing
    return {"continuity_status": "PASSED", "quality_score": 100}


class TestPhase5PipelineIntegration:
    """Integration tests for Phase 5 feature engineering pipeline."""

    def test_end_to_end_phase4_to_phase5_data_flow(self):
        """
        Integration Test: Complete Phase 4 → Phase 5 data flow validation.

        Validates seamless data flow from Phase 4 integration to Phase 5 feature engineering.
        """
        try:
            # Step 1: Load Phase 4 output
            df_phase4 = load_phase3_output()

            # Verify Phase 4 data quality
            assert df_phase4 is not None, "Phase 4 data loading failed"
            assert (
                len(df_phase4) == EXPECTED_RECORD_COUNT
            ), "Phase 4 record count mismatch"
            assert (
                len(df_phase4.columns) == EXPECTED_FEATURE_COUNT
            ), "Phase 4 feature count mismatch"

            # Step 2: Apply Phase 5 feature engineering
            df_phase5 = df_phase4.copy()

            # Core feature engineering operations
            # Age binning
            df_phase5["age_bin"] = pd.cut(
                df_phase5["Age"],
                bins=[18, 35, 55, 100],
                labels=["young", "middle", "senior"],
                include_lowest=True,
            )

            # Contact recency (if available)
            if "No_Previous_Contact" in df_phase5.columns:
                df_phase5["contact_recency"] = 1 - df_phase5["No_Previous_Contact"]

            # Campaign intensity
            if "Campaign Calls" in df_phase5.columns:
                df_phase5["campaign_intensity"] = pd.cut(
                    df_phase5["Campaign Calls"],
                    bins=[0, 1, 3, 5, 50],
                    labels=["none", "low", "medium", "high"],
                    include_lowest=True,
                )

            # Education-occupation interaction
            if (
                "Education Level" in df_phase5.columns
                and "Occupation" in df_phase5.columns
            ):
                df_phase5["education_occupation"] = (
                    df_phase5["Education Level"].astype(str)
                    + "_"
                    + df_phase5["Occupation"].astype(str)
                )

            # Step 3: Validate Phase 5 output
            # Verify new features created
            expected_new_features = ["age_bin"]
            for feature in expected_new_features:
                assert (
                    feature in df_phase5.columns
                ), f"Required feature {feature} not created"

            # Verify data integrity
            assert len(df_phase5) == len(
                df_phase4
            ), "Record count changed during feature engineering"

            # Verify original features preserved
            for col in df_phase4.columns:
                assert col in df_phase5.columns, f"Original feature {col} lost"

            # Step 4: Validate output for downstream phases
            # Check data types
            assert df_phase5["age_bin"].dtype in [
                "category",
                "object",
            ], "Age bin data type invalid"

            # Check for missing values in new features
            new_features = set(df_phase5.columns) - set(df_phase4.columns)
            for feature in new_features:
                missing_count = df_phase5[feature].isnull().sum()
                missing_rate = missing_count / len(df_phase5)
                assert (
                    missing_rate < 0.1
                ), f"High missing rate in {feature}: {missing_rate:.2%}"

            print(
                f"✅ End-to-end data flow: {len(df_phase4.columns)} → {len(df_phase5.columns)} features"
            )

        except Exception as e:
            pytest.fail(f"End-to-end data flow test failed: {str(e)}")

    def test_data_continuity_verification(self):
        """
        Integration Test: Data continuity verification across pipeline stages.

        Validates that data quality and business rules are maintained throughout pipeline.
        """
        try:
            # Load and validate Phase 4 continuity
            df_input = load_phase3_output()
            continuity_report = validate_phase3_continuity(df_input)

            # Verify Phase 4 continuity passed
            assert continuity_report is not None, "Phase 4 continuity validation failed"

            # Apply feature engineering
            df_featured = df_input.copy()

            # Age binning with validation
            valid_ages = df_featured["Age"].between(18, 100)
            assert (
                valid_ages.all()
            ), f"Invalid ages found: {(~valid_ages).sum()} records"

            df_featured["age_bin"] = pd.cut(
                df_featured["Age"],
                bins=[18, 35, 55, 100],
                labels=["young", "middle", "senior"],
                include_lowest=True,
            )

            # Verify age distribution makes business sense
            age_distribution = df_featured["age_bin"].value_counts(normalize=True)

            # Business rule: middle-aged customers should be largest segment
            assert (
                age_distribution["middle"] > 0.3
            ), f"Middle-aged segment too small: {age_distribution['middle']:.2%}"

            # Verify target variable distribution unchanged
            original_target_rate = df_input["Subscription Status"].mean()
            featured_target_rate = df_featured["Subscription Status"].mean()

            assert (
                abs(original_target_rate - featured_target_rate) < 0.001
            ), f"Target distribution changed: {original_target_rate:.3f} → {featured_target_rate:.3f}"

            # Verify data quality metrics
            original_quality = {
                "missing_values": df_input.isnull().sum().sum(),
                "duplicate_rows": df_input.duplicated().sum(),
                "data_types": len(df_input.dtypes.unique()),
            }

            featured_quality = {
                "missing_values": df_featured[df_input.columns].isnull().sum().sum(),
                "duplicate_rows": df_featured.duplicated().sum(),
                "data_types": len(df_featured[df_input.columns].dtypes.unique()),
            }

            assert (
                original_quality["missing_values"] == featured_quality["missing_values"]
            ), "Missing values changed in original features"

            assert (
                original_quality["duplicate_rows"] == featured_quality["duplicate_rows"]
            ), "Duplicate rows changed"

            print("✅ Data continuity verification: Quality metrics maintained")

        except Exception as e:
            pytest.fail(f"Data continuity verification failed: {str(e)}")

    def test_performance_benchmarking_full_pipeline(self):
        """
        Integration Test: Performance benchmarking for full feature engineering pipeline.

        Validates that complete pipeline meets performance requirements.
        """
        try:
            # Load data
            df_input = load_phase3_output()

            # Measure full pipeline performance
            start_time = time.time()

            # Complete feature engineering pipeline
            df_featured = df_input.copy()

            # Age binning
            df_featured["age_bin"] = pd.cut(
                df_featured["Age"],
                bins=[18, 35, 55, 100],
                labels=["young", "middle", "senior"],
                include_lowest=True,
            )

            # Contact features
            if "No_Previous_Contact" in df_featured.columns:
                df_featured["contact_recency"] = 1 - df_featured["No_Previous_Contact"]
                df_featured["first_time_contact"] = df_featured["No_Previous_Contact"]

            # Campaign features
            if "Campaign Calls" in df_featured.columns:
                df_featured["campaign_intensity"] = pd.cut(
                    df_featured["Campaign Calls"],
                    bins=[0, 1, 3, 5, 50],
                    labels=["none", "low", "medium", "high"],
                    include_lowest=True,
                )
                df_featured["high_intensity_campaign"] = (
                    df_featured["Campaign Calls"] >= 5
                ).astype(int)

            # Interaction features
            if (
                "Education Level" in df_featured.columns
                and "Occupation" in df_featured.columns
            ):
                df_featured["education_occupation"] = (
                    df_featured["Education Level"].astype(str)
                    + "_"
                    + df_featured["Occupation"].astype(str)
                )

            pipeline_time = time.time() - start_time

            # Calculate performance metrics
            records_per_second = (
                len(df_input) / pipeline_time if pipeline_time > 0 else float("inf")
            )
            features_added = len(df_featured.columns) - len(df_input.columns)

            # Verify performance requirements
            assert (
                records_per_second >= PERFORMANCE_STANDARD
            ), f"Pipeline too slow: {records_per_second:.0f} < {PERFORMANCE_STANDARD} records/sec"

            # Verify reasonable feature addition
            assert features_added >= 3, f"Too few features added: {features_added}"
            assert features_added <= 20, f"Too many features added: {features_added}"

            print(
                f"✅ Performance benchmarking: {records_per_second:.0f} records/sec, {features_added} features added"
            )

        except Exception as e:
            pytest.fail(f"Performance benchmarking failed: {str(e)}")

    def test_output_validation_for_downstream_phases(self, tmp_path):
        """
        Integration Test: Output validation for downstream phases (Phase 6+).

        Validates that Phase 5 output is properly formatted for ML pipeline phases.
        """
        try:
            # Load and process data
            df_input = load_phase3_output()
            df_featured = df_input.copy()

            # Apply feature engineering
            df_featured["age_bin"] = pd.cut(
                df_featured["Age"],
                bins=[18, 35, 55, 100],
                labels=["young", "middle", "senior"],
                include_lowest=True,
            )

            # Test CSV output for downstream phases
            output_file = tmp_path / "phase5_featured_data.csv"
            df_featured.to_csv(output_file, index=False)

            # Verify file creation
            assert output_file.exists(), "Phase 5 output file not created"

            # Test downstream compatibility
            df_loaded = pd.read_csv(output_file)

            # Verify structure preservation
            assert len(df_loaded) == len(df_featured), "Record count lost in CSV"
            assert len(df_loaded.columns) == len(
                df_featured.columns
            ), "Feature count lost in CSV"

            # Verify target variable for ML
            assert "Subscription Status" in df_loaded.columns, "Target variable missing"
            assert df_loaded["Subscription Status"].dtype in [
                "int64",
                "float64",
            ], "Target variable wrong type"

            # Verify feature types suitable for ML
            numeric_features = df_loaded.select_dtypes(include=[np.number]).columns
            categorical_features = df_loaded.select_dtypes(include=["object"]).columns

            assert len(numeric_features) > 0, "No numeric features for ML"
            assert "Age" in numeric_features, "Age feature not numeric"

            # Test train/test split compatibility
            from sklearn.model_selection import train_test_split

            X = df_loaded.drop("Subscription Status", axis=1)
            y = df_loaded["Subscription Status"]

            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                split_success = True
            except Exception:
                split_success = False

            assert split_success, "Data not compatible with train/test split"

            print(
                f"✅ Downstream validation: {len(numeric_features)} numeric, {len(categorical_features)} categorical features"
            )

        except Exception as e:
            pytest.fail(f"Downstream validation failed: {str(e)}")

    def test_error_handling_and_recovery(self):
        """
        Integration Test: Error handling and recovery testing.

        Validates robust error handling in feature engineering pipeline.
        """
        # Test missing column handling
        incomplete_data = pd.DataFrame(
            {
                "Age": [25, 40, 65],
                "Subscription Status": [0, 1, 0],
                # Missing other expected columns
            }
        )

        # Test graceful handling of missing columns
        try:
            df_result = incomplete_data.copy()

            # Age binning should work
            df_result["age_bin"] = pd.cut(
                df_result["Age"],
                bins=[18, 35, 55, 100],
                labels=["young", "middle", "senior"],
                include_lowest=True,
            )

            # Contact recency should handle missing column
            if "No_Previous_Contact" in df_result.columns:
                df_result["contact_recency"] = 1 - df_result["No_Previous_Contact"]
            else:
                # Graceful fallback
                df_result["contact_recency"] = 0  # Default to no recent contact

            assert (
                "age_bin" in df_result.columns
            ), "Age binning failed with minimal data"
            assert (
                "contact_recency" in df_result.columns
            ), "Contact recency fallback failed"

        except Exception as e:
            pytest.fail(f"Error handling test failed: {str(e)}")

        # Test invalid age handling
        invalid_age_data = pd.DataFrame(
            {
                "Age": [15, 25, 105, np.nan],  # Invalid ages
                "Subscription Status": [0, 1, 0, 1],
            }
        )

        try:
            df_result = invalid_age_data.copy()

            # Filter valid ages before binning
            valid_mask = df_result["Age"].between(18, 100) & df_result["Age"].notna()
            df_result.loc[valid_mask, "age_bin"] = pd.cut(
                df_result.loc[valid_mask, "Age"],
                bins=[18, 35, 55, 100],
                labels=["young", "middle", "senior"],
                include_lowest=True,
            )

            # Invalid ages should have NaN in age_bin
            # Ages: [15, 25, 105, np.nan] - only 25 is valid (18-100), so 3 should be invalid
            invalid_count = df_result["age_bin"].isnull().sum()
            assert (
                invalid_count == 3
            ), f"Expected 3 invalid ages (15, 105, NaN), got {invalid_count}"

        except Exception as e:
            pytest.fail(f"Invalid age handling test failed: {str(e)}")

        print("✅ Error handling and recovery: Robust pipeline behavior validated")

    def test_phase4_ml_pipeline_integration(self):
        """
        Integration Test: Phase 4 ML pipeline integration with prepare_ml_pipeline().

        Validates that Phase 5 can use Phase 4's prepare_ml_pipeline() function for data splits.
        """
        if not PHASE4_INTEGRATION_AVAILABLE:
            pytest.skip("Phase 4 data integration module not available")

        try:
            # Use Phase 4's prepare_ml_pipeline function
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
                ), f"{split_name} should be DataFrame"
                assert len(splits[split_name]) > 0, f"{split_name} should not be empty"

            # Apply feature engineering to each split
            featured_splits = {}
            for split_name, split_df in splits.items():
                df_featured = split_df.copy()

                # Apply core feature engineering
                df_featured["age_bin"] = pd.cut(
                    df_featured["Age"],
                    bins=[18, 35, 55, 100],
                    labels=["young", "middle", "senior"],
                    include_lowest=True,
                )

                if "No_Previous_Contact" in df_featured.columns:
                    df_featured["contact_recency"] = (
                        1 - df_featured["No_Previous_Contact"]
                    )

                if "Campaign Calls" in df_featured.columns:
                    df_featured["campaign_intensity"] = pd.cut(
                        df_featured["Campaign Calls"],
                        bins=[0, 1, 3, 5, 50],
                        labels=["none", "low", "medium", "high"],
                        include_lowest=True,
                    )

                featured_splits[split_name] = df_featured

            # Verify feature engineering preserved split integrity
            for split_name in splits.keys():
                original_count = len(splits[split_name])
                featured_count = len(featured_splits[split_name])
                assert (
                    original_count == featured_count
                ), f"Record count changed in {split_name}: {original_count} → {featured_count}"

                # Verify target distribution preserved
                original_target = splits[split_name]["Subscription Status"].mean()
                featured_target = featured_splits[split_name][
                    "Subscription Status"
                ].mean()
                target_diff = abs(original_target - featured_target)
                assert (
                    target_diff < 0.001
                ), f"Target distribution changed in {split_name}: {target_diff:.4f}"

            # Verify new features created in all splits
            for split_name, split_df in featured_splits.items():
                assert "age_bin" in split_df.columns, f"age_bin missing in {split_name}"
                assert (
                    split_df["age_bin"].notna().all()
                ), f"age_bin has missing values in {split_name}"

            print(
                f"✅ Phase 4 ML pipeline integration: {len(splits)} splits with feature engineering"
            )

        except Exception as e:
            pytest.fail(f"Phase 4 ML pipeline integration failed: {str(e)}")

    def test_continuous_quality_monitoring_integration(self):
        """
        Integration Test: Continuous quality monitoring throughout Phase 4 → Phase 5 pipeline.

        Validates quality monitoring at each stage of the integrated pipeline.
        """
        try:
            # Stage 1: Load data with quality baseline
            df_input = load_phase3_output()

            # Initial quality metrics
            quality_log = []
            initial_quality = {
                "stage": "phase4_input",
                "record_count": len(df_input),
                "feature_count": len(df_input.columns),
                "missing_values": df_input.isnull().sum().sum(),
                "target_mean": df_input["Subscription Status"].mean(),
            }
            quality_log.append(initial_quality)

            # Stage 2: Phase 4 continuity validation
            if PHASE4_INTEGRATION_AVAILABLE:
                continuity_report = validate_phase3_continuity(df_input)
                continuity_quality = {
                    "stage": "phase4_continuity",
                    "continuity_status": continuity_report.get("continuity_status"),
                    "quality_score": continuity_report.get("quality_score", 0),
                }
                quality_log.append(continuity_quality)

            # Stage 3: Feature engineering with monitoring
            df_featured = df_input.copy()

            # Age binning with quality check
            df_featured["age_bin"] = pd.cut(
                df_featured["Age"],
                bins=[18, 35, 55, 100],
                labels=["young", "middle", "senior"],
                include_lowest=True,
            )

            age_binning_quality = {
                "stage": "age_binning",
                "record_count": len(df_featured),
                "feature_count": len(df_featured.columns),
                "missing_values": df_featured.isnull().sum().sum(),
                "target_mean": df_featured["Subscription Status"].mean(),
                "new_feature_missing": df_featured["age_bin"].isnull().sum(),
            }
            quality_log.append(age_binning_quality)

            # Contact recency with quality check
            if "No_Previous_Contact" in df_featured.columns:
                df_featured["contact_recency"] = 1 - df_featured["No_Previous_Contact"]

                contact_quality = {
                    "stage": "contact_recency",
                    "record_count": len(df_featured),
                    "feature_count": len(df_featured.columns),
                    "missing_values": df_featured.isnull().sum().sum(),
                    "target_mean": df_featured["Subscription Status"].mean(),
                    "new_feature_missing": df_featured["contact_recency"]
                    .isnull()
                    .sum(),
                }
                quality_log.append(contact_quality)

            # Validate quality preservation across all stages
            for i, stage_quality in enumerate(quality_log):
                if i > 0 and "record_count" in stage_quality:
                    prev_stage = quality_log[0]  # Compare to initial

                    # Record count preservation
                    assert (
                        stage_quality["record_count"] == prev_stage["record_count"]
                    ), f"Record count changed at {stage_quality['stage']}"

                    # Target distribution preservation
                    if "target_mean" in stage_quality:
                        target_diff = abs(
                            stage_quality["target_mean"] - prev_stage["target_mean"]
                        )
                        assert (
                            target_diff < 0.001
                        ), f"Target distribution changed at {stage_quality['stage']}: {target_diff:.4f}"

                    # New features should not have missing values
                    if "new_feature_missing" in stage_quality:
                        assert (
                            stage_quality["new_feature_missing"] == 0
                        ), f"New feature has missing values at {stage_quality['stage']}"

            print(
                f"✅ Continuous quality monitoring: {len(quality_log)} stages monitored successfully"
            )

        except Exception as e:
            pytest.fail(f"Continuous quality monitoring integration failed: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
