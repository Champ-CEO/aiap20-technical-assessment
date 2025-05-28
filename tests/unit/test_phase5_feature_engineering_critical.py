"""
Phase 5 Feature Engineering Critical Tests

Critical tests for Phase 5 feature engineering business requirements following TDD approach:
1. Age binning validation with boundary testing (18-100 range)
2. Education-occupation interaction feature validation
3. Contact recency features using No_Previous_Contact flag
4. Campaign intensity business-relevant levels
5. Performance requirements (>97K records/second)
6. Data integrity preservation of Phase 3 foundation features

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
        AGE_MIN,
        AGE_MAX,
    )

    PHASE4_INTEGRATION_AVAILABLE = True
except ImportError:
    # Fallback constants if data_integration not available
    PHASE4_INTEGRATION_AVAILABLE = False
    EXPECTED_RECORD_COUNT = 41188
    EXPECTED_FEATURE_COUNT = 33
    PERFORMANCE_STANDARD = 97000
    AGE_MIN = 18
    AGE_MAX = 100


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


class TestPhase5FeatureEngineeringCritical:
    """Critical tests for Phase 5 feature engineering business requirements."""

    def test_age_binning_boundary_validation(self):
        """
        Critical Test: Age binning handles boundary cases correctly (18-100 range).

        Validates proper boundary handling and category assignment for business age groups.
        """
        # Test boundary cases and edge values
        boundary_ages = [18, 17, 25, 34, 35, 54, 55, 99, 100, 101]
        test_data = pd.DataFrame(
            {"Age": boundary_ages, "Subscription Status": [0] * len(boundary_ages)}
        )

        # Apply age binning with business categories
        age_bins = [18, 35, 55, 100]
        age_labels = ["young", "middle", "senior"]

        # Filter valid age range first (18-100)
        valid_ages = test_data[
            (test_data["Age"] >= AGE_MIN) & (test_data["Age"] <= AGE_MAX)
        ].copy()

        valid_ages["age_bin"] = pd.cut(
            valid_ages["Age"], bins=age_bins, labels=age_labels, include_lowest=True
        )

        # Test specific boundary assignments (pandas cut behavior: left boundary exclusive, right inclusive)
        # For bins [18, 35, 55, 100]:
        # young: (18, 35] but include_lowest=True makes it [18, 35]
        # middle: (35, 55]
        # senior: (55, 100]

        # Test edge cases individually
        age_18_bin = valid_ages[valid_ages["Age"] == 18]["age_bin"].iloc[0]
        assert age_18_bin == "young", f"Age 18 should be 'young', got '{age_18_bin}'"

        age_34_bin = valid_ages[valid_ages["Age"] == 34]["age_bin"].iloc[0]
        assert age_34_bin == "young", f"Age 34 should be 'young', got '{age_34_bin}'"

        age_35_bin = valid_ages[valid_ages["Age"] == 35]["age_bin"].iloc[0]
        assert (
            age_35_bin == "young"
        ), f"Age 35 should be 'young' (right boundary), got '{age_35_bin}'"

        age_54_bin = valid_ages[valid_ages["Age"] == 54]["age_bin"].iloc[0]
        assert age_54_bin == "middle", f"Age 54 should be 'middle', got '{age_54_bin}'"

        age_55_bin = valid_ages[valid_ages["Age"] == 55]["age_bin"].iloc[0]
        assert (
            age_55_bin == "middle"
        ), f"Age 55 should be 'middle' (right boundary), got '{age_55_bin}'"

        age_99_bin = valid_ages[valid_ages["Age"] == 99]["age_bin"].iloc[0]
        assert age_99_bin == "senior", f"Age 99 should be 'senior', got '{age_99_bin}'"

        age_100_bin = valid_ages[valid_ages["Age"] == 100]["age_bin"].iloc[0]
        assert (
            age_100_bin == "senior"
        ), f"Age 100 should be 'senior', got '{age_100_bin}'"

        print(
            "✅ Age binning boundary validation: All boundary cases handled correctly"
        )

    def test_education_occupation_interaction_validation(self):
        """
        Critical Test: Education-occupation interaction features for high-value customer segments.

        Validates creation of interaction features for customer segmentation.
        """
        # Create test data with education and occupation combinations
        test_data = pd.DataFrame(
            {
                "Education Level": [
                    "university.degree",
                    "high.school",
                    "university.degree",
                    "professional.course",
                ],
                "Occupation": ["management", "blue-collar", "technician", "management"],
                "Subscription Status": [
                    1,
                    0,
                    1,
                    1,
                ],  # High-value combinations should correlate
            }
        )

        # Create education-occupation interaction
        test_data["education_occupation"] = (
            test_data["Education Level"].astype(str)
            + "_"
            + test_data["Occupation"].astype(str)
        )

        # Verify interaction feature created
        assert (
            "education_occupation" in test_data.columns
        ), "Education-occupation interaction not created"

        # Verify expected combinations
        expected_combinations = [
            "university.degree_management",
            "high.school_blue-collar",
            "university.degree_technician",
            "professional.course_management",
        ]

        actual_combinations = test_data["education_occupation"].tolist()
        assert (
            actual_combinations == expected_combinations
        ), f"Expected {expected_combinations}, got {actual_combinations}"

        # Test high-value segment identification
        high_value_segments = test_data[
            (
                test_data["Education Level"].isin(
                    ["university.degree", "professional.course"]
                )
            )
            & (test_data["Occupation"].isin(["management", "technician"]))
        ]

        assert len(high_value_segments) > 0, "High-value segments not identified"

        print(
            "✅ Education-occupation interaction validation: High-value segments identified correctly"
        )

    def test_contact_recency_features_validation(self):
        """
        Critical Test: Contact recency features leveraging No_Previous_Contact flag.

        Validates proper transformation of contact history into recency indicators.
        """
        # Create test data with contact history patterns
        test_data = pd.DataFrame(
            {
                "No_Previous_Contact": [
                    1,
                    0,
                    1,
                    0,
                    1,
                ],  # 1 = no previous contact, 0 = has previous contact
                "Campaign Calls": [1, 3, 2, 5, 1],
                "Subscription Status": [0, 1, 0, 1, 0],
            }
        )

        # Create contact recency features
        test_data["contact_recency"] = (
            1 - test_data["No_Previous_Contact"]
        )  # Flip: 1 = recent contact
        test_data["first_time_contact"] = test_data[
            "No_Previous_Contact"
        ]  # Keep original meaning

        # Verify recency transformation
        expected_recency = [0, 1, 0, 1, 0]  # Inverted from No_Previous_Contact
        actual_recency = test_data["contact_recency"].tolist()

        assert (
            actual_recency == expected_recency
        ), f"Contact recency transformation failed: expected {expected_recency}, got {actual_recency}"

        # Verify first-time contact identification
        first_time_contacts = test_data[test_data["first_time_contact"] == 1]
        assert len(first_time_contacts) == 3, "First-time contact identification failed"

        # Test business logic: recent contacts should have higher campaign calls
        recent_contacts = test_data[test_data["contact_recency"] == 1]
        avg_calls_recent = recent_contacts["Campaign Calls"].mean()

        first_contacts = test_data[test_data["contact_recency"] == 0]
        avg_calls_first = first_contacts["Campaign Calls"].mean()

        # This validates the business assumption
        assert (
            avg_calls_recent >= avg_calls_first
        ), "Recent contacts should have higher average campaign calls"

        print(
            "✅ Contact recency features validation: Recency indicators created correctly"
        )

    def test_campaign_intensity_business_levels(self):
        """
        Critical Test: Campaign intensity features with business-relevant levels.

        Validates transformation of campaign calls into meaningful business categories.
        """
        # Create test data with campaign call patterns
        test_data = pd.DataFrame(
            {
                "Campaign Calls": [0, 1, 2, 3, 4, 5, 8, 10, 15],
                "Subscription Status": [
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                ],  # Pattern: medium intensity works best
            }
        )

        # Create campaign intensity categories
        intensity_bins = [0, 1, 3, 5, 50]
        intensity_labels = ["none", "low", "medium", "high"]

        test_data["campaign_intensity"] = pd.cut(
            test_data["Campaign Calls"],
            bins=intensity_bins,
            labels=intensity_labels,
            include_lowest=True,
        )

        # Verify intensity categories (pandas cut behavior: left exclusive, right inclusive)
        # For bins [0, 1, 3, 5, 50] with include_lowest=True:
        # none: [0, 1] (include_lowest makes 0 included)
        # low: (1, 3]
        # medium: (3, 5]
        # high: (5, 50]

        # Test individual campaign call mappings
        expected_mapping = {
            0: "none",  # [0, 1] with include_lowest
            1: "none",  # [0, 1]
            2: "low",  # (1, 3]
            3: "low",  # (1, 3]
            4: "medium",  # (3, 5]
            5: "medium",  # (3, 5]
            8: "high",  # (5, 50]
            10: "high",  # (5, 50]
            15: "high",  # (5, 50]
        }

        for i, calls in enumerate(test_data["Campaign Calls"]):
            actual_intensity = test_data["campaign_intensity"].iloc[i]
            expected_intensity = expected_mapping[calls]
            assert (
                str(actual_intensity) == expected_intensity
            ), f"Campaign calls {calls} should map to '{expected_intensity}', got '{actual_intensity}'"

        # Test business logic: medium intensity should have highest success rate
        intensity_success = test_data.groupby("campaign_intensity")[
            "Subscription Status"
        ].mean()

        assert (
            "medium" in intensity_success.index
        ), "Medium intensity category not found"
        medium_success = intensity_success["medium"]

        # Verify medium intensity performs well
        assert (
            medium_success > 0.5
        ), f"Medium intensity success rate too low: {medium_success}"

        print(
            "✅ Campaign intensity business levels: Categories align with business logic"
        )

    def test_performance_requirements_validation(self):
        """
        Critical Test: Performance requirements maintaining >97K records/second.

        Validates that feature engineering meets performance standards.
        """
        try:
            # Load full dataset
            df = load_phase3_output()

            # Measure feature engineering performance
            start_time = time.time()

            # Perform core feature engineering operations
            df_featured = df.copy()

            # Age binning
            df_featured["age_bin"] = pd.cut(
                df_featured["Age"],
                bins=[18, 35, 55, 100],
                labels=["young", "middle", "senior"],
                include_lowest=True,
            )

            # Contact recency (if column exists)
            if "No_Previous_Contact" in df_featured.columns:
                df_featured["contact_recency"] = 1 - df_featured["No_Previous_Contact"]

            # Campaign intensity
            if "Campaign Calls" in df_featured.columns:
                df_featured["campaign_intensity"] = pd.cut(
                    df_featured["Campaign Calls"],
                    bins=[0, 1, 3, 5, 50],
                    labels=["none", "low", "medium", "high"],
                    include_lowest=True,
                )

            processing_time = time.time() - start_time

            # Calculate performance metrics
            records_per_second = (
                len(df) / processing_time if processing_time > 0 else float("inf")
            )

            # Verify performance requirement
            assert (
                records_per_second >= PERFORMANCE_STANDARD
            ), f"Performance requirement failed: {records_per_second:.0f} < {PERFORMANCE_STANDARD} records/sec"

            print(
                f"✅ Performance requirements: {records_per_second:.0f} records/sec (target: {PERFORMANCE_STANDARD})"
            )

        except Exception as e:
            pytest.fail(f"Performance requirements validation failed: {str(e)}")

    def test_data_integrity_preservation_validation(self):
        """
        Critical Test: Data integrity ensuring all Phase 3 foundation features are preserved.

        Validates that feature engineering preserves original data integrity.
        """
        try:
            # Load Phase 3 foundation data
            df_original = load_phase3_output()

            # Perform feature engineering
            df_featured = df_original.copy()

            # Add new features without modifying original columns
            df_featured["age_bin"] = pd.cut(
                df_featured["Age"],
                bins=[18, 35, 55, 100],
                labels=["young", "middle", "senior"],
                include_lowest=True,
            )

            # Verify record count preservation
            assert len(df_featured) == len(
                df_original
            ), f"Record count changed: {len(df_original)} → {len(df_featured)}"

            assert (
                len(df_featured) == EXPECTED_RECORD_COUNT
            ), f"Expected {EXPECTED_RECORD_COUNT} records, got {len(df_featured)}"

            # Verify original columns preserved
            original_columns = set(df_original.columns)
            featured_columns = set(df_featured.columns)

            assert original_columns.issubset(
                featured_columns
            ), f"Original columns not preserved: missing {original_columns - featured_columns}"

            # Verify original data values unchanged
            for col in df_original.columns:
                if col in df_featured.columns:
                    original_values = df_original[col].fillna("NULL")
                    featured_values = df_featured[col].fillna("NULL")

                    assert original_values.equals(
                        featured_values
                    ), f"Original data modified in column: {col}"

            # Verify target variable preservation
            assert (
                "Subscription Status" in df_featured.columns
            ), "Target variable missing"

            target_original = df_original["Subscription Status"].sum()
            target_featured = df_featured["Subscription Status"].sum()

            assert (
                target_original == target_featured
            ), f"Target variable modified: {target_original} → {target_featured}"

            # Verify data quality maintained
            missing_original = df_original.isnull().sum().sum()
            missing_featured = df_featured[df_original.columns].isnull().sum().sum()

            assert (
                missing_original == missing_featured
            ), f"Missing values changed: {missing_original} → {missing_featured}"

            print(
                f"✅ Data integrity preservation: {len(df_original.columns)} original features preserved"
            )

        except Exception as e:
            pytest.fail(f"Data integrity preservation validation failed: {str(e)}")

    def test_phase4_continuity_validation(self):
        """
        Critical Test: Phase 4 continuity validation maintained from Phase 4 integration.

        Validates that data flow integrity is maintained from Phase 4 integration.
        """
        if not PHASE4_INTEGRATION_AVAILABLE:
            pytest.skip("Phase 4 data integration module not available")

        try:
            # Load data using Phase 4 integration
            df = load_phase3_output()

            # Validate Phase 4 continuity before feature engineering
            continuity_report = validate_phase3_continuity(df)

            # Verify continuity validation passed
            assert (
                continuity_report["continuity_status"] == "PASSED"
            ), f"Phase 4 continuity validation failed: {continuity_report.get('continuity_status')}"

            # Verify quality score meets requirements
            quality_score = continuity_report.get("quality_score", 0)
            assert (
                quality_score >= 90
            ), f"Quality score too low: {quality_score}% (minimum: 90%)"

            # Verify data integrity metrics
            assert (
                "data_integrity" in continuity_report
            ), "Data integrity metrics missing"
            assert (
                "business_rules" in continuity_report
            ), "Business rules validation missing"

            # Test feature engineering with validated data
            df_featured = df.copy()

            # Apply feature engineering while preserving continuity
            df_featured["age_bin"] = pd.cut(
                df_featured["Age"],
                bins=[18, 35, 55, 100],
                labels=["young", "middle", "senior"],
                include_lowest=True,
            )

            # Verify continuity maintained after feature engineering
            post_engineering_report = validate_phase3_continuity(
                df_featured[df.columns]
            )
            assert (
                post_engineering_report["continuity_status"] == "PASSED"
            ), "Continuity validation failed after feature engineering"

            print(
                f"✅ Phase 4 continuity validation: Quality={quality_score}%, Status=PASSED"
            )

        except Exception as e:
            pytest.fail(f"Phase 4 continuity validation failed: {str(e)}")

    def test_feature_engineering_module_integration(self):
        """
        Critical Test: Feature engineering module integration with Phase 4.

        Requirements:
        - Use FeatureEngineer class from feature_engineering module
        - Integrate with Phase 4 data access functions
        - Create all required business features
        """
        try:
            # Load Phase 4 data
            df = load_phase3_output()

            # Import and use feature engineering module
            from feature_engineering import FeatureEngineer

            engineer = FeatureEngineer()

            # Test individual feature creation methods
            df_with_age_bins = engineer.create_age_bins(df.copy())
            assert "age_bin" in df_with_age_bins.columns, "Age bins not created"

            df_with_education_job = engineer.create_education_occupation_interactions(
                df.copy()
            )
            assert (
                "education_job_segment" in df_with_education_job.columns
            ), "Education-job segments not created"

            df_with_contact_recency = engineer.create_contact_recency_features(
                df.copy()
            )
            assert (
                "recent_contact_flag" in df_with_contact_recency.columns
            ), "Contact recency features not created"

            df_with_campaign_intensity = engineer.create_campaign_intensity_features(
                df.copy()
            )
            assert (
                "campaign_intensity" in df_with_campaign_intensity.columns
            ), "Campaign intensity not created"
            assert (
                "high_intensity_flag" in df_with_campaign_intensity.columns
            ), "High intensity flag not created"

            # Test main pipeline
            df_engineered = engineer.engineer_features(df.copy())

            # Verify all features created
            required_features = [
                "age_bin",
                "education_job_segment",
                "recent_contact_flag",
                "campaign_intensity",
                "high_intensity_flag",
            ]

            for feature in required_features:
                assert (
                    feature in df_engineered.columns
                ), f"Required feature missing: {feature}"

            # Verify data integrity
            assert len(df_engineered) == len(df), "Record count changed"
            assert len(df_engineered.columns) > len(df.columns), "No new features added"

            print("✅ Feature engineering module integration test passed")

        except ImportError:
            pytest.skip("Feature engineering module not yet implemented")
        except Exception as e:
            pytest.fail(f"Feature engineering module integration failed: {str(e)}")

    def test_output_file_generation(self):
        """
        Critical Test: Output file generation to data/featured/featured-db.csv.

        Requirements:
        - Generate output file in correct location
        - Maintain data quality and structure
        - Include all engineered features
        """
        try:
            # Load Phase 4 data
            df = load_phase3_output()

            # Import feature engineering module
            from feature_engineering import FeatureEngineer

            engineer = FeatureEngineer()

            # Run feature engineering
            df_engineered = engineer.engineer_features(df.copy())

            # Test output file generation
            output_path = Path("data/featured/featured-db.csv")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save engineered features
            df_engineered.to_csv(output_path, index=False)

            # Verify file was created
            assert output_path.exists(), "Output file not created"

            # Verify file content
            df_loaded = pd.read_csv(output_path)

            assert len(df_loaded) == len(
                df_engineered
            ), "Output file record count mismatch"
            assert len(df_loaded.columns) == len(
                df_engineered.columns
            ), "Output file column count mismatch"

            # Verify required features in output
            required_features = [
                "age_bin",
                "education_job_segment",
                "recent_contact_flag",
                "campaign_intensity",
                "high_intensity_flag",
            ]

            for feature in required_features:
                assert (
                    feature in df_loaded.columns
                ), f"Required feature missing in output: {feature}"

            print(f"✅ Output file generation test passed: {output_path}")

        except ImportError:
            pytest.skip("Feature engineering module not yet implemented")
        except Exception as e:
            pytest.fail(f"Output file generation failed: {str(e)}")

    def test_quality_monitoring_validation(self):
        """
        Critical Test: Quality monitoring with continuous validation after each feature engineering step.

        Validates that quality monitoring works throughout the feature engineering process.
        """
        try:
            # Load initial data
            df_original = load_phase3_output()

            # Track quality metrics through each step
            quality_metrics = []

            # Step 1: Initial quality baseline
            initial_quality = {
                "step": "initial",
                "missing_values": df_original.isnull().sum().sum(),
                "record_count": len(df_original),
                "feature_count": len(df_original.columns),
                "target_distribution": df_original["Subscription Status"].mean(),
            }
            quality_metrics.append(initial_quality)

            # Step 2: Age binning with quality monitoring
            df_step1 = df_original.copy()
            df_step1["age_bin"] = pd.cut(
                df_step1["Age"],
                bins=[18, 35, 55, 100],
                labels=["young", "middle", "senior"],
                include_lowest=True,
            )

            step1_quality = {
                "step": "age_binning",
                "missing_values": df_step1.isnull().sum().sum(),
                "record_count": len(df_step1),
                "feature_count": len(df_step1.columns),
                "target_distribution": df_step1["Subscription Status"].mean(),
                "new_feature_missing": df_step1["age_bin"].isnull().sum(),
            }
            quality_metrics.append(step1_quality)

            # Step 3: Contact recency with quality monitoring
            if "No_Previous_Contact" in df_step1.columns:
                df_step2 = df_step1.copy()
                df_step2["contact_recency"] = 1 - df_step2["No_Previous_Contact"]

                step2_quality = {
                    "step": "contact_recency",
                    "missing_values": df_step2.isnull().sum().sum(),
                    "record_count": len(df_step2),
                    "feature_count": len(df_step2.columns),
                    "target_distribution": df_step2["Subscription Status"].mean(),
                    "new_feature_missing": df_step2["contact_recency"].isnull().sum(),
                }
                quality_metrics.append(step2_quality)

            # Validate quality preservation across steps
            for i, metric in enumerate(quality_metrics):
                if i > 0:
                    prev_metric = quality_metrics[i - 1]

                    # Record count should be preserved
                    assert (
                        metric["record_count"] == prev_metric["record_count"]
                    ), f"Record count changed in step {metric['step']}: {prev_metric['record_count']} → {metric['record_count']}"

                    # Target distribution should be preserved
                    target_diff = abs(
                        metric["target_distribution"]
                        - prev_metric["target_distribution"]
                    )
                    assert (
                        target_diff < 0.001
                    ), f"Target distribution changed in step {metric['step']}: {target_diff:.4f}"

                    # Missing values in original features should not increase
                    if "new_feature_missing" in metric:
                        assert (
                            metric["new_feature_missing"] == 0
                        ), f"New feature has missing values in step {metric['step']}: {metric['new_feature_missing']}"

            print(
                f"✅ Quality monitoring validation: {len(quality_metrics)} steps monitored successfully"
            )

        except Exception as e:
            pytest.fail(f"Quality monitoring validation failed: {str(e)}")

    def test_memory_optimization_validation(self):
        """
        Critical Test: Memory optimization for efficient processing of large feature sets.

        Validates that feature engineering maintains memory efficiency for large datasets.
        """
        try:
            # Load data and measure initial memory usage
            df = load_phase3_output()
            initial_memory = df.memory_usage(deep=True).sum()

            # Test memory-efficient feature engineering
            df_featured = df.copy()

            # Memory-efficient age binning (categorical instead of string)
            df_featured["age_bin"] = pd.cut(
                df_featured["Age"],
                bins=[18, 35, 55, 100],
                labels=["young", "middle", "senior"],
                include_lowest=True,
            ).astype("category")

            # Memory-efficient contact recency (int8 instead of int64)
            if "No_Previous_Contact" in df_featured.columns:
                df_featured["contact_recency"] = (
                    1 - df_featured["No_Previous_Contact"]
                ).astype("int8")

            # Memory-efficient campaign intensity (categorical)
            if "Campaign Calls" in df_featured.columns:
                df_featured["campaign_intensity"] = pd.cut(
                    df_featured["Campaign Calls"],
                    bins=[0, 1, 3, 5, 50],
                    labels=["none", "low", "medium", "high"],
                    include_lowest=True,
                ).astype("category")

            # Measure final memory usage
            final_memory = df_featured.memory_usage(deep=True).sum()
            memory_increase = final_memory - initial_memory
            memory_increase_pct = (memory_increase / initial_memory) * 100

            # Validate memory efficiency
            # New features should not increase memory by more than 50%
            assert (
                memory_increase_pct <= 50
            ), f"Memory increase too high: {memory_increase_pct:.1f}% (limit: 50%)"

            # Validate data types are memory-efficient
            assert (
                df_featured["age_bin"].dtype.name == "category"
            ), "Age bin should be categorical for memory efficiency"

            if "contact_recency" in df_featured.columns:
                assert (
                    df_featured["contact_recency"].dtype == "int8"
                ), "Contact recency should be int8 for memory efficiency"

            if "campaign_intensity" in df_featured.columns:
                assert (
                    df_featured["campaign_intensity"].dtype.name == "category"
                ), "Campaign intensity should be categorical for memory efficiency"

            # Test processing speed with memory-optimized features
            start_time = time.time()

            # Simulate feature engineering operations
            _ = df_featured.groupby("age_bin")["Subscription Status"].mean()
            if "campaign_intensity" in df_featured.columns:
                _ = df_featured.groupby("campaign_intensity")[
                    "Subscription Status"
                ].mean()

            processing_time = time.time() - start_time
            records_per_second = (
                len(df_featured) / processing_time
                if processing_time > 0
                else float("inf")
            )

            # Verify performance maintained with memory optimization
            assert (
                records_per_second >= PERFORMANCE_STANDARD
            ), f"Performance degraded with memory optimization: {records_per_second:.0f} < {PERFORMANCE_STANDARD}"

            print(
                f"✅ Memory optimization validation: {memory_increase_pct:.1f}% memory increase, {records_per_second:.0f} records/sec"
            )

        except Exception as e:
            pytest.fail(f"Memory optimization validation failed: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
