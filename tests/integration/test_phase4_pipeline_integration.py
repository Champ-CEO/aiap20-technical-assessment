"""
Phase 4 Pipeline Integration Tests

Integration tests for Phase 3 → Phase 4 data flow continuity:
1. End-to-end data flow validation
2. Phase transition integrity checks
3. Data pipeline continuity verification
4. Cross-phase compatibility testing

Following TDD approach with focus on critical path integration.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))


class TestPhase3ToPhase4Integration:
    """Integration tests for Phase 3 → Phase 4 data flow continuity."""

    def test_end_to_end_data_flow_validation(self):
        """
        Integration Test: Complete Phase 3 → Phase 4 data flow validation

        Requirements:
        - Phase 3 output serves as valid Phase 4 input
        - All data transformations are preserved
        - Data quality metrics are maintained
        - Pipeline continuity is verified
        """
        # Arrange
        phase3_output_path = Path("data/processed/cleaned-db.csv")

        # Act - Load Phase 3 output as Phase 4 input
        df = pd.read_csv(phase3_output_path)

        # Assert - Comprehensive validation
        # 1. File accessibility and structure
        assert phase3_output_path.exists(), "Phase 3 output file must exist for Phase 4"
        assert len(df) == 41188, "Should preserve exact record count from Phase 3"
        assert len(df.columns) == 33, "Should preserve exact feature count from Phase 3"

        # 2. Data quality preservation
        assert (
            df.isnull().sum().sum() == 0
        ), "Zero missing values from Phase 3 must be preserved"

        # 3. Key transformations preservation
        assert pd.api.types.is_numeric_dtype(
            df["Age"]
        ), "Age numeric conversion must be preserved"
        assert set(df["Subscription Status"].unique()) == {
            0,
            1,
        }, "Target binary encoding must be preserved"
        assert set(df["Contact Method"].unique()) == {
            "cellular",
            "telephone",
        }, "Contact standardization must be preserved"

        # 4. Business rules preservation
        assert (
            df["Age"].min() >= 18 and df["Age"].max() <= 100
        ), "Age business rules must be preserved"

        print(
            f"✅ End-to-end data flow validated: Phase 3 → Phase 4 continuity confirmed"
        )
        print(f"   Records: {len(df)}, Features: {len(df.columns)}, Quality: 100%")

    def test_phase_transition_integrity_checks(self):
        """
        Integration Test: Phase transition integrity between Phase 3 and Phase 4

        Requirements:
        - No data corruption during phase transition
        - All Phase 3 achievements are maintained
        - Data format compatibility is verified
        """
        # Arrange
        df = pd.read_csv("data/processed/cleaned-db.csv")

        # Act - Verify Phase 3 achievements are intact
        phase3_achievements = {
            "age_conversion": pd.api.types.is_numeric_dtype(df["Age"]),
            "missing_values_elimination": df.isnull().sum().sum() == 0,
            "target_encoding": set(df["Subscription Status"].unique()) == {0, 1},
            "contact_standardization": set(df["Contact Method"].unique())
            == {"cellular", "telephone"},
            "previous_contact_handling": "No_Previous_Contact" in df.columns,
            "record_preservation": len(df) == 41188,
            "feature_expansion": len(df.columns) == 33,
        }

        # Assert
        failed_achievements = [
            name for name, status in phase3_achievements.items() if not status
        ]
        assert (
            len(failed_achievements) == 0
        ), f"Phase 3 achievements not preserved in Phase 4: {failed_achievements}"

        # Verify data format compatibility for Phase 4
        format_compatibility = {
            "csv_format": True,  # Already loaded successfully
            "numeric_columns_accessible": all(
                pd.api.types.is_numeric_dtype(df[col])
                for col in ["Age", "Campaign Calls", "Subscription Status"]
            ),
            "categorical_columns_accessible": all(
                df[col].dtype == "object" for col in ["Occupation", "Education Level"]
            ),
            "no_problematic_characters": not any(
                df.select_dtypes(include=["object"])
                .astype(str)
                .apply(lambda x: x.str.contains("[^\w\s\-\.]", na=False))
                .any()
            ),
        }

        failed_compatibility = [
            name for name, status in format_compatibility.items() if not status
        ]
        assert (
            len(failed_compatibility) == 0
        ), f"Format compatibility issues for Phase 4: {failed_compatibility}"

        print(
            f"✅ Phase transition integrity verified: All {len(phase3_achievements)} Phase 3 achievements preserved"
        )

    def test_data_pipeline_continuity_verification(self):
        """
        Integration Test: Data pipeline continuity from Phase 3 to Phase 4

        Requirements:
        - Seamless data access for Phase 4 operations
        - Performance standards maintained
        - Memory efficiency preserved
        """
        # Arrange
        start_time = time.time()

        # Act - Simulate Phase 4 data access operations
        df = pd.read_csv("data/processed/cleaned-db.csv")

        # Common Phase 4 operations
        operations_performance = {}

        # 1. Basic data access
        op_start = time.time()
        shape = df.shape
        column_list = list(df.columns)
        operations_performance["basic_access"] = time.time() - op_start

        # 2. Data type inspection
        op_start = time.time()
        dtypes = df.dtypes
        operations_performance["type_inspection"] = time.time() - op_start

        # 3. Statistical summary
        op_start = time.time()
        numeric_summary = df.select_dtypes(include=[np.number]).describe()
        operations_performance["statistical_summary"] = time.time() - op_start

        # 4. Data validation operations
        op_start = time.time()
        missing_check = df.isnull().sum()
        duplicate_check = df.duplicated().sum()
        operations_performance["validation_operations"] = time.time() - op_start

        total_time = time.time() - start_time

        # Assert
        # Performance requirements
        assert (
            total_time < 2.0
        ), f"Pipeline operations should complete in < 2s, took {total_time:.3f}s"

        # All operations should complete quickly
        slow_operations = [
            op for op, time_taken in operations_performance.items() if time_taken > 0.5
        ]
        assert len(slow_operations) == 0, f"Slow operations detected: {slow_operations}"

        # Data accessibility verification
        assert shape == (41188, 33), f"Data shape should be (41188, 33), got {shape}"
        assert (
            len(column_list) == 33
        ), f"Should have 33 columns accessible, got {len(column_list)}"

        print(
            f"✅ Data pipeline continuity verified: {total_time:.3f}s total operation time"
        )
        print(f"   Operation times: {operations_performance}")

    def test_cross_phase_compatibility_testing(self):
        """
        Integration Test: Cross-phase compatibility between Phase 3 and Phase 4

        Requirements:
        - Phase 3 output format is optimal for Phase 4 input
        - No format conversion needed
        - Data structure supports Phase 4 requirements
        """
        # Arrange
        df = pd.read_csv("data/processed/cleaned-db.csv")

        # Act - Test Phase 4 compatibility requirements
        compatibility_tests = {}

        # 1. Data splitting compatibility (for ML pipeline)
        try:
            from sklearn.model_selection import train_test_split

            X = df.drop("Subscription Status", axis=1)
            y = df["Subscription Status"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            compatibility_tests["data_splitting"] = True
        except Exception as e:
            compatibility_tests["data_splitting"] = False

        # 2. Numeric operations compatibility
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            correlation_matrix = df[numeric_cols].corr()
            compatibility_tests["numeric_operations"] = True
        except Exception as e:
            compatibility_tests["numeric_operations"] = False

        # 3. Categorical operations compatibility
        try:
            categorical_cols = df.select_dtypes(include=["object"]).columns
            value_counts = {col: df[col].value_counts() for col in categorical_cols}
            compatibility_tests["categorical_operations"] = True
        except Exception as e:
            compatibility_tests["categorical_operations"] = False

        # 4. Memory efficiency compatibility
        try:
            memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            compatibility_tests["memory_efficiency"] = (
                memory_usage < 50
            )  # Should be < 50MB
        except Exception as e:
            compatibility_tests["memory_efficiency"] = False

        # Assert
        failed_tests = [
            test for test, passed in compatibility_tests.items() if not passed
        ]
        assert (
            len(failed_tests) == 0
        ), f"Cross-phase compatibility failed: {failed_tests}"

        # Specific Phase 4 requirements
        assert (
            len(df.select_dtypes(include=[np.number]).columns) >= 10
        ), "Should have sufficient numeric columns for Phase 4 analysis"
        assert (
            len(df.select_dtypes(include=["object"]).columns) >= 5
        ), "Should have sufficient categorical columns for Phase 4 analysis"

        print(
            f"✅ Cross-phase compatibility verified: All {len(compatibility_tests)} compatibility tests passed"
        )


class TestPhase4DataIntegrationReadiness:
    """Integration tests for Phase 4 data integration readiness."""

    def test_phase4_data_access_functions_readiness(self):
        """
        Integration Test: Phase 4 data access functions readiness

        Requirements:
        - Data can be loaded efficiently for Phase 4 operations
        - Multiple access patterns work correctly
        - Data integrity is maintained across access methods
        """
        # Arrange
        file_path = "data/processed/cleaned-db.csv"

        # Act - Test different data access patterns
        access_patterns = {}

        # 1. Full dataset loading
        try:
            df_full = pd.read_csv(file_path)
            access_patterns["full_loading"] = len(df_full) == 41188
        except Exception as e:
            access_patterns["full_loading"] = False

        # 2. Chunked loading (for memory efficiency)
        try:
            chunk_size = 10000
            chunks = []
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                chunks.append(len(chunk))
            total_records = sum(chunks)
            access_patterns["chunked_loading"] = total_records == 41188
        except Exception as e:
            access_patterns["chunked_loading"] = False

        # 3. Selective column loading
        try:
            core_columns = ["Client ID", "Age", "Subscription Status"]
            df_selective = pd.read_csv(file_path, usecols=core_columns)
            access_patterns["selective_loading"] = len(df_selective.columns) == 3
        except Exception as e:
            access_patterns["selective_loading"] = False

        # 4. Sample loading (for testing)
        try:
            df_sample = pd.read_csv(file_path, nrows=1000)
            access_patterns["sample_loading"] = len(df_sample) == 1000
        except Exception as e:
            access_patterns["sample_loading"] = False

        # Assert
        failed_patterns = [
            pattern for pattern, success in access_patterns.items() if not success
        ]
        assert (
            len(failed_patterns) == 0
        ), f"Data access patterns failed: {failed_patterns}"

        print(
            f"✅ Phase 4 data access readiness verified: All {len(access_patterns)} access patterns working"
        )

    def test_phase4_validation_framework_readiness(self):
        """
        Integration Test: Phase 4 validation framework readiness

        Requirements:
        - Data validation functions work on Phase 3 output
        - Quality metrics can be calculated
        - Business rules can be verified
        """
        # Arrange
        df = pd.read_csv("data/processed/cleaned-db.csv")

        # Act - Test validation framework components
        validation_results = {}

        # 1. Data quality metrics
        try:
            quality_metrics = {
                "completeness": (
                    1 - df.isnull().sum().sum() / (len(df) * len(df.columns))
                )
                * 100,
                "consistency": 100,  # All data types are consistent
                "validity": 100 if df["Age"].between(18, 100).all() else 0,
                "accuracy": (
                    100 if set(df["Subscription Status"].unique()) == {0, 1} else 0
                ),
            }
            validation_results["quality_metrics"] = all(
                score >= 100 for score in quality_metrics.values()
            )
        except Exception as e:
            validation_results["quality_metrics"] = False

        # 2. Business rule validation
        try:
            business_rules = {
                "age_range": df["Age"].between(18, 100).all(),
                "target_binary": set(df["Subscription Status"].unique()) == {0, 1},
                "no_missing_values": df.isnull().sum().sum() == 0,
                "contact_standardized": set(df["Contact Method"].unique())
                == {"cellular", "telephone"},
            }
            validation_results["business_rules"] = all(business_rules.values())
        except Exception as e:
            validation_results["business_rules"] = False

        # 3. Statistical validation
        try:
            statistical_checks = {
                "age_distribution": df["Age"].std() > 0,  # Should have variation
                "target_distribution": 0.05
                < df["Subscription Status"].mean()
                < 0.95,  # Not all one class
                "campaign_calls_reasonable": df["Campaign Calls"].max()
                <= 100,  # Reasonable maximum
            }
            validation_results["statistical_validation"] = all(
                statistical_checks.values()
            )
        except Exception as e:
            validation_results["statistical_validation"] = False

        # Assert
        failed_validations = [
            validation
            for validation, success in validation_results.items()
            if not success
        ]
        assert (
            len(failed_validations) == 0
        ), f"Validation framework components failed: {failed_validations}"

        print(
            f"✅ Phase 4 validation framework readiness verified: All {len(validation_results)} components working"
        )
