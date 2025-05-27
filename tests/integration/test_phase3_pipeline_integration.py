"""
Phase 3 Pipeline Integration Tests

End-to-end integration tests for the complete Phase 3 data cleaning and preprocessing pipeline.
Tests the integration between all Phase 3 components and validates the complete workflow.

Integration Points:
1. Data loading → Cleaning → Validation → Output
2. Component interactions (Cleaner + Validator + Feature Engineer)
3. End-to-end pipeline with realistic EDA data
4. Output validation for Phase 4 readiness
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import tempfile
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from preprocessing.data_cleaner import BankingDataCleaner
from preprocessing.validation import DataValidator
from preprocessing.feature_engineering import FeatureEngineer


class TestPhase3PipelineIntegration:
    """
    Test complete Phase 3 pipeline integration.

    Tests the full workflow from raw EDA data to cleaned output ready for Phase 4.
    """

    def test_complete_pipeline_workflow(
        self, phase3_raw_sample_data, phase3_expected_cleaned_schema
    ):
        """Test complete Phase 3 pipeline workflow."""
        # Initialize components
        cleaner = BankingDataCleaner()
        validator = DataValidator()
        engineer = FeatureEngineer()

        # Use EDA sample data
        raw_data = phase3_raw_sample_data.copy()
        initial_shape = raw_data.shape

        # Step 1: Validate raw data
        raw_validation = validator.generate_validation_report(raw_data)
        assert raw_validation is not None, "Should generate raw data validation report"

        # Step 2: Clean data
        cleaned_data = cleaner.clean_banking_data(raw_data)
        assert cleaned_data is not None, "Should produce cleaned data"
        assert len(cleaned_data) == len(raw_data), "Should preserve all records"

        # Step 3: Validate cleaned data
        cleaned_validation = validator.generate_validation_report(cleaned_data)
        assert (
            cleaned_validation is not None
        ), "Should generate cleaned data validation report"

        # Step 4: Engineer features
        final_data = engineer.engineer_features(cleaned_data)
        assert final_data is not None, "Should produce feature-engineered data"

        # Step 5: Final validation
        final_validation = validator.generate_validation_report(final_data)
        assert final_validation is not None, "Should generate final validation report"

        # Verify pipeline success
        assert len(final_data) == len(
            raw_data
        ), "Should preserve all records through pipeline"
        assert (
            final_data.isna().sum().sum() == 0
        ), "Should have no missing values in final output"

    def test_component_interaction_data_flow(self, phase3_raw_sample_data):
        """Test data flow between Phase 3 components."""
        # Initialize components
        cleaner = BankingDataCleaner()
        validator = DataValidator()
        engineer = FeatureEngineer()

        # Use EDA sample data
        raw_data = phase3_raw_sample_data.copy()

        # Test Cleaner → Validator interaction
        cleaned_data = cleaner.clean_banking_data(raw_data)
        validation_result = validator.generate_validation_report(cleaned_data)

        # Verify cleaner output is valid for validator
        assert (
            validation_result.get("overall_quality_score", 0) > 80
        ), "Cleaned data should have high quality score"

        # Test Validator → Engineer interaction
        if validation_result.get("overall_quality_score", 0) >= 80:
            engineered_data = engineer.engineer_features(cleaned_data)

            # Verify engineer can process validated data
            assert len(engineered_data) == len(
                cleaned_data
            ), "Feature engineer should preserve record count"
            assert (
                engineered_data.columns.tolist() != cleaned_data.columns.tolist()
            ), "Feature engineer should add new features"

        # Test Engineer → Validator interaction
        final_validation = validator.generate_validation_report(engineered_data)
        assert (
            final_validation.get("overall_quality_score", 0) >= 90
        ), "Final engineered data should have very high quality score"

    def test_pipeline_error_handling_and_recovery(self, phase3_raw_sample_data):
        """Test pipeline error handling and recovery mechanisms."""
        cleaner = BankingDataCleaner()
        validator = DataValidator()

        # Test with problematic data
        problematic_data = phase3_raw_sample_data.copy()

        # Introduce various data issues
        problematic_data.loc[0, "Age"] = "completely invalid"
        problematic_data.loc[1, "Campaign Calls"] = -999999
        problematic_data.loc[2, "Contact Method"] = "unknown_method"

        # Pipeline should handle errors gracefully
        try:
            cleaned_data = cleaner.clean_banking_data(problematic_data)
            pipeline_success = True
        except Exception as e:
            pipeline_success = False
            pytest.fail(
                f"Pipeline should handle errors gracefully, but failed with: {str(e)}"
            )

        assert pipeline_success, "Pipeline should complete despite data issues"

        # Verify error recovery
        if pipeline_success:
            validation_result = validator.generate_validation_report(cleaned_data)
            assert (
                validation_result.get("overall_quality_score", 0) > 70
            ), "Pipeline should recover from errors and maintain reasonable quality"

    def test_pipeline_performance_with_realistic_data_size(
        self, phase3_raw_sample_data
    ):
        """Test pipeline performance with realistic data size."""
        import time

        # Scale up sample data to simulate realistic size
        raw_data = phase3_raw_sample_data.copy()

        # Replicate data to simulate larger dataset (but keep manageable for testing)
        scaled_data = pd.concat([raw_data] * 10, ignore_index=True)
        scaled_data["Client ID"] = range(1, len(scaled_data) + 1)  # Ensure unique IDs

        # Initialize components
        cleaner = BankingDataCleaner()
        validator = DataValidator()

        # Measure pipeline performance
        start_time = time.time()

        # Execute pipeline
        cleaned_data = cleaner.clean_banking_data(scaled_data)
        validation_result = validator.generate_validation_report(cleaned_data)

        end_time = time.time()
        processing_time = end_time - start_time

        # Verify performance benchmarks
        records_processed = len(scaled_data)
        records_per_second = (
            records_processed / processing_time if processing_time > 0 else float("inf")
        )

        assert (
            processing_time < 30.0
        ), f"Pipeline should complete within 30 seconds for {records_processed} records"
        assert (
            records_per_second > 10
        ), f"Should process >10 records/second, achieved {records_per_second:.1f}"

        # Verify quality is maintained at scale
        assert (
            validation_result.get("overall_quality_score", 0) >= 90
        ), "Quality should be maintained even with larger datasets"


class TestPhase3OutputValidation:
    """
    Test Phase 3 output validation for Phase 4 readiness.

    Validates that the Phase 3 output meets all requirements for Phase 4 feature engineering.
    """

    def test_phase4_input_requirements(
        self, phase3_raw_sample_data, phase3_expected_cleaned_schema
    ):
        """Test that Phase 3 output meets Phase 4 input requirements."""
        # Execute complete Phase 3 pipeline
        cleaner = BankingDataCleaner()
        engineer = FeatureEngineer()

        raw_data = phase3_raw_sample_data.copy()
        cleaned_data = cleaner.clean_banking_data(raw_data)
        final_data = engineer.engineer_features(cleaned_data)

        schema = phase3_expected_cleaned_schema

        # Verify Phase 4 readiness checklist
        phase4_requirements = {
            "no_missing_values": final_data.isna().sum().sum() == 0,
            "numeric_target": pd.api.types.is_numeric_dtype(
                final_data["Subscription Status"]
            ),
            "numeric_age": pd.api.types.is_numeric_dtype(final_data["Age"]),
            "standardized_contact_methods": len(final_data["Contact Method"].unique())
            <= 3,
            "valid_age_range": (
                final_data["Age"].min() >= 18 and final_data["Age"].max() <= 100
            ),
            "binary_target": set(final_data["Subscription Status"].unique()).issubset(
                {0, 1}
            ),
            "sufficient_features": len(final_data.columns)
            >= 15,  # Should have added features
            "preserved_records": len(final_data) == len(raw_data),
        }

        # Verify all requirements are met
        unmet_requirements = [
            req for req, met in phase4_requirements.items() if not met
        ]
        assert (
            len(unmet_requirements) == 0
        ), f"Phase 4 requirements not met: {unmet_requirements}"

    def test_output_file_format_compatibility(self, phase3_raw_sample_data, tmp_path):
        """Test that output file format is compatible with Phase 4."""
        # Execute complete Phase 3 pipeline
        cleaner = BankingDataCleaner()
        engineer = FeatureEngineer()

        raw_data = phase3_raw_sample_data.copy()
        cleaned_data = cleaner.clean_banking_data(raw_data)
        final_data = engineer.engineer_features(cleaned_data)

        # Test CSV output format (required for Phase 4)
        output_file = tmp_path / "test_cleaned_data.csv"

        try:
            final_data.to_csv(output_file, index=False)
            csv_save_success = True
        except Exception as e:
            csv_save_success = False
            pytest.fail(f"Failed to save as CSV: {str(e)}")

        assert csv_save_success, "Should be able to save as CSV for Phase 4"

        # Test CSV loading (Phase 4 will need to load this)
        try:
            loaded_data = pd.read_csv(output_file)
            csv_load_success = True
        except Exception as e:
            csv_load_success = False
            pytest.fail(f"Failed to load CSV: {str(e)}")

        assert csv_load_success, "CSV should be loadable by Phase 4"

        # Verify data integrity after save/load cycle
        assert len(loaded_data) == len(final_data), "Record count should be preserved"
        assert list(loaded_data.columns) == list(
            final_data.columns
        ), "Columns should be preserved"
        assert (
            loaded_data.isna().sum().sum() == 0
        ), "No missing values should be introduced"

    def test_data_continuity_documentation(self, phase3_raw_sample_data):
        """Test data continuity documentation for Phase 4."""
        # Execute complete Phase 3 pipeline
        cleaner = BankingDataCleaner()
        validator = DataValidator()
        engineer = FeatureEngineer()

        raw_data = phase3_raw_sample_data.copy()

        # Document data flow
        data_flow_log = {
            "phase2_input": {
                "shape": raw_data.shape,
                "missing_values": raw_data.isna().sum().sum(),
                "data_types": raw_data.dtypes.to_dict(),
            }
        }

        # Step 1: Cleaning
        cleaned_data = cleaner.clean_banking_data(raw_data)
        data_flow_log["phase3_cleaning"] = {
            "shape": cleaned_data.shape,
            "missing_values": cleaned_data.isna().sum().sum(),
            "cleaning_stats": cleaner.cleaning_stats,
        }

        # Step 2: Feature Engineering
        final_data = engineer.engineer_features(cleaned_data)
        data_flow_log["phase3_feature_engineering"] = {
            "shape": final_data.shape,
            "missing_values": final_data.isna().sum().sum(),
            "new_features": len(final_data.columns) - len(cleaned_data.columns),
        }

        # Step 3: Final Validation
        validation_result = validator.generate_validation_report(final_data)
        data_flow_log["phase3_output"] = {
            "quality_score": validation_result.get("overall_quality_score", 0),
            "phase4_ready": validation_result.get("overall_quality_score", 0) >= 90,
        }

        # Verify data continuity
        assert (
            data_flow_log["phase3_cleaning"]["shape"][0]
            == data_flow_log["phase2_input"]["shape"][0]
        ), "Record count should be preserved during cleaning"

        assert (
            data_flow_log["phase3_feature_engineering"]["shape"][0]
            == data_flow_log["phase3_cleaning"]["shape"][0]
        ), "Record count should be preserved during feature engineering"

        assert data_flow_log["phase3_output"][
            "phase4_ready"
        ], "Output should be ready for Phase 4"

        # Verify improvement metrics
        assert (
            data_flow_log["phase3_cleaning"]["missing_values"]
            <= data_flow_log["phase2_input"]["missing_values"]
        ), "Missing values should be reduced or eliminated"

        assert (
            data_flow_log["phase3_feature_engineering"]["new_features"] > 0
        ), "Should add new features during feature engineering"


class TestPhase3BusinessValidation:
    """
    Test business validation for Phase 3 output.

    Validates that the cleaned and processed data supports business objectives.
    """

    def test_marketing_campaign_analysis_readiness(self, phase3_raw_sample_data):
        """Test readiness for marketing campaign analysis."""
        # Execute complete Phase 3 pipeline
        cleaner = BankingDataCleaner()
        engineer = FeatureEngineer()

        raw_data = phase3_raw_sample_data.copy()
        cleaned_data = cleaner.clean_banking_data(raw_data)
        final_data = engineer.engineer_features(cleaned_data)

        # Test marketing analysis capabilities
        marketing_analysis_tests = {
            "customer_segmentation": {
                "age_numeric": pd.api.types.is_numeric_dtype(final_data["Age"]),
                "age_groups_available": "Age_Group" in final_data.columns
                or pd.api.types.is_numeric_dtype(final_data["Age"]),
                "occupation_categories": final_data["Occupation"].nunique() > 1,
            },
            "campaign_optimization": {
                "contact_method_clean": final_data["Contact Method"].nunique() <= 3,
                "campaign_calls_valid": final_data["Campaign Calls"].min() >= 0,
                "previous_contact_handled": "No_Previous_Contact" in final_data.columns
                or final_data["Previous Contact Days"].nunique() > 1,
            },
            "subscription_prediction": {
                "target_binary": set(
                    final_data["Subscription Status"].unique()
                ).issubset({0, 1}),
                "target_balanced": 0.05
                <= final_data["Subscription Status"].mean()
                <= 0.95,
                "features_available": len(final_data.columns) >= 15,
            },
        }

        # Verify all marketing analysis requirements
        for analysis_type, requirements in marketing_analysis_tests.items():
            unmet_requirements = [req for req, met in requirements.items() if not met]
            assert (
                len(unmet_requirements) == 0
            ), f"Marketing analysis '{analysis_type}' requirements not met: {unmet_requirements}"

    def test_business_rule_compliance(
        self, phase3_raw_sample_data, phase3_validation_rules
    ):
        """Test compliance with business rules."""
        # Execute complete Phase 3 pipeline
        cleaner = BankingDataCleaner()
        engineer = FeatureEngineer()

        raw_data = phase3_raw_sample_data.copy()
        cleaned_data = cleaner.clean_banking_data(raw_data)
        final_data = engineer.engineer_features(cleaned_data)

        rules = phase3_validation_rules["business_validation_rules"]

        # Test business rule compliance
        business_compliance = {
            "age_business_range": (
                final_data["Age"].min() >= rules["age_business_min"]
                and final_data["Age"].max() <= rules["age_business_max"]
            ),
            "campaign_calls_realistic": (
                final_data["Campaign Calls"].min() >= rules["campaign_calls_min"]
                and final_data["Campaign Calls"].max() <= rules["campaign_calls_max"]
            ),
            "subscription_rate_reasonable": (
                abs(
                    final_data["Subscription Status"].mean()
                    - rules["subscription_rate_expected"]
                )
                <= 0.15
            ),
            "data_completeness": final_data.isna().sum().sum() == 0,
            "data_consistency": len(final_data) == len(raw_data),
        }

        # Verify business rule compliance
        non_compliant_rules = [
            rule for rule, compliant in business_compliance.items() if not compliant
        ]
        assert (
            len(non_compliant_rules) == 0
        ), f"Business rules not compliant: {non_compliant_rules}"
