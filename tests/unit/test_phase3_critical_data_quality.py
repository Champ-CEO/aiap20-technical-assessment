"""
Phase 3 Critical Data Quality Tests (Priority 1)

Tests for the most critical data quality issues identified in EDA:
1. Age conversion verification: Text to numeric with edge cases
2. Missing value handling validation: 28,935 missing values strategy
3. Special value cleaning tests: 12,008 'unknown' values handling

Testing Strategy: Use EDA-identified patterns for realistic test cases
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from preprocessing.data_cleaner import BankingDataCleaner


class TestAgeConversionVerification:
    """
    Test age conversion from text to numeric format.

    EDA Finding: Age stored as text format ('57 years', '55 years', etc.)
    Requirement: Convert to numeric with validation (18-100 range)
    """

    def test_valid_age_text_conversion(self, phase3_raw_sample_data):
        """Test conversion of valid age text formats to numeric."""
        cleaner = BankingDataCleaner()

        # Create test data with valid age formats
        test_data = pd.DataFrame(
            {"Age": ["25 years", "45 years", "67 years", "30 years", "55 years"]}
        )

        # Apply age cleaning
        cleaned_data = cleaner.clean_age_column(test_data)

        # Verify conversion
        assert cleaned_data["Age"].dtype in [
            "int64",
            "float64",
        ], "Age should be numeric after conversion"
        assert all(
            cleaned_data["Age"] == [25, 45, 67, 30, 55]
        ), "Age values should match extracted numbers"
        assert cleaned_data["Age"].isna().sum() == 0, "No valid ages should become NaN"

    def test_invalid_age_format_handling(self):
        """Test handling of invalid age formats."""
        cleaner = BankingDataCleaner()

        # Test data with invalid formats (from EDA edge cases)
        test_data = pd.DataFrame(
            {"Age": ["invalid age", "unknown", "", "abc years", "years old"]}
        )

        # Apply age cleaning
        cleaned_data = cleaner.clean_age_column(test_data)

        # Verify invalid formats are handled appropriately
        assert cleaned_data["Age"].dtype in [
            "int64",
            "float64",
        ], "Age column should be numeric"
        # Invalid formats should either be converted to NaN or handled by business rules
        assert cleaned_data["Age"].isna().sum() > 0 or all(
            cleaned_data["Age"] >= 18
        ), "Invalid ages should be handled"

    def test_age_outlier_capping(self, phase3_validation_rules):
        """Test age outlier capping based on business rules."""
        cleaner = BankingDataCleaner()
        rules = phase3_validation_rules["age_conversion_rules"]

        # Test data with outliers (from EDA findings)
        test_data = pd.DataFrame(
            {"Age": ["150 years", "17 years", "200 years", "10 years", "0 years"]}
        )

        # Apply age cleaning
        cleaned_data = cleaner.clean_age_column(test_data)

        # Verify outliers are capped to business rules
        min_age = rules["min_age"]
        max_age = rules["max_age"]

        valid_ages = cleaned_data["Age"].dropna()
        assert all(valid_ages >= min_age), f"All ages should be >= {min_age}"
        assert all(valid_ages <= max_age), f"All ages should be <= {max_age}"

    def test_age_conversion_with_eda_sample(self, phase3_raw_sample_data):
        """Test age conversion with realistic EDA sample data."""
        cleaner = BankingDataCleaner()

        # Use EDA-based sample data
        sample_data = phase3_raw_sample_data.copy()

        # Apply age cleaning
        cleaned_data = cleaner.clean_age_column(sample_data)

        # Verify conversion results
        assert cleaned_data["Age"].dtype in [
            "int64",
            "float64",
        ], "Age should be numeric"

        # Check that valid ages are properly converted
        valid_ages = cleaned_data["Age"].dropna()
        if len(valid_ages) > 0:
            assert all(valid_ages >= 18), "All valid ages should be >= 18"
            assert all(valid_ages <= 100), "All valid ages should be <= 100"


class TestMissingValueHandlingValidation:
    """
    Test missing value handling strategy.

    EDA Finding: 28,935 missing values total
    - Housing Loan: 60.2% missing
    - Personal Loan: 10.1% missing
    Strategy: Domain-specific imputation with 'Information Not Available'
    """

    def test_housing_loan_missing_value_handling(
        self, phase3_raw_sample_data, phase3_validation_rules
    ):
        """Test Housing Loan missing value handling (60.2% missing from EDA)."""
        cleaner = BankingDataCleaner()
        rules = phase3_validation_rules["missing_value_rules"]

        # Use EDA sample data with realistic missing patterns
        sample_data = phase3_raw_sample_data.copy()

        # Verify initial missing percentage is close to EDA finding
        initial_missing_pct = sample_data["Housing Loan"].isna().sum() / len(
            sample_data
        )
        assert (
            initial_missing_pct >= 0.5
        ), f"Housing Loan missing rate should be high (EDA: 60.2%), got {initial_missing_pct:.1%}"

        # Apply missing value handling
        cleaned_data = cleaner.handle_missing_values(sample_data)

        # Verify missing values are handled
        assert (
            cleaned_data["Housing Loan"].isna().sum() == 0
        ), "No missing values should remain after cleaning"

        # Verify imputation strategy
        imputation_value = rules["imputation_strategy"]
        imputed_values = cleaned_data["Housing Loan"].value_counts()
        assert (
            imputation_value in imputed_values.index
        ), f"Should contain imputation value: {imputation_value}"

    def test_personal_loan_missing_value_handling(
        self, phase3_raw_sample_data, phase3_validation_rules
    ):
        """Test Personal Loan missing value handling (10.1% missing from EDA)."""
        cleaner = BankingDataCleaner()
        rules = phase3_validation_rules["missing_value_rules"]

        # Use EDA sample data
        sample_data = phase3_raw_sample_data.copy()

        # Verify initial missing percentage
        initial_missing_pct = sample_data["Personal Loan"].isna().sum() / len(
            sample_data
        )
        assert (
            initial_missing_pct >= 0.05
        ), f"Personal Loan missing rate should be moderate (EDA: 10.1%), got {initial_missing_pct:.1%}"

        # Apply missing value handling
        cleaned_data = cleaner.handle_missing_values(sample_data)

        # Verify missing values are handled
        assert (
            cleaned_data["Personal Loan"].isna().sum() == 0
        ), "No missing values should remain after cleaning"

        # Verify consistent imputation strategy with Housing Loan
        imputation_value = rules["imputation_strategy"]
        imputed_values = cleaned_data["Personal Loan"].value_counts()
        assert (
            imputation_value in imputed_values.index
        ), f"Should contain imputation value: {imputation_value}"

    def test_missing_value_strategy_consistency(self, phase3_raw_sample_data):
        """Test that missing value handling is consistent across loan columns."""
        cleaner = BankingDataCleaner()

        # Create test data with missing values in both loan columns
        test_data = pd.DataFrame(
            {
                "Housing Loan": ["yes", "no", np.nan, np.nan, "yes"],
                "Personal Loan": ["no", np.nan, "yes", np.nan, "no"],
            }
        )

        # Apply missing value handling
        cleaned_data = cleaner.handle_missing_values(test_data)

        # Verify consistency
        housing_imputed = (
            cleaned_data.loc[
                cleaned_data["Housing Loan"].str.contains("Information", na=False),
                "Housing Loan",
            ].iloc[0]
            if any(cleaned_data["Housing Loan"].str.contains("Information", na=False))
            else None
        )
        personal_imputed = (
            cleaned_data.loc[
                cleaned_data["Personal Loan"].str.contains("Information", na=False),
                "Personal Loan",
            ].iloc[0]
            if any(cleaned_data["Personal Loan"].str.contains("Information", na=False))
            else None
        )

        if housing_imputed and personal_imputed:
            assert (
                housing_imputed == personal_imputed
            ), "Imputation strategy should be consistent across loan columns"


class TestSpecialValueCleaning:
    """
    Test special value cleaning for 'unknown' categories.

    EDA Finding: 12,008 'unknown' values requiring attention
    - Credit Default: 20.9% unknown
    - Education Level: 4.2% unknown
    Strategy: Retain as distinct business category
    """

    def test_credit_default_unknown_handling(
        self, phase3_raw_sample_data, phase3_validation_rules
    ):
        """Test Credit Default unknown value handling (20.9% unknown from EDA)."""
        cleaner = BankingDataCleaner()
        rules = phase3_validation_rules["special_value_rules"]

        # Use EDA sample data
        sample_data = phase3_raw_sample_data.copy()

        # Verify initial unknown percentage
        initial_unknown_pct = (sample_data["Credit Default"] == "unknown").sum() / len(
            sample_data
        )
        assert (
            initial_unknown_pct >= 0.15
        ), f"Credit Default unknown rate should be high (EDA: 20.9%), got {initial_unknown_pct:.1%}"

        # Apply special value cleaning
        cleaned_data = cleaner.clean_special_values(sample_data)

        # Verify unknown values are retained as business category
        unknown_values = cleaned_data["Credit Default"].value_counts()
        assert (
            "unknown" in unknown_values.index
        ), "Unknown values should be retained as business category"

        # Verify retention strategy
        retention_strategy = rules["retention_strategy"]
        assert (
            retention_strategy == "keep_as_category"
        ), "Should use retention strategy for unknown values"

    def test_education_level_unknown_handling(
        self, phase3_raw_sample_data, phase3_validation_rules
    ):
        """Test Education Level unknown value handling (4.2% unknown from EDA)."""
        cleaner = BankingDataCleaner()

        # Use EDA sample data
        sample_data = phase3_raw_sample_data.copy()

        # Verify initial unknown percentage
        initial_unknown_pct = (sample_data["Education Level"] == "unknown").sum() / len(
            sample_data
        )
        assert (
            initial_unknown_pct >= 0.02
        ), f"Education Level unknown rate should be moderate (EDA: 4.2%), got {initial_unknown_pct:.1%}"

        # Apply special value cleaning
        cleaned_data = cleaner.clean_special_values(sample_data)

        # Verify unknown values are retained
        unknown_values = cleaned_data["Education Level"].value_counts()
        assert (
            "unknown" in unknown_values.index
        ), "Unknown values should be retained as business category"

    def test_unknown_value_consistency(self, phase3_validation_rules):
        """Test consistent handling of unknown values across categories."""
        cleaner = BankingDataCleaner()
        rules = phase3_validation_rules["special_value_rules"]

        # Test data with various unknown formats
        test_data = pd.DataFrame(
            {
                "Credit Default": ["yes", "no", "unknown", "unknown", "unknown"],
                "Education Level": [
                    "primary",
                    "unknown",
                    "secondary",
                    "unknown",
                    "tertiary",
                ],
            }
        )

        # Apply special value cleaning
        cleaned_data = cleaner.clean_special_values(test_data)

        # Verify unknown values are preserved as business categories
        retention_strategy = rules["retention_strategy"]
        assert (
            retention_strategy == "keep_as_category"
        ), "Should use retention strategy for unknown values"

        # Verify unknown values are retained in both columns
        for col in ["Credit Default", "Education Level"]:
            values = cleaned_data[col].unique()
            # Should preserve 'unknown' as distinct business category
            assert (
                "unknown" in values
            ), f"Should preserve 'unknown' values in {col} as business category"

    def test_special_values_business_meaning_preservation(self, phase3_raw_sample_data):
        """Test that special value cleaning preserves business meaning."""
        cleaner = BankingDataCleaner()

        # Use EDA sample data
        sample_data = phase3_raw_sample_data.copy()
        initial_credit_unknown = (sample_data["Credit Default"] == "unknown").sum()
        initial_education_unknown = (sample_data["Education Level"] == "unknown").sum()

        # Apply special value cleaning
        cleaned_data = cleaner.clean_special_values(sample_data)

        # Verify business meaning is preserved (unknown values still identifiable)
        final_credit_unknown = (
            cleaned_data["Credit Default"]
            .str.contains("unknown", case=False, na=False)
            .sum()
        )
        final_education_unknown = (
            cleaned_data["Education Level"]
            .str.contains("unknown", case=False, na=False)
            .sum()
        )

        assert (
            final_credit_unknown >= initial_credit_unknown * 0.9
        ), "Most Credit Default unknown values should be preserved"
        assert (
            final_education_unknown >= initial_education_unknown * 0.9
        ), "Most Education Level unknown values should be preserved"
