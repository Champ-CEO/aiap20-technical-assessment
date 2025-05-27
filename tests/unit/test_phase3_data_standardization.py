"""
Phase 3 Data Standardization Tests (Priority 2)

Tests for data standardization requirements:
1. Contact method standardization: Cell/cellular, Telephone/telephone consistency
2. Previous Contact Days handling: 999 → 'No Previous Contact' flag validation
3. Target variable encoding: 'yes'/'no' → 1/0 binary conversion

Focus: Ensure standardization maintains business meaning
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from preprocessing.data_cleaner import BankingDataCleaner


class TestContactMethodStandardization:
    """
    Test contact method standardization.

    EDA Finding: Inconsistent contact method values
    - 'Cell' vs 'cellular': 31.8% vs 31.7%
    - 'Telephone' vs 'telephone': 18.4% vs 18.1%
    Requirement: Standardize to consistent casing and terminology
    """

    def test_cell_cellular_standardization(self, phase3_validation_rules):
        """Test standardization of Cell/cellular variations."""
        cleaner = BankingDataCleaner()
        rules = phase3_validation_rules["standardization_rules"]

        # Test data with Cell/cellular variations
        test_data = pd.DataFrame(
            {"Contact Method": ["Cell", "cellular", "Cell", "cellular", "Cell"]}
        )

        # Apply contact method standardization
        cleaned_data = cleaner.standardize_contact_methods(test_data)

        # Verify standardization
        unique_values = cleaned_data["Contact Method"].unique()
        mapping = rules["contact_method_mapping"]

        # Should have only one standardized value for cell contacts
        cell_values = [
            val
            for val in unique_values
            if val in [mapping["Cell"], mapping["cellular"]]
        ]
        assert (
            len(set(cell_values)) == 1
        ), "Cell/cellular should be standardized to single value"
        assert cell_values[0] == "cellular", "Should standardize to 'cellular'"

    def test_telephone_standardization(self, phase3_validation_rules):
        """Test standardization of Telephone/telephone variations."""
        cleaner = BankingDataCleaner()
        rules = phase3_validation_rules["standardization_rules"]

        # Test data with Telephone/telephone variations
        test_data = pd.DataFrame(
            {
                "Contact Method": [
                    "Telephone",
                    "telephone",
                    "Telephone",
                    "telephone",
                    "Telephone",
                ]
            }
        )

        # Apply contact method standardization
        cleaned_data = cleaner.standardize_contact_methods(test_data)

        # Verify standardization
        unique_values = cleaned_data["Contact Method"].unique()
        mapping = rules["contact_method_mapping"]

        # Should have only one standardized value for telephone contacts
        telephone_values = [
            val
            for val in unique_values
            if val in [mapping["Telephone"], mapping["telephone"]]
        ]
        assert (
            len(set(telephone_values)) == 1
        ), "Telephone/telephone should be standardized to single value"
        assert telephone_values[0] == "telephone", "Should standardize to 'telephone'"

    def test_contact_method_distribution_preservation(self, phase3_raw_sample_data):
        """Test that contact method distribution is preserved after standardization."""
        cleaner = BankingDataCleaner()

        # Use EDA sample data with realistic distribution
        sample_data = phase3_raw_sample_data.copy()

        # Count initial distribution
        initial_cell_count = (
            (sample_data["Contact Method"] == "Cell")
            | (sample_data["Contact Method"] == "cellular")
        ).sum()
        initial_telephone_count = (
            (sample_data["Contact Method"] == "Telephone")
            | (sample_data["Contact Method"] == "telephone")
        ).sum()

        # Apply standardization
        cleaned_data = cleaner.standardize_contact_methods(sample_data)

        # Count final distribution
        final_cellular_count = (cleaned_data["Contact Method"] == "cellular").sum()
        final_telephone_count = (cleaned_data["Contact Method"] == "telephone").sum()

        # Verify distribution is preserved
        assert (
            final_cellular_count == initial_cell_count
        ), "Cell/cellular count should be preserved"
        assert (
            final_telephone_count == initial_telephone_count
        ), "Telephone count should be preserved"

    def test_contact_method_eda_consistency(
        self, phase3_raw_sample_data, phase3_expected_cleaned_schema
    ):
        """Test that standardized values match expected schema from EDA."""
        cleaner = BankingDataCleaner()
        schema = phase3_expected_cleaned_schema

        # Use EDA sample data
        sample_data = phase3_raw_sample_data.copy()

        # Apply standardization
        cleaned_data = cleaner.standardize_contact_methods(sample_data)

        # Verify standardized values match expected schema
        unique_values = cleaned_data["Contact Method"].unique()
        expected_values = schema["standardized_contact_methods"]

        for value in unique_values:
            assert (
                value in expected_values
            ), f"Standardized value '{value}' should be in expected values: {expected_values}"


class TestPreviousContactDaysHandling:
    """
    Test Previous Contact Days special value handling.

    EDA Finding: 39,673 rows (96.3%) have '999' indicating no previous contact
    Requirement: Create 'No Previous Contact' binary flag
    """

    def test_999_to_binary_flag_conversion(
        self, phase3_raw_sample_data, phase3_validation_rules
    ):
        """Test conversion of 999 values to binary flag."""
        cleaner = BankingDataCleaner()
        rules = phase3_validation_rules["standardization_rules"]

        # Use EDA sample data with realistic 999 distribution
        sample_data = phase3_raw_sample_data.copy()

        # Verify initial 999 percentage is close to EDA finding
        initial_999_pct = (sample_data["Previous Contact Days"] == 999).sum() / len(
            sample_data
        )
        assert (
            initial_999_pct >= 0.9
        ), f"Previous Contact Days 999 rate should be high (EDA: 96.3%), got {initial_999_pct:.1%}"

        # Apply previous contact days handling (done in clean_special_values)
        cleaned_data = cleaner.clean_special_values(sample_data)

        # Verify binary flag is created
        flag_column = rules["previous_contact_999_flag"]
        assert (
            flag_column in cleaned_data.columns
        ), f"Should create binary flag column: {flag_column}"

        # Verify binary flag values
        flag_values = cleaned_data[flag_column].unique()
        assert set(flag_values).issubset(
            {0, 1}
        ), "Binary flag should only contain 0 and 1"

    def test_no_previous_contact_flag_accuracy(self, phase3_validation_rules):
        """Test accuracy of No Previous Contact flag creation."""
        cleaner = BankingDataCleaner()
        rules = phase3_validation_rules["standardization_rules"]

        # Test data with known 999 and non-999 values
        test_data = pd.DataFrame(
            {"Previous Contact Days": [999, 999, 15, 999, 30, 999, 5]}
        )

        # Apply handling (done in clean_special_values)
        cleaned_data = cleaner.clean_special_values(test_data)

        # Verify flag accuracy
        flag_column = rules["previous_contact_999_flag"]
        expected_flags = [1, 1, 0, 1, 0, 1, 0]  # 1 for 999, 0 for others

        assert (
            list(cleaned_data[flag_column]) == expected_flags
        ), "Binary flag should accurately reflect 999 values"

    def test_previous_contact_days_value_handling(self):
        """Test that non-999 Previous Contact Days values are preserved."""
        cleaner = BankingDataCleaner()

        # Test data with mix of 999 and valid contact days
        test_data = pd.DataFrame(
            {"Previous Contact Days": [999, 15, 999, 30, 5, 999, 45]}
        )

        original_non_999 = test_data[test_data["Previous Contact Days"] != 999][
            "Previous Contact Days"
        ].tolist()

        # Apply handling (done in clean_special_values)
        cleaned_data = cleaner.clean_special_values(test_data)

        # Verify non-999 values are preserved
        final_non_999 = cleaned_data[cleaned_data["Previous Contact Days"] != 999][
            "Previous Contact Days"
        ].tolist()
        assert set(original_non_999) == set(
            final_non_999
        ), "Non-999 values should be preserved"

    def test_business_meaning_preservation(self, phase3_raw_sample_data):
        """Test that business meaning is preserved in Previous Contact Days handling."""
        cleaner = BankingDataCleaner()

        # Use EDA sample data
        sample_data = phase3_raw_sample_data.copy()

        # Count initial patterns
        initial_999_count = (sample_data["Previous Contact Days"] == 999).sum()
        initial_non_999_count = (sample_data["Previous Contact Days"] != 999).sum()

        # Apply handling (done in clean_special_values)
        cleaned_data = cleaner.clean_special_values(sample_data)

        # Verify business meaning preservation
        if "No_Previous_Contact" in cleaned_data.columns:
            flag_1_count = (cleaned_data["No_Previous_Contact"] == 1).sum()
            flag_0_count = (cleaned_data["No_Previous_Contact"] == 0).sum()

            assert (
                flag_1_count == initial_999_count
            ), "Flag=1 count should match initial 999 count"
            assert (
                flag_0_count == initial_non_999_count
            ), "Flag=0 count should match initial non-999 count"


class TestTargetVariableEncoding:
    """
    Test target variable binary encoding.

    EDA Finding: Text values ('yes', 'no') for Subscription Status
    Requirement: Binary encoding (1=yes, 0=no) for model compatibility
    """

    def test_yes_no_to_binary_conversion(self, phase3_validation_rules):
        """Test conversion of 'yes'/'no' to 1/0 binary encoding."""
        cleaner = BankingDataCleaner()
        rules = phase3_validation_rules["standardization_rules"]

        # Test data with yes/no values
        test_data = pd.DataFrame(
            {"Subscription Status": ["yes", "no", "yes", "no", "yes"]}
        )

        # Apply target variable encoding
        cleaned_data = cleaner.encode_target_variable(test_data)

        # Verify binary encoding
        encoding_map = rules["target_encoding"]
        expected_values = [
            encoding_map["yes"],
            encoding_map["no"],
            encoding_map["yes"],
            encoding_map["no"],
            encoding_map["yes"],
        ]

        assert (
            list(cleaned_data["Subscription Status"]) == expected_values
        ), "Should convert yes/no to 1/0"
        assert cleaned_data["Subscription Status"].dtype in [
            "int64",
            "int32",
        ], "Target should be integer type"

    def test_target_variable_value_range(
        self, phase3_raw_sample_data, phase3_expected_cleaned_schema
    ):
        """Test that target variable values are in expected range."""
        cleaner = BankingDataCleaner()
        schema = phase3_expected_cleaned_schema

        # Use EDA sample data
        sample_data = phase3_raw_sample_data.copy()

        # Apply target encoding
        cleaned_data = cleaner.encode_target_variable(sample_data)

        # Verify value range
        expected_range = schema["expected_ranges"]["Subscription Status"]
        unique_values = cleaned_data["Subscription Status"].unique()

        for value in unique_values:
            assert (
                expected_range[0] <= value <= expected_range[1]
            ), f"Target value {value} should be in range {expected_range}"

    def test_target_distribution_preservation(self, phase3_raw_sample_data):
        """Test that target variable distribution is preserved after encoding."""
        cleaner = BankingDataCleaner()

        # Use EDA sample data
        sample_data = phase3_raw_sample_data.copy()

        # Count initial distribution
        initial_yes_count = (sample_data["Subscription Status"] == "yes").sum()
        initial_no_count = (sample_data["Subscription Status"] == "no").sum()

        # Apply encoding
        cleaned_data = cleaner.encode_target_variable(sample_data)

        # Count final distribution
        final_1_count = (cleaned_data["Subscription Status"] == 1).sum()
        final_0_count = (cleaned_data["Subscription Status"] == 0).sum()

        # Verify distribution preservation
        assert (
            final_1_count == initial_yes_count
        ), "Count of 'yes' should equal count of 1"
        assert (
            final_0_count == initial_no_count
        ), "Count of 'no' should equal count of 0"

    def test_subscription_rate_consistency(
        self, phase3_raw_sample_data, phase3_validation_rules
    ):
        """Test that subscription rate remains consistent with EDA findings."""
        cleaner = BankingDataCleaner()
        rules = phase3_validation_rules["business_validation_rules"]

        # Use EDA sample data
        sample_data = phase3_raw_sample_data.copy()

        # Apply encoding
        cleaned_data = cleaner.encode_target_variable(sample_data)

        # Calculate subscription rate
        subscription_rate = (cleaned_data["Subscription Status"] == 1).sum() / len(
            cleaned_data
        )
        expected_rate = rules["subscription_rate_expected"]

        # Allow for some variance due to sampling
        tolerance = 0.05  # 5% tolerance
        assert (
            abs(subscription_rate - expected_rate) <= tolerance
        ), f"Subscription rate {subscription_rate:.3f} should be close to EDA finding {expected_rate:.3f}"

    def test_model_compatibility(self, phase3_raw_sample_data):
        """Test that encoded target variable is compatible with ML models."""
        cleaner = BankingDataCleaner()

        # Use EDA sample data
        sample_data = phase3_raw_sample_data.copy()

        # Apply encoding
        cleaned_data = cleaner.encode_target_variable(sample_data)

        # Verify ML compatibility
        target_column = cleaned_data["Subscription Status"]

        # Should be numeric
        assert pd.api.types.is_numeric_dtype(
            target_column
        ), "Target should be numeric for ML compatibility"

        # Should not have missing values
        assert target_column.isna().sum() == 0, "Target should not have missing values"

        # Should be binary (only 0 and 1)
        unique_values = set(target_column.unique())
        assert unique_values.issubset(
            {0, 1}
        ), "Target should only contain 0 and 1 for binary classification"
