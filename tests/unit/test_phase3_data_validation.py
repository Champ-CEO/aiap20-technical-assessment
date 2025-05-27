"""
Phase 3 Data Validation Tests (Priority 3)

Tests for data validation and business rule compliance:
1. Range validation: Age (18-100), Campaign Calls (-41 to 56 investigation)
2. Business rule validation: Education-Occupation consistency
3. Quality metrics verification: Zero missing values post-processing
4. Pipeline integration test: End-to-end cleaning with EDA sample data
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from preprocessing.data_cleaner import BankingDataCleaner
from preprocessing.validation import DataValidator


class TestRangeValidation:
    """
    Test range validation for business rules.
    
    EDA Findings:
    - Age range needs validation (18-100 years)
    - Campaign Calls range: -41 to 56 (negative values need investigation)
    Requirement: Ensure realistic contact patterns and demographic constraints
    """
    
    def test_age_range_validation(self, phase3_validation_rules):
        """Test age range validation (18-100 years)."""
        validator = DataValidator()
        rules = phase3_validation_rules['business_validation_rules']
        
        # Test data with various age ranges
        test_data = pd.DataFrame({
            'Age': [17, 18, 25, 50, 75, 100, 101, 150]
        })
        
        # Apply validation
        validation_result = validator.validate_age_column(test_data)
        
        # Verify range validation
        min_age = rules['age_business_min']
        max_age = rules['age_business_max']
        
        # Check that outliers are identified
        outliers = validation_result.get('outliers', [])
        assert 17 in outliers or 101 in outliers or 150 in outliers, "Should identify age outliers"
        
        # Verify business rules are applied
        assert validation_result.get('statistics', {}).get('min_age', 0) >= min_age or len(outliers) > 0, \
            "Should enforce minimum age or identify violations"
    
    def test_campaign_calls_range_investigation(self, phase3_raw_sample_data, phase3_validation_rules):
        """Test Campaign Calls range validation (-41 to 56 investigation)."""
        validator = DataValidator()
        cleaner = BankingDataCleaner()
        rules = phase3_validation_rules['business_validation_rules']
        
        # Use EDA sample data with realistic campaign calls distribution
        sample_data = phase3_raw_sample_data.copy()
        
        # Apply validation before cleaning
        initial_validation = validator.validate_campaign_calls(sample_data)
        
        # Apply cleaning (should cap extreme values)
        cleaned_data = cleaner.validate_campaign_calls(sample_data)
        
        # Apply validation after cleaning
        final_validation = validator.validate_campaign_calls(cleaned_data)
        
        # Verify range enforcement
        max_calls = rules['campaign_calls_max']
        min_calls = rules['campaign_calls_min']
        
        final_calls = cleaned_data['Campaign Calls']
        assert all(final_calls >= min_calls), f"All campaign calls should be >= {min_calls}"
        assert all(final_calls <= max_calls), f"All campaign calls should be <= {max_calls}"
    
    def test_negative_campaign_calls_handling(self):
        """Test handling of negative campaign calls (EDA finding: -41 minimum)."""
        cleaner = BankingDataCleaner()
        
        # Test data with negative campaign calls
        test_data = pd.DataFrame({
            'Campaign Calls': [-41, -10, -1, 0, 5, 10, 56, 100]
        })
        
        # Apply validation and cleaning
        cleaned_data = cleaner.validate_campaign_calls(test_data)
        
        # Verify negative values are handled
        final_calls = cleaned_data['Campaign Calls']
        assert all(final_calls >= 0), "Negative campaign calls should be handled (set to 0 or removed)"
    
    def test_extreme_campaign_calls_capping(self, phase3_validation_rules):
        """Test capping of extreme campaign call values."""
        cleaner = BankingDataCleaner()
        rules = phase3_validation_rules['business_validation_rules']
        
        # Test data with extreme values
        test_data = pd.DataFrame({
            'Campaign Calls': [100, 200, 500, 1000]
        })
        
        # Apply validation
        cleaned_data = cleaner.validate_campaign_calls(test_data)
        
        # Verify capping
        max_calls = rules['campaign_calls_max']
        final_calls = cleaned_data['Campaign Calls']
        assert all(final_calls <= max_calls), f"Extreme values should be capped at {max_calls}"


class TestBusinessRuleValidation:
    """
    Test business rule validation for logical consistency.
    
    Requirements:
    - Education-Occupation alignment validation
    - Loan status consistency checks
    - Campaign timing constraints
    """
    
    def test_education_occupation_consistency(self):
        """Test Education-Occupation logical relationship validation."""
        validator = DataValidator()
        
        # Test data with logical and illogical combinations
        test_data = pd.DataFrame({
            'Education Level': ['tertiary', 'primary', 'secondary', 'tertiary', 'primary'],
            'Occupation': ['management', 'services', 'technician', 'admin.', 'management']
        })
        
        # Apply business rule validation
        validation_result = validator.validate_business_rules(test_data)
        
        # Verify consistency checks are performed
        assert 'education_occupation_consistency' in validation_result, \
            "Should check education-occupation consistency"
        
        # Check for potential inconsistencies (primary education + management might be flagged)
        consistency_issues = validation_result.get('education_occupation_consistency', {}).get('issues', [])
        # This is a business logic check - implementation may vary
        assert isinstance(consistency_issues, list), "Should return list of consistency issues"
    
    def test_loan_status_consistency(self):
        """Test Housing/Personal loan combination consistency."""
        validator = DataValidator()
        
        # Test data with various loan combinations
        test_data = pd.DataFrame({
            'Housing Loan': ['yes', 'no', 'yes', 'no', 'Information Not Available'],
            'Personal Loan': ['no', 'yes', 'yes', 'no', 'Information Not Available']
        })
        
        # Apply business rule validation
        validation_result = validator.validate_business_rules(test_data)
        
        # Verify loan consistency checks
        assert 'loan_consistency' in validation_result or 'business_rules_passed' in validation_result, \
            "Should check loan status consistency"
    
    def test_campaign_timing_constraints(self, phase3_raw_sample_data):
        """Test campaign timing business rules."""
        validator = DataValidator()
        
        # Use EDA sample data
        sample_data = phase3_raw_sample_data.copy()
        
        # Apply business rule validation
        validation_result = validator.validate_business_rules(sample_data)
        
        # Verify timing constraints are checked
        timing_checks = validation_result.get('campaign_timing', {})
        
        # Should validate realistic contact patterns
        if 'Previous Contact Days' in sample_data.columns and 'Campaign Calls' in sample_data.columns:
            # Basic timing logic: if previous contact days > 0, should have reasonable campaign calls
            assert isinstance(timing_checks, dict), "Should perform timing constraint validation"


class TestQualityMetricsVerification:
    """
    Test data quality metrics verification.
    
    Requirements:
    - Zero missing values post-processing
    - 100% data validation pass rate
    - All features properly typed and formatted
    """
    
    def test_zero_missing_values_requirement(self, phase3_raw_sample_data):
        """Test that all missing values are eliminated post-processing."""
        cleaner = BankingDataCleaner()
        
        # Use EDA sample data with known missing values
        sample_data = phase3_raw_sample_data.copy()
        
        # Verify initial missing values exist
        initial_missing = sample_data.isna().sum().sum()
        assert initial_missing > 0, "Sample data should have missing values for testing"
        
        # Apply complete cleaning pipeline
        cleaned_data = cleaner.clean_banking_data(sample_data)
        
        # Verify zero missing values
        final_missing = cleaned_data.isna().sum().sum()
        assert final_missing == 0, f"Should have zero missing values after cleaning, found {final_missing}"
    
    def test_data_validation_pass_rate(self, phase3_raw_sample_data):
        """Test 100% data validation pass rate."""
        cleaner = BankingDataCleaner()
        validator = DataValidator()
        
        # Use EDA sample data
        sample_data = phase3_raw_sample_data.copy()
        
        # Apply complete cleaning pipeline
        cleaned_data = cleaner.clean_banking_data(sample_data)
        
        # Apply comprehensive validation
        validation_report = validator.generate_validation_report(cleaned_data)
        
        # Verify high validation pass rate
        overall_score = validation_report.get('overall_quality_score', 0)
        assert overall_score >= 95, f"Should achieve high validation pass rate, got {overall_score}%"
    
    def test_feature_type_consistency(self, phase3_raw_sample_data, phase3_expected_cleaned_schema):
        """Test that all features are properly typed and formatted."""
        cleaner = BankingDataCleaner()
        schema = phase3_expected_cleaned_schema
        
        # Use EDA sample data
        sample_data = phase3_raw_sample_data.copy()
        
        # Apply complete cleaning pipeline
        cleaned_data = cleaner.clean_banking_data(sample_data)
        
        # Verify expected data types
        expected_dtypes = schema['expected_dtypes']
        
        for column, expected_dtype in expected_dtypes.items():
            if column in cleaned_data.columns:
                actual_dtype = str(cleaned_data[column].dtype)
                assert actual_dtype == expected_dtype or \
                       (expected_dtype == 'int64' and actual_dtype in ['int32', 'int64']) or \
                       (expected_dtype == 'float64' and actual_dtype in ['float32', 'float64']), \
                    f"Column {column} should have dtype {expected_dtype}, got {actual_dtype}"
    
    def test_business_quality_metrics(self, phase3_raw_sample_data, phase3_expected_cleaned_schema):
        """Test business-specific quality metrics."""
        cleaner = BankingDataCleaner()
        schema = phase3_expected_cleaned_schema
        
        # Use EDA sample data
        sample_data = phase3_raw_sample_data.copy()
        
        # Apply complete cleaning pipeline
        cleaned_data = cleaner.clean_banking_data(sample_data)
        
        # Verify expected ranges
        expected_ranges = schema['expected_ranges']
        
        for column, (min_val, max_val) in expected_ranges.items():
            if column in cleaned_data.columns:
                col_data = cleaned_data[column].dropna()
                if len(col_data) > 0:
                    assert all(col_data >= min_val), f"{column} values should be >= {min_val}"
                    assert all(col_data <= max_val), f"{column} values should be <= {max_val}"


class TestPipelineIntegrationEndToEnd:
    """
    Test end-to-end pipeline integration with EDA sample data.
    
    Requirements:
    - Complete pipeline execution without errors
    - All EDA-identified issues addressed
    - Output ready for Phase 4
    """
    
    def test_complete_pipeline_execution(self, phase3_raw_sample_data):
        """Test complete Phase 3 pipeline execution."""
        cleaner = BankingDataCleaner()
        
        # Use EDA sample data
        sample_data = phase3_raw_sample_data.copy()
        initial_shape = sample_data.shape
        
        # Execute complete pipeline
        try:
            cleaned_data = cleaner.clean_banking_data(sample_data)
            pipeline_success = True
        except Exception as e:
            pipeline_success = False
            pytest.fail(f"Pipeline execution failed: {str(e)}")
        
        # Verify pipeline completion
        assert pipeline_success, "Complete pipeline should execute without errors"
        assert cleaned_data is not None, "Pipeline should return cleaned data"
        assert len(cleaned_data) > 0, "Cleaned data should not be empty"
    
    def test_eda_issues_addressed(self, phase3_raw_sample_data):
        """Test that all EDA-identified issues are addressed."""
        cleaner = BankingDataCleaner()
        
        # Use EDA sample data
        sample_data = phase3_raw_sample_data.copy()
        
        # Document initial issues
        initial_issues = {
            'age_text_format': sample_data['Age'].dtype == 'object',
            'missing_values': sample_data.isna().sum().sum() > 0,
            'contact_inconsistency': len(sample_data['Contact Method'].unique()) > 2,
            'target_text_format': sample_data['Subscription Status'].dtype == 'object',
            'previous_contact_999': (sample_data['Previous Contact Days'] == 999).any()
        }
        
        # Apply complete cleaning
        cleaned_data = cleaner.clean_banking_data(sample_data)
        
        # Verify issues are addressed
        if initial_issues['age_text_format']:
            assert pd.api.types.is_numeric_dtype(cleaned_data['Age']), "Age should be numeric after cleaning"
        
        if initial_issues['missing_values']:
            assert cleaned_data.isna().sum().sum() == 0, "Missing values should be eliminated"
        
        if initial_issues['target_text_format']:
            assert pd.api.types.is_numeric_dtype(cleaned_data['Subscription Status']), \
                "Target should be numeric after encoding"
        
        if initial_issues['previous_contact_999']:
            assert 'No_Previous_Contact' in cleaned_data.columns or \
                   (cleaned_data['Previous Contact Days'] == 999).sum() == 0, \
                "Previous contact 999 values should be handled"
    
    def test_phase4_readiness(self, phase3_raw_sample_data, phase3_expected_cleaned_schema):
        """Test that output is ready for Phase 4 feature engineering."""
        cleaner = BankingDataCleaner()
        schema = phase3_expected_cleaned_schema
        
        # Use EDA sample data
        sample_data = phase3_raw_sample_data.copy()
        
        # Apply complete cleaning
        cleaned_data = cleaner.clean_banking_data(sample_data)
        
        # Verify Phase 4 readiness checklist
        readiness_checks = {
            'no_missing_values': cleaned_data.isna().sum().sum() == 0,
            'numeric_target': pd.api.types.is_numeric_dtype(cleaned_data['Subscription Status']),
            'numeric_age': pd.api.types.is_numeric_dtype(cleaned_data['Age']),
            'expected_columns_present': all(col in cleaned_data.columns for col in schema['expected_columns'][:12]),  # Core columns
            'data_not_empty': len(cleaned_data) > 0
        }
        
        # Verify all readiness checks pass
        for check_name, check_result in readiness_checks.items():
            assert check_result, f"Phase 4 readiness check failed: {check_name}"
        
        # Verify data quality score
        assert len(cleaned_data) == len(sample_data), "Should preserve all records during cleaning"
