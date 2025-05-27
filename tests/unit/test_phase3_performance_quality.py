"""
Phase 3 Performance and Quality Assurance Tests (Priority 4)

Tests for performance monitoring and quality assurance:
1. Data quality metrics: Track cleaning success rates
2. Transformation logging: Verify all EDA-identified issues addressed
3. Business impact validation: Cleaned data supports marketing analysis
"""

import pytest
import pandas as pd
import numpy as np
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from preprocessing.data_cleaner import BankingDataCleaner
from preprocessing.validation import DataValidator


class TestDataQualityMetrics:
    """
    Test data quality metrics and cleaning success rates.
    
    Requirements:
    - Track cleaning success rates for each transformation
    - Monitor data quality improvement metrics
    - Validate cleaning effectiveness
    """
    
    def test_cleaning_success_rate_tracking(self, phase3_raw_sample_data):
        """Test tracking of cleaning success rates."""
        cleaner = BankingDataCleaner()
        
        # Use EDA sample data
        sample_data = phase3_raw_sample_data.copy()
        
        # Apply cleaning with metrics tracking
        cleaned_data = cleaner.clean_banking_data(sample_data)
        
        # Verify cleaning stats are tracked
        cleaning_stats = cleaner.cleaning_stats
        
        # Check key metrics are tracked
        expected_metrics = [
            'age_conversions',
            'missing_values_handled',
            'special_values_cleaned',
            'contact_methods_standardized',
            'target_variables_encoded'
        ]
        
        tracked_metrics = [metric for metric in expected_metrics if metric in cleaning_stats]
        assert len(tracked_metrics) >= 3, f"Should track key cleaning metrics, found: {tracked_metrics}"
        
        # Verify metrics have reasonable values
        for metric in tracked_metrics:
            value = cleaning_stats[metric]
            assert isinstance(value, (int, float)), f"Metric {metric} should be numeric"
            assert value >= 0, f"Metric {metric} should be non-negative"
    
    def test_data_quality_improvement_measurement(self, phase3_raw_sample_data):
        """Test measurement of data quality improvement."""
        cleaner = BankingDataCleaner()
        validator = DataValidator()
        
        # Use EDA sample data
        sample_data = phase3_raw_sample_data.copy()
        
        # Measure initial quality
        initial_report = validator.generate_validation_report(sample_data)
        initial_quality = initial_report.get('overall_quality_score', 0)
        
        # Apply cleaning
        cleaned_data = cleaner.clean_banking_data(sample_data)
        
        # Measure final quality
        final_report = validator.generate_validation_report(cleaned_data)
        final_quality = final_report.get('overall_quality_score', 0)
        
        # Verify quality improvement
        quality_improvement = final_quality - initial_quality
        assert quality_improvement > 0, f"Data quality should improve after cleaning: {initial_quality} -> {final_quality}"
        assert final_quality >= 90, f"Final quality should be high (>=90%), got {final_quality}%"
    
    def test_missing_value_elimination_rate(self, phase3_raw_sample_data):
        """Test missing value elimination success rate."""
        cleaner = BankingDataCleaner()
        
        # Use EDA sample data with known missing values
        sample_data = phase3_raw_sample_data.copy()
        
        # Count initial missing values
        initial_missing = sample_data.isna().sum().sum()
        assert initial_missing > 0, "Sample data should have missing values for testing"
        
        # Apply cleaning
        cleaned_data = cleaner.clean_banking_data(sample_data)
        
        # Count final missing values
        final_missing = cleaned_data.isna().sum().sum()
        
        # Calculate elimination rate
        elimination_rate = (initial_missing - final_missing) / initial_missing if initial_missing > 0 else 1.0
        
        # Verify high elimination rate
        assert elimination_rate >= 0.95, f"Should eliminate >=95% of missing values, achieved {elimination_rate:.1%}"
        assert final_missing == 0, f"Should eliminate all missing values, {final_missing} remaining"
    
    def test_special_value_handling_effectiveness(self, phase3_raw_sample_data):
        """Test effectiveness of special value handling."""
        cleaner = BankingDataCleaner()
        
        # Use EDA sample data
        sample_data = phase3_raw_sample_data.copy()
        
        # Count initial special values (unknown, 999, etc.)
        initial_unknown = (sample_data == 'unknown').sum().sum()
        initial_999 = (sample_data == 999).sum().sum()
        
        # Apply cleaning
        cleaned_data = cleaner.clean_banking_data(sample_data)
        
        # Verify special values are handled appropriately
        # Unknown values should be retained as business categories
        final_unknown = cleaned_data.astype(str).str.contains('unknown', case=False, na=False).sum().sum()
        
        # 999 values should be converted to binary flags
        final_999 = (cleaned_data == 999).sum().sum()
        no_contact_flags = cleaned_data.get('No_Previous_Contact', pd.Series()).sum()
        
        # Verify handling effectiveness
        if initial_unknown > 0:
            assert final_unknown >= initial_unknown * 0.8, "Most unknown values should be preserved as business categories"
        
        if initial_999 > 0:
            assert final_999 + no_contact_flags >= initial_999 * 0.8, "999 values should be handled (preserved or flagged)"


class TestTransformationLogging:
    """
    Test transformation logging and verification.
    
    Requirements:
    - Verify all EDA-identified issues are addressed
    - Log transformation operations
    - Track data lineage and changes
    """
    
    def test_eda_issue_resolution_logging(self, phase3_raw_sample_data):
        """Test logging of EDA issue resolution."""
        cleaner = BankingDataCleaner()
        
        # Use EDA sample data
        sample_data = phase3_raw_sample_data.copy()
        
        # Document EDA issues present in sample
        eda_issues = {
            'age_text_format': sample_data['Age'].dtype == 'object',
            'missing_housing_loan': sample_data['Housing Loan'].isna().any(),
            'missing_personal_loan': sample_data['Personal Loan'].isna().any(),
            'unknown_credit_default': (sample_data['Credit Default'] == 'unknown').any(),
            'contact_method_inconsistency': len(sample_data['Contact Method'].unique()) > 2,
            'previous_contact_999': (sample_data['Previous Contact Days'] == 999).any(),
            'target_text_format': sample_data['Subscription Status'].dtype == 'object'
        }
        
        # Apply cleaning
        cleaned_data = cleaner.clean_banking_data(sample_data)
        
        # Verify each EDA issue is addressed
        resolution_status = {}
        
        if eda_issues['age_text_format']:
            resolution_status['age_converted'] = pd.api.types.is_numeric_dtype(cleaned_data['Age'])
        
        if eda_issues['missing_housing_loan'] or eda_issues['missing_personal_loan']:
            resolution_status['missing_handled'] = cleaned_data.isna().sum().sum() == 0
        
        if eda_issues['unknown_credit_default']:
            resolution_status['unknown_preserved'] = 'unknown' in str(cleaned_data['Credit Default'].unique()).lower()
        
        if eda_issues['contact_method_inconsistency']:
            resolution_status['contact_standardized'] = len(cleaned_data['Contact Method'].unique()) <= 2
        
        if eda_issues['previous_contact_999']:
            resolution_status['999_handled'] = ('No_Previous_Contact' in cleaned_data.columns or 
                                               (cleaned_data['Previous Contact Days'] == 999).sum() == 0)
        
        if eda_issues['target_text_format']:
            resolution_status['target_encoded'] = pd.api.types.is_numeric_dtype(cleaned_data['Subscription Status'])
        
        # Verify all addressed issues are resolved
        unresolved_issues = [issue for issue, resolved in resolution_status.items() if not resolved]
        assert len(unresolved_issues) == 0, f"EDA issues not resolved: {unresolved_issues}"
    
    def test_transformation_operation_tracking(self, phase3_raw_sample_data):
        """Test tracking of transformation operations."""
        cleaner = BankingDataCleaner()
        
        # Use EDA sample data
        sample_data = phase3_raw_sample_data.copy()
        
        # Apply cleaning
        cleaned_data = cleaner.clean_banking_data(sample_data)
        
        # Verify transformation tracking
        cleaning_stats = cleaner.cleaning_stats
        
        # Check that operations are tracked
        operation_indicators = [
            'age_conversions',
            'missing_values_handled', 
            'special_values_cleaned',
            'contact_methods_standardized',
            'target_variables_encoded'
        ]
        
        tracked_operations = [op for op in operation_indicators if op in cleaning_stats and cleaning_stats[op] > 0]
        assert len(tracked_operations) >= 3, f"Should track multiple transformation operations, found: {tracked_operations}"
    
    def test_data_lineage_preservation(self, phase3_raw_sample_data):
        """Test preservation of data lineage through transformations."""
        cleaner = BankingDataCleaner()
        
        # Use EDA sample data
        sample_data = phase3_raw_sample_data.copy()
        initial_shape = sample_data.shape
        initial_client_ids = set(sample_data['Client ID'])
        
        # Apply cleaning
        cleaned_data = cleaner.clean_banking_data(sample_data)
        final_shape = cleaned_data.shape
        final_client_ids = set(cleaned_data['Client ID'])
        
        # Verify data lineage preservation
        assert final_shape[0] == initial_shape[0], "Should preserve all records during cleaning"
        assert final_client_ids == initial_client_ids, "Should preserve all client IDs"
        
        # Verify no data corruption
        assert len(cleaned_data) > 0, "Cleaned data should not be empty"
        assert 'Client ID' in cleaned_data.columns, "Should preserve key identifier columns"


class TestBusinessImpactValidation:
    """
    Test business impact validation.
    
    Requirements:
    - Cleaned data supports marketing analysis
    - Business rules are enforced
    - Data quality enables reliable insights
    """
    
    def test_marketing_analysis_readiness(self, phase3_raw_sample_data, phase3_expected_cleaned_schema):
        """Test that cleaned data supports marketing analysis."""
        cleaner = BankingDataCleaner()
        schema = phase3_expected_cleaned_schema
        
        # Use EDA sample data
        sample_data = phase3_raw_sample_data.copy()
        
        # Apply cleaning
        cleaned_data = cleaner.clean_banking_data(sample_data)
        
        # Verify marketing analysis requirements
        marketing_requirements = {
            'numeric_age_for_segmentation': pd.api.types.is_numeric_dtype(cleaned_data['Age']),
            'binary_target_for_modeling': pd.api.types.is_numeric_dtype(cleaned_data['Subscription Status']),
            'standardized_contact_methods': len(cleaned_data['Contact Method'].unique()) <= 3,
            'no_missing_values': cleaned_data.isna().sum().sum() == 0,
            'valid_age_range': (cleaned_data['Age'].min() >= 18 and cleaned_data['Age'].max() <= 100),
            'binary_target_range': set(cleaned_data['Subscription Status'].unique()).issubset({0, 1})
        }
        
        # Verify all marketing requirements are met
        unmet_requirements = [req for req, met in marketing_requirements.items() if not met]
        assert len(unmet_requirements) == 0, f"Marketing analysis requirements not met: {unmet_requirements}"
    
    def test_business_rule_enforcement(self, phase3_raw_sample_data, phase3_validation_rules):
        """Test enforcement of business rules."""
        cleaner = BankingDataCleaner()
        rules = phase3_validation_rules['business_validation_rules']
        
        # Use EDA sample data
        sample_data = phase3_raw_sample_data.copy()
        
        # Apply cleaning
        cleaned_data = cleaner.clean_banking_data(sample_data)
        
        # Verify business rule enforcement
        business_rule_compliance = {
            'age_within_business_range': (
                cleaned_data['Age'].min() >= rules['age_business_min'] and
                cleaned_data['Age'].max() <= rules['age_business_max']
            ),
            'campaign_calls_realistic': (
                cleaned_data['Campaign Calls'].min() >= rules['campaign_calls_min'] and
                cleaned_data['Campaign Calls'].max() <= rules['campaign_calls_max']
            ),
            'subscription_rate_reasonable': (
                abs((cleaned_data['Subscription Status'] == 1).mean() - rules['subscription_rate_expected']) <= 0.1
            )
        }
        
        # Verify business rule compliance
        non_compliant_rules = [rule for rule, compliant in business_rule_compliance.items() if not compliant]
        assert len(non_compliant_rules) == 0, f"Business rules not enforced: {non_compliant_rules}"
    
    def test_data_quality_for_reliable_insights(self, phase3_raw_sample_data):
        """Test that data quality enables reliable business insights."""
        cleaner = BankingDataCleaner()
        validator = DataValidator()
        
        # Use EDA sample data
        sample_data = phase3_raw_sample_data.copy()
        
        # Apply cleaning
        cleaned_data = cleaner.clean_banking_data(sample_data)
        
        # Generate quality report
        quality_report = validator.generate_validation_report(cleaned_data)
        
        # Verify quality metrics for reliable insights
        quality_requirements = {
            'high_overall_quality': quality_report.get('overall_quality_score', 0) >= 95,
            'complete_data': cleaned_data.isna().sum().sum() == 0,
            'consistent_data_types': all(
                pd.api.types.is_numeric_dtype(cleaned_data[col]) 
                for col in ['Age', 'Campaign Calls', 'Subscription Status']
            ),
            'valid_categorical_data': all(
                cleaned_data[col].dtype == 'object' 
                for col in ['Occupation', 'Marital Status', 'Contact Method']
                if col in cleaned_data.columns
            )
        }
        
        # Verify quality requirements
        unmet_quality = [req for req, met in quality_requirements.items() if not met]
        assert len(unmet_quality) == 0, f"Quality requirements for reliable insights not met: {unmet_quality}"
    
    def test_performance_benchmarks(self, phase3_raw_sample_data):
        """Test performance benchmarks for cleaning pipeline."""
        cleaner = BankingDataCleaner()
        
        # Use EDA sample data
        sample_data = phase3_raw_sample_data.copy()
        
        # Measure cleaning performance
        start_time = time.time()
        cleaned_data = cleaner.clean_banking_data(sample_data)
        end_time = time.time()
        
        processing_time = end_time - start_time
        records_per_second = len(sample_data) / processing_time if processing_time > 0 else float('inf')
        
        # Verify performance benchmarks
        assert processing_time < 10.0, f"Cleaning should complete within 10 seconds for sample data, took {processing_time:.2f}s"
        assert records_per_second > 10, f"Should process >10 records/second, achieved {records_per_second:.1f}"
        
        # Verify memory efficiency
        memory_usage_mb = cleaned_data.memory_usage(deep=True).sum() / 1024**2
        assert memory_usage_mb < 100, f"Memory usage should be reasonable (<100MB), used {memory_usage_mb:.1f}MB"
