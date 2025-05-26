"""
Essential Data Validation Tests for Phase 2

This test suite implements the streamlined testing approach for Phase 2,
focusing on critical path verification rather than exhaustive validation.

Test Coverage:
- Smoke Test: Database connection works
- Data Validation: Expected columns and basic statistics  
- Sanity Check: Data types and value ranges

Testing Philosophy: Quick verification, not exhaustive validation
Business Value: Ensure data quality for marketing decision making
"""

import pytest
import pandas as pd
import sqlite3
import sys
from pathlib import Path

# Import the module under test
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
from data.data_loader import BankingDataLoader


@pytest.mark.unit
class TestDatabaseConnectionSmoke:
    """Smoke tests for database connection functionality."""
    
    def test_database_connection_works(self, sample_database_path):
        """
        Smoke Test: Verify database connection works.
        
        Testing Philosophy: Quick verification of core functionality.
        Business Value: Ensure data access for marketing analysis.
        """
        loader = BankingDataLoader()
        
        # Test connection can be established
        with loader.get_connection() as conn:
            assert conn is not None
            
            # Test basic query execution
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            assert result is not None
    
    def test_can_load_sample_data(self, sample_database_path):
        """
        Smoke Test: Verify basic data loading functionality.
        
        Efficiency: Small sample for fast testing.
        """
        loader = BankingDataLoader()
        
        # Load small sample for quick validation
        df = loader.load_data(limit=10)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert len(df.columns) > 0


@pytest.mark.unit  
class TestDataValidationEssentials:
    """Essential data validation tests for Phase 2."""
    
    def test_expected_columns_present(self, sample_database_path, expected_columns):
        """
        Data Validation: Verify expected columns are present.
        
        Business Value: Ensures data structure consistency for marketing analysis.
        Testing Philosophy: Quick verification of data integrity.
        """
        loader = BankingDataLoader()
        df = loader.load_data(limit=50)  # Small sample for speed
        
        # Check that all expected columns are present
        missing_columns = set(expected_columns) - set(df.columns)
        assert len(missing_columns) == 0, f"Missing columns: {missing_columns}"
        
        # Verify minimum column count
        assert len(df.columns) >= 10, f"Too few columns: {len(df.columns)}"
    
    def test_data_types_sanity_check(self, sample_database_path, expected_data_types):
        """
        Sanity Check: Verify data types are as expected.
        
        Testing Philosophy: Quick verification, not exhaustive validation.
        """
        loader = BankingDataLoader()
        df = loader.load_data(limit=50)  # Small sample for speed
        
        for column, expected_types in expected_data_types.items():
            if column in df.columns:
                actual_type = str(df[column].dtype)
                assert any(expected_type in actual_type for expected_type in expected_types), \
                    f"Column '{column}' has unexpected type: {actual_type}"
    
    def test_basic_statistics_sanity(self, sample_database_path):
        """
        Sanity Check: Verify basic statistics make business sense.
        
        Business Value: Ensure data quality for marketing decisions.
        """
        loader = BankingDataLoader()
        df = loader.load_data(limit=100)  # Small sample for speed
        
        # Test data shape
        assert len(df) > 0, "No data loaded"
        assert len(df.columns) >= 10, "Too few columns"
        
        # Test Client ID uniqueness in sample
        if 'Client ID' in df.columns:
            assert df['Client ID'].nunique() == len(df), "Client IDs should be unique"
        
        # Test subscription status values
        if 'Subscription Status' in df.columns:
            subscription_values = df['Subscription Status'].unique()
            assert 'yes' in subscription_values or 'no' in subscription_values, \
                "Subscription Status should contain 'yes' or 'no'"
    
    def test_value_ranges_sanity_check(self, sample_database_path, data_validation_rules):
        """
        Sanity Check: Verify data values are within reasonable ranges.
        
        Testing Philosophy: Quick verification of business rules.
        """
        loader = BankingDataLoader()
        df = loader.load_data(limit=100)  # Small sample for speed
        
        # Test campaign calls range
        if 'Campaign Calls' in df.columns:
            campaign_calls = df['Campaign Calls']
            min_val, max_val = data_validation_rules['campaign_calls_range']
            assert campaign_calls.min() >= min_val, f"Campaign calls too low: {campaign_calls.min()}"
            assert campaign_calls.max() <= max_val, f"Campaign calls too high: {campaign_calls.max()}"
        
        # Test subscription status values
        if 'Subscription Status' in df.columns:
            subscription_values = set(df['Subscription Status'].unique())
            expected_values = set(data_validation_rules['subscription_values'])
            assert subscription_values.issubset(expected_values), \
                f"Unexpected subscription values: {subscription_values - expected_values}"


@pytest.mark.unit
class TestDataQualityEssentials:
    """Essential data quality tests for Phase 2."""
    
    def test_no_completely_empty_columns(self, sample_database_path):
        """
        Data Quality: Verify no columns are completely empty.
        
        Business Value: Ensure data completeness for analysis.
        """
        loader = BankingDataLoader()
        df = loader.load_data(limit=100)  # Small sample for speed
        
        for column in df.columns:
            non_null_count = df[column].notna().sum()
            assert non_null_count > 0, f"Column '{column}' is completely empty"
    
    def test_target_variable_distribution(self, sample_database_path):
        """
        Data Quality: Verify target variable has reasonable distribution.
        
        Business Value: Ensure target variable is suitable for modeling.
        """
        loader = BankingDataLoader()
        df = loader.load_data(limit=500)  # Larger sample for distribution check
        
        if 'Subscription Status' in df.columns:
            target_counts = df['Subscription Status'].value_counts()
            
            # Should have both classes
            assert len(target_counts) >= 2, "Target variable should have multiple classes"
            
            # No class should be completely dominant (> 95%)
            max_proportion = target_counts.max() / len(df)
            assert max_proportion < 0.95, f"Target variable too imbalanced: {max_proportion:.2%}"
    
    def test_data_loading_performance(self, sample_database_path):
        """
        Performance: Verify data loading is reasonably fast.
        
        Testing Philosophy: Quick verification for development efficiency.
        """
        import time
        
        loader = BankingDataLoader()
        
        start_time = time.time()
        df = loader.load_data(limit=1000)
        load_time = time.time() - start_time
        
        # Should load 1000 records quickly (< 2 seconds)
        assert load_time < 2.0, f"Data loading too slow: {load_time:.2f} seconds"
        assert len(df) == 1000, "Incorrect number of rows loaded"


@pytest.mark.unit
class TestFixtureValidation:
    """Validate test fixtures work correctly."""
    
    def test_small_sample_dataframe_fixture(self, small_sample_dataframe):
        """
        Fixture Test: Verify small sample dataframe fixture.
        
        Efficiency: Ensure test fixtures are properly sized for fast testing.
        """
        df = small_sample_dataframe
        
        # Test size constraint (< 100 rows)
        assert len(df) < 100, f"Sample too large: {len(df)} rows"
        assert len(df) > 0, "Sample is empty"
        
        # Test has required structure
        assert 'Client ID' in df.columns
        assert 'Subscription Status' in df.columns
        
        # Test data types
        assert df['Client ID'].dtype in ['int64', 'int32']
        assert df['Subscription Status'].dtype == 'object'
    
    def test_expected_columns_fixture(self, expected_columns):
        """
        Fixture Test: Verify expected columns fixture.
        
        Testing Philosophy: Validate test infrastructure.
        """
        assert isinstance(expected_columns, list)
        assert len(expected_columns) >= 10
        assert 'Client ID' in expected_columns
        assert 'Subscription Status' in expected_columns
    
    def test_data_validation_rules_fixture(self, data_validation_rules):
        """
        Fixture Test: Verify data validation rules fixture.
        
        Business Value: Ensure validation rules are properly defined.
        """
        assert isinstance(data_validation_rules, dict)
        assert 'subscription_values' in data_validation_rules
        assert 'campaign_calls_range' in data_validation_rules
        assert 'min_rows_expected' in data_validation_rules
        
        # Test rule values
        assert len(data_validation_rules['subscription_values']) == 2
        assert isinstance(data_validation_rules['campaign_calls_range'], tuple)
        assert data_validation_rules['min_rows_expected'] > 0
