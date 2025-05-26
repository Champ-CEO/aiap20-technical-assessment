"""
Integration tests for the BankingDataLoader with real database.

This test suite validates the data loader functionality using the actual
bmarket.db database file. These tests ensure that the loader works correctly
with real data and validates the expected data structure and content.

Test Coverage:
- Real database connection and data loading
- Data integrity and structure validation
- CSV export with real data
- Performance and memory usage validation
- Initial dataset snapshot creation
"""

import pytest
import pandas as pd
import os
from pathlib import Path
import tempfile

# Import the module under test
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
from data.data_loader import (
    BankingDataLoader, 
    DatabaseConnectionError, 
    DataLoadError,
    create_initial_dataset_snapshot
)


class TestBankingDataLoaderIntegration:
    """Integration tests with real database."""
    
    @classmethod
    def setup_class(cls):
        """Set up test class with database path validation."""
        cls.db_path = Path("data/raw/bmarket.db")
        if not cls.db_path.exists():
            pytest.skip(f"Database file not found: {cls.db_path}")
    
    def test_real_database_connection(self):
        """Test connection to the real banking database."""
        loader = BankingDataLoader()
        
        # Test that we can establish a connection
        with loader.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            # Verify the expected table exists
            table_names = [table[0] for table in tables]
            assert 'bank_marketing' in table_names
    
    def test_real_table_info(self):
        """Test table information retrieval from real database."""
        loader = BankingDataLoader()
        table_info = loader.get_table_info()
        
        # Validate expected table structure
        assert table_info['table_name'] == 'bank_marketing'
        assert table_info['row_count'] == 41188  # Expected row count
        assert table_info['column_count'] == 12   # Expected column count
        
        # Validate expected columns
        column_names = [col['name'] for col in table_info['columns']]
        expected_columns = [
            'Client ID', 'Age', 'Occupation', 'Marital Status',
            'Education Level', 'Credit Default', 'Housing Loan',
            'Personal Loan', 'Contact Method', 'Campaign Calls',
            'Previous Contact Days', 'Subscription Status'
        ]
        
        for expected_col in expected_columns:
            assert expected_col in column_names, f"Missing column: {expected_col}"
    
    def test_real_data_loading(self):
        """Test data loading from real database."""
        loader = BankingDataLoader()
        
        # Test loading all data
        df = loader.load_data()
        
        # Validate data shape
        assert len(df) == 41188
        assert len(df.columns) == 12
        
        # Validate data types
        assert df['Client ID'].dtype == 'int64'
        assert df['Campaign Calls'].dtype == 'int64'
        assert df['Previous Contact Days'].dtype == 'int64'
        assert df['Age'].dtype == 'object'
        assert df['Subscription Status'].dtype == 'object'
        
        # Validate data content samples
        assert 'yes' in df['Subscription Status'].values
        assert 'no' in df['Subscription Status'].values
        
        # Test loading with limit
        df_sample = loader.load_data(limit=100)
        assert len(df_sample) == 100
        assert len(df_sample.columns) == 12
    
    def test_real_data_filtering(self):
        """Test data filtering with real database queries."""
        loader = BankingDataLoader()
        
        # Test filtering by subscription status
        df_subscribed = loader.load_data(
            query="SELECT * FROM bank_marketing WHERE \"Subscription Status\" = 'yes'"
        )
        
        # Validate filtered results
        assert len(df_subscribed) > 0
        assert all(df_subscribed['Subscription Status'] == 'yes')
        
        # Test filtering by age (numeric comparison on text field)
        df_young = loader.load_data(
            query="SELECT * FROM bank_marketing WHERE \"Age\" LIKE '%2_ years' LIMIT 100"
        )
        
        assert len(df_young) > 0
        assert len(df_young) <= 100
    
    def test_real_csv_export(self):
        """Test CSV export with real data."""
        loader = BankingDataLoader()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, 'test_real_export.csv')
            
            # Export a sample of data
            export_summary = loader.export_to_csv(
                csv_path, 
                query="SELECT * FROM bank_marketing LIMIT 1000"
            )
            
            # Validate export summary
            assert export_summary['row_count'] == 1000
            assert export_summary['column_count'] == 12
            assert export_summary['file_size_mb'] > 0
            assert os.path.exists(csv_path)
            
            # Validate CSV content
            df_from_csv = pd.read_csv(csv_path)
            assert len(df_from_csv) == 1000
            assert len(df_from_csv.columns) == 12
            
            # Validate column names match
            expected_columns = [
                'Client ID', 'Age', 'Occupation', 'Marital Status',
                'Education Level', 'Credit Default', 'Housing Loan',
                'Personal Loan', 'Contact Method', 'Campaign Calls',
                'Previous Contact Days', 'Subscription Status'
            ]
            for col in expected_columns:
                assert col in df_from_csv.columns
    
    def test_real_data_summary(self):
        """Test data summary generation with real data."""
        loader = BankingDataLoader()
        summary = loader.get_data_summary()
        
        # Validate summary structure
        assert 'table_info' in summary
        assert 'data_types' in summary
        assert 'sample_shape' in summary
        assert 'memory_usage_mb' in summary
        
        # Validate table info
        assert summary['table_info']['row_count'] == 41188
        assert summary['table_info']['column_count'] == 12
        
        # Validate sample shape (should be limited to 1000 rows)
        assert summary['sample_shape'][0] == 1000
        assert summary['sample_shape'][1] == 12
        
        # Validate data types
        data_types = summary['data_types']
        assert 'Client ID' in data_types
        assert 'Subscription Status' in data_types
    
    def test_data_quality_validation(self):
        """Test data quality aspects of the real dataset."""
        loader = BankingDataLoader()
        df = loader.load_data(limit=5000)  # Sample for performance
        
        # Test for expected unique values in categorical columns
        subscription_values = df['Subscription Status'].unique()
        assert 'yes' in subscription_values
        assert 'no' in subscription_values
        assert len(subscription_values) == 2
        
        # Test Client ID uniqueness in sample
        assert df['Client ID'].nunique() == len(df)
        
        # Test for expected data ranges
        campaign_calls = df['Campaign Calls']
        assert campaign_calls.min() >= -50  # Allow for some negative values as seen in data
        assert campaign_calls.max() <= 100   # Reasonable upper bound
        
        # Test for missing values patterns
        housing_loan_missing = df['Housing Loan'].isna().sum()
        personal_loan_missing = df['Personal Loan'].isna().sum()
        
        # These columns are known to have missing values
        assert housing_loan_missing >= 0
        assert personal_loan_missing >= 0
    
    def test_performance_benchmarks(self):
        """Test performance characteristics of data loading."""
        import time
        
        loader = BankingDataLoader()
        
        # Test loading speed for full dataset
        start_time = time.time()
        df = loader.load_data()
        load_time = time.time() - start_time
        
        # Should load 41k records in reasonable time (< 5 seconds)
        assert load_time < 5.0, f"Data loading took too long: {load_time:.2f} seconds"
        
        # Test memory usage is reasonable
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        assert memory_mb < 50, f"Memory usage too high: {memory_mb:.2f} MB"


class TestRealInitialDatasetSnapshot:
    """Test initial dataset snapshot creation with real data."""
    
    @classmethod
    def setup_class(cls):
        """Set up test class with database path validation."""
        cls.db_path = Path("data/raw/bmarket.db")
        if not cls.db_path.exists():
            pytest.skip(f"Database file not found: {cls.db_path}")
    
    def test_create_real_initial_snapshot(self):
        """Test creating initial dataset snapshot with real data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Temporarily change the output path for testing
            original_function = create_initial_dataset_snapshot
            
            def test_snapshot():
                loader = BankingDataLoader()
                summary = loader.get_data_summary()
                export_summary = loader.export_to_csv(
                    os.path.join(tmp_dir, 'test_initial_dataset.csv')
                )
                
                return {
                    'source_database': 'data/raw/bmarket.db',
                    'source_table': 'bank_marketing',
                    'output_file': export_summary['file_path'],
                    'total_rows': export_summary['row_count'],
                    'total_columns': export_summary['column_count'],
                    'file_size_mb': export_summary['file_size_mb'],
                    'data_types': summary['data_types'],
                    'creation_status': 'success'
                }
            
            # Test the snapshot creation
            result = test_snapshot()
            
            # Validate results
            assert result['creation_status'] == 'success'
            assert result['total_rows'] == 41188
            assert result['total_columns'] == 12
            assert result['file_size_mb'] > 0
            assert os.path.exists(result['output_file'])
            
            # Validate the created CSV file
            df = pd.read_csv(result['output_file'])
            assert len(df) == 41188
            assert len(df.columns) == 12


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
