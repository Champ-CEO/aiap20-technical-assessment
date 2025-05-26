"""
Smoke tests for the BankingDataLoader.

These tests verify that the basic functionality of the data loader works
correctly in the production environment. Smoke tests are designed to be
fast and catch major issues quickly.

Test Coverage:
- Basic database connection
- Data loading functionality
- CSV export capability
- Initial dataset snapshot creation
"""

import pytest
import os
from pathlib import Path

# Import the module under test
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
from data.data_loader import BankingDataLoader, create_initial_dataset_snapshot


class TestDataLoaderSmoke:
    """Smoke tests for basic data loader functionality."""
    
    @classmethod
    def setup_class(cls):
        """Set up test class with database path validation."""
        cls.db_path = Path("data/raw/bmarket.db")
        if not cls.db_path.exists():
            pytest.skip(f"Database file not found: {cls.db_path}")
    
    def test_can_initialize_loader(self):
        """Test that we can initialize the data loader."""
        loader = BankingDataLoader()
        assert loader is not None
        assert loader.default_table == "bank_marketing"
    
    def test_can_connect_to_database(self):
        """Test that we can connect to the database."""
        loader = BankingDataLoader()
        
        with loader.get_connection() as conn:
            assert conn is not None
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1
    
    def test_can_get_table_info(self):
        """Test that we can retrieve table information."""
        loader = BankingDataLoader()
        table_info = loader.get_table_info()
        
        assert table_info['table_name'] == 'bank_marketing'
        assert table_info['row_count'] > 0
        assert table_info['column_count'] > 0
        assert len(table_info['columns']) > 0
    
    def test_can_load_sample_data(self):
        """Test that we can load a sample of data."""
        loader = BankingDataLoader()
        df = loader.load_data(limit=10)
        
        assert len(df) == 10
        assert len(df.columns) > 0
        assert 'Client ID' in df.columns
        assert 'Subscription Status' in df.columns
    
    def test_can_generate_data_summary(self):
        """Test that we can generate a data summary."""
        loader = BankingDataLoader()
        summary = loader.get_data_summary()
        
        assert 'table_info' in summary
        assert 'data_types' in summary
        assert 'sample_shape' in summary
        assert 'memory_usage_mb' in summary
    
    def test_initial_dataset_exists(self):
        """Test that the initial dataset CSV file exists."""
        csv_path = Path("data/raw/initial_dataset.csv")
        assert csv_path.exists(), "Initial dataset CSV file should exist"
        
        # Check file is not empty
        assert csv_path.stat().st_size > 0, "Initial dataset CSV file should not be empty"
    
    def test_initial_dataset_has_correct_structure(self):
        """Test that the initial dataset has the expected structure."""
        import pandas as pd
        
        csv_path = Path("data/raw/initial_dataset.csv")
        if not csv_path.exists():
            pytest.skip("Initial dataset CSV file not found")
        
        df = pd.read_csv(csv_path, nrows=5)  # Just check first 5 rows for speed
        
        # Check expected columns exist
        expected_columns = [
            'Client ID', 'Age', 'Occupation', 'Marital Status',
            'Education Level', 'Credit Default', 'Housing Loan',
            'Personal Loan', 'Contact Method', 'Campaign Calls',
            'Previous Contact Days', 'Subscription Status'
        ]
        
        for col in expected_columns:
            assert col in df.columns, f"Expected column '{col}' not found in CSV"


class TestDataLoaderPerformanceSmoke:
    """Smoke tests for performance characteristics."""
    
    @classmethod
    def setup_class(cls):
        """Set up test class with database path validation."""
        cls.db_path = Path("data/raw/bmarket.db")
        if not cls.db_path.exists():
            pytest.skip(f"Database file not found: {cls.db_path}")
    
    def test_loading_sample_is_fast(self):
        """Test that loading a sample of data is reasonably fast."""
        import time
        
        loader = BankingDataLoader()
        
        start_time = time.time()
        df = loader.load_data(limit=1000)
        load_time = time.time() - start_time
        
        # Should load 1000 records quickly (< 2 seconds)
        assert load_time < 2.0, f"Sample loading took too long: {load_time:.2f} seconds"
        assert len(df) == 1000
    
    def test_table_info_is_fast(self):
        """Test that getting table info is fast."""
        import time
        
        loader = BankingDataLoader()
        
        start_time = time.time()
        table_info = loader.get_table_info()
        info_time = time.time() - start_time
        
        # Should get table info quickly (< 1 second)
        assert info_time < 1.0, f"Table info retrieval took too long: {info_time:.2f} seconds"
        assert table_info['row_count'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
