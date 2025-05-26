"""
Unit tests for the BankingDataLoader module.

This test suite validates the core functionality of the data loading system
including database connections, data retrieval, error handling, and CSV export.

Test Coverage:
- Database connection and validation
- Data loading with various parameters
- Error handling for missing files and invalid queries
- CSV export functionality
- Data summary generation
- Initial dataset snapshot creation
"""

import pytest
import pandas as pd
import sqlite3
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the module under test
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
from data.data_loader import (
    BankingDataLoader, 
    DatabaseConnectionError, 
    DataLoadError,
    create_initial_dataset_snapshot
)


class TestBankingDataLoader:
    """Test suite for BankingDataLoader class."""
    
    def test_init_with_default_path(self):
        """Test initialization with default database path."""
        # This will fail if the actual database doesn't exist, which is expected
        with pytest.raises(DatabaseConnectionError):
            loader = BankingDataLoader()
    
    def test_init_with_nonexistent_path(self):
        """Test initialization with non-existent database path."""
        with pytest.raises(DatabaseConnectionError, match="Database file not found"):
            loader = BankingDataLoader("nonexistent.db")
    
    def test_init_with_valid_path(self):
        """Test initialization with valid database path."""
        # Create a temporary SQLite database for testing
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            # Create a simple test database
            conn = sqlite3.connect(tmp_db.name)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE bank_marketing (
                    "Client ID" INTEGER,
                    "Age" TEXT,
                    "Occupation" TEXT,
                    "Subscription Status" TEXT
                )
            ''')
            cursor.execute('''
                INSERT INTO bank_marketing VALUES 
                (1, '25 years', 'admin.', 'yes'),
                (2, '30 years', 'technician', 'no'),
                (3, '35 years', 'services', 'yes')
            ''')
            conn.commit()
            conn.close()
            
            try:
                # Test successful initialization
                loader = BankingDataLoader(tmp_db.name)
                assert loader.db_path == Path(tmp_db.name)
                assert loader.default_table == "bank_marketing"
            finally:
                # Clean up
                os.unlink(tmp_db.name)
    
    def test_get_connection_context_manager(self):
        """Test the database connection context manager."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            # Create test database
            conn = sqlite3.connect(tmp_db.name)
            conn.execute('CREATE TABLE test (id INTEGER)')
            conn.commit()
            conn.close()
            
            try:
                loader = BankingDataLoader(tmp_db.name)
                
                # Test successful connection
                with loader.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = cursor.fetchall()
                    assert len(tables) > 0
                    
            finally:
                os.unlink(tmp_db.name)
    
    def test_get_table_info(self):
        """Test table information retrieval."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            # Create test database with known structure
            conn = sqlite3.connect(tmp_db.name)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE bank_marketing (
                    "Client ID" INTEGER PRIMARY KEY,
                    "Age" TEXT NOT NULL,
                    "Occupation" TEXT,
                    "Subscription Status" TEXT DEFAULT 'no'
                )
            ''')
            cursor.execute('INSERT INTO bank_marketing VALUES (1, "25 years", "admin.", "yes")')
            cursor.execute('INSERT INTO bank_marketing VALUES (2, "30 years", "technician", "no")')
            conn.commit()
            conn.close()
            
            try:
                loader = BankingDataLoader(tmp_db.name)
                table_info = loader.get_table_info()
                
                assert table_info['table_name'] == 'bank_marketing'
                assert table_info['row_count'] == 2
                assert table_info['column_count'] == 4
                assert len(table_info['columns']) == 4
                
                # Check specific column info
                client_id_col = next(col for col in table_info['columns'] if col['name'] == 'Client ID')
                assert client_id_col['type'] == 'INTEGER'
                assert client_id_col['primary_key'] == True
                
            finally:
                os.unlink(tmp_db.name)
    
    def test_load_data_basic(self):
        """Test basic data loading functionality."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            # Create test database
            conn = sqlite3.connect(tmp_db.name)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE bank_marketing (
                    "Client ID" INTEGER,
                    "Age" TEXT,
                    "Occupation" TEXT,
                    "Subscription Status" TEXT
                )
            ''')
            test_data = [
                (1, '25 years', 'admin.', 'yes'),
                (2, '30 years', 'technician', 'no'),
                (3, '35 years', 'services', 'yes'),
                (4, '40 years', 'management', 'no'),
                (5, '45 years', 'retired', 'yes')
            ]
            cursor.executemany('INSERT INTO bank_marketing VALUES (?, ?, ?, ?)', test_data)
            conn.commit()
            conn.close()
            
            try:
                loader = BankingDataLoader(tmp_db.name)
                
                # Test loading all data
                df = loader.load_data()
                assert len(df) == 5
                assert len(df.columns) == 4
                assert 'Client ID' in df.columns
                assert 'Subscription Status' in df.columns
                
                # Test loading with limit
                df_limited = loader.load_data(limit=3)
                assert len(df_limited) == 3
                
                # Test custom query
                df_filtered = loader.load_data(query="SELECT * FROM bank_marketing WHERE \"Subscription Status\" = 'yes'")
                assert len(df_filtered) == 3  # Should have 3 'yes' records
                
            finally:
                os.unlink(tmp_db.name)
    
    def test_export_to_csv(self):
        """Test CSV export functionality."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Create test database
                conn = sqlite3.connect(tmp_db.name)
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE bank_marketing (
                        "Client ID" INTEGER,
                        "Age" TEXT,
                        "Subscription Status" TEXT
                    )
                ''')
                cursor.execute('INSERT INTO bank_marketing VALUES (1, "25 years", "yes")')
                cursor.execute('INSERT INTO bank_marketing VALUES (2, "30 years", "no")')
                conn.commit()
                conn.close()
                
                try:
                    loader = BankingDataLoader(tmp_db.name)
                    csv_path = os.path.join(tmp_dir, 'test_export.csv')
                    
                    # Test export
                    export_summary = loader.export_to_csv(csv_path)
                    
                    assert export_summary['row_count'] == 2
                    assert export_summary['column_count'] == 3
                    assert export_summary['file_size_mb'] > 0
                    assert os.path.exists(csv_path)
                    
                    # Verify CSV content
                    df_from_csv = pd.read_csv(csv_path)
                    assert len(df_from_csv) == 2
                    assert 'Client ID' in df_from_csv.columns
                    
                finally:
                    os.unlink(tmp_db.name)
    
    def test_get_data_summary(self):
        """Test data summary generation."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            # Create test database
            conn = sqlite3.connect(tmp_db.name)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE bank_marketing (
                    "Client ID" INTEGER,
                    "Age" TEXT,
                    "Campaign Calls" INTEGER
                )
            ''')
            # Insert more than 1000 rows to test sampling
            for i in range(1500):
                cursor.execute('INSERT INTO bank_marketing VALUES (?, ?, ?)', 
                             (i, f'{20+i%50} years', i%10))
            conn.commit()
            conn.close()
            
            try:
                loader = BankingDataLoader(tmp_db.name)
                summary = loader.get_data_summary()
                
                assert 'table_info' in summary
                assert 'data_types' in summary
                assert 'sample_shape' in summary
                assert 'memory_usage_mb' in summary
                
                assert summary['table_info']['row_count'] == 1500
                assert summary['sample_shape'][0] == 1000  # Sample size
                assert summary['sample_shape'][1] == 3     # Number of columns
                
            finally:
                os.unlink(tmp_db.name)


class TestDataLoaderErrors:
    """Test error handling in data loader."""
    
    def test_invalid_table_name(self):
        """Test error handling for invalid table names."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            # Create empty database
            conn = sqlite3.connect(tmp_db.name)
            conn.close()
            
            try:
                loader = BankingDataLoader(tmp_db.name)
                
                with pytest.raises(DataLoadError, match="Table 'nonexistent' not found"):
                    loader.get_table_info('nonexistent')
                    
            finally:
                os.unlink(tmp_db.name)
    
    def test_invalid_sql_query(self):
        """Test error handling for invalid SQL queries."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            # Create test database
            conn = sqlite3.connect(tmp_db.name)
            conn.execute('CREATE TABLE bank_marketing (id INTEGER)')
            conn.close()
            
            try:
                loader = BankingDataLoader(tmp_db.name)
                
                with pytest.raises(DataLoadError, match="SQLite error loading data"):
                    loader.load_data(query="INVALID SQL QUERY")
                    
            finally:
                os.unlink(tmp_db.name)


class TestInitialDatasetSnapshot:
    """Test the initial dataset snapshot creation function."""
    
    @patch('data.data_loader.BankingDataLoader')
    def test_create_initial_dataset_snapshot_success(self, mock_loader_class):
        """Test successful snapshot creation."""
        # Mock the loader instance
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader
        
        # Mock the methods
        mock_loader.get_data_summary.return_value = {
            'data_types': {'Client ID': 'int64', 'Age': 'object'}
        }
        mock_loader.export_to_csv.return_value = {
            'file_path': '/path/to/initial_dataset.csv',
            'row_count': 41188,
            'column_count': 12,
            'file_size_mb': 2.9
        }
        
        # Test the function
        result = create_initial_dataset_snapshot()
        
        assert result['creation_status'] == 'success'
        assert result['total_rows'] == 41188
        assert result['total_columns'] == 12
        assert result['source_database'] == 'data/raw/bmarket.db'
        assert result['source_table'] == 'bank_marketing'
        
        # Verify method calls
        mock_loader.get_data_summary.assert_called_once()
        mock_loader.export_to_csv.assert_called_once_with('data/raw/initial_dataset.csv')
    
    @patch('data.data_loader.BankingDataLoader')
    def test_create_initial_dataset_snapshot_failure(self, mock_loader_class):
        """Test snapshot creation failure handling."""
        # Mock the loader to raise an exception
        mock_loader_class.side_effect = DatabaseConnectionError("Database not found")
        
        with pytest.raises(DataLoadError, match="Failed to create initial dataset snapshot"):
            create_initial_dataset_snapshot()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
