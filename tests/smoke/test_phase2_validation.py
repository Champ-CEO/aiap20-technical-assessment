"""
Phase 2 Validation Smoke Tests

Quick verification tests for Phase 2 requirements focusing on critical path
verification. These tests ensure the essential functionality works correctly
without exhaustive validation.

Test Coverage:
- Database connection and basic operations
- Essential data validation
- Performance characteristics
- Core functionality verification

Testing Philosophy: Quick verification, not exhaustive validation
Business Value: Ensure development environment works correctly for Phase 2
"""

import pytest
import pandas as pd
import time
import sys
from pathlib import Path

# Import the module under test
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
from data.data_loader import BankingDataLoader


@pytest.mark.smoke
class TestPhase2DatabaseSmoke:
    """Smoke tests for Phase 2 database functionality."""

    @classmethod
    def setup_class(cls):
        """Set up test class with database path validation."""
        cls.db_path = Path("data/raw/bmarket.db")
        if not cls.db_path.exists():
            pytest.skip(f"Database file not found: {cls.db_path}")

    def test_database_connection_smoke(self):
        """
        Smoke Test: Database connection works.

        Critical Path: Essential for all Phase 2 functionality.
        Business Value: Ensure data access for marketing analysis.
        """
        loader = BankingDataLoader()

        # Test connection establishment
        with loader.get_connection() as conn:
            assert conn is not None

            # Test table exists
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [table[0] for table in cursor.fetchall()]
            assert 'bank_marketing' in tables

    def test_basic_data_loading_smoke(self):
        """
        Smoke Test: Basic data loading functionality.

        Critical Path: Core functionality for Phase 2.
        Efficiency: Small sample for fast testing.
        """
        loader = BankingDataLoader()

        # Test loading small sample
        df = loader.load_data(limit=10)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert len(df.columns) >= 10

    def test_data_export_smoke(self):
        """
        Smoke Test: Data export functionality.

        Critical Path: Required for Phase 2 data snapshot creation.
        """
        import tempfile
        import os

        loader = BankingDataLoader()

        # Test export to temporary file
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_file:
            temp_path = tmp_file.name

        try:
            # Use custom query to limit data for testing
            summary = loader.export_to_csv(
                temp_path,
                query="SELECT * FROM bank_marketing LIMIT 50"
            )

            assert 'file_path' in summary
            assert 'row_count' in summary
            assert summary['row_count'] == 50
            assert Path(temp_path).exists()

        finally:
            # Cleanup
            try:
                os.unlink(temp_path)
            except (OSError, PermissionError):
                pass  # Windows file locking


@pytest.mark.smoke
class TestPhase2DataValidationSmoke:
    """Smoke tests for Phase 2 data validation."""

    @classmethod
    def setup_class(cls):
        """Set up test class with database path validation."""
        cls.db_path = Path("data/raw/bmarket.db")
        if not cls.db_path.exists():
            pytest.skip(f"Database file not found: {cls.db_path}")

    def test_expected_columns_smoke(self):
        """
        Smoke Test: Expected columns are present.

        Data Validation: Core data structure verification.
        Business Value: Ensure data consistency for marketing analysis.
        """
        loader = BankingDataLoader()
        df = loader.load_data(limit=10)

        # Test essential columns exist
        essential_columns = ['Client ID', 'Subscription Status']
        for col in essential_columns:
            assert col in df.columns, f"Essential column missing: {col}"

        # Test minimum column count
        assert len(df.columns) >= 10, f"Too few columns: {len(df.columns)}"

    def test_data_types_smoke(self):
        """
        Smoke Test: Data types are reasonable.

        Sanity Check: Quick verification of data integrity.
        """
        loader = BankingDataLoader()
        df = loader.load_data(limit=10)

        # Test key data types
        if 'Client ID' in df.columns:
            assert df['Client ID'].dtype in ['int64', 'int32'], \
                f"Client ID wrong type: {df['Client ID'].dtype}"

        if 'Subscription Status' in df.columns:
            assert df['Subscription Status'].dtype == 'object', \
                f"Subscription Status wrong type: {df['Subscription Status'].dtype}"

    def test_value_ranges_smoke(self):
        """
        Smoke Test: Data values are within reasonable ranges.

        Sanity Check: Quick verification of business rules.
        """
        loader = BankingDataLoader()
        df = loader.load_data(limit=50)

        # Test subscription status values
        if 'Subscription Status' in df.columns:
            subscription_values = set(df['Subscription Status'].unique())
            expected_values = {'yes', 'no'}
            assert subscription_values.issubset(expected_values), \
                f"Unexpected subscription values: {subscription_values - expected_values}"

        # Test Client ID uniqueness
        if 'Client ID' in df.columns:
            assert df['Client ID'].nunique() == len(df), "Client IDs should be unique"


@pytest.mark.smoke
class TestPhase2PerformanceSmoke:
    """Smoke tests for Phase 2 performance characteristics."""

    @classmethod
    def setup_class(cls):
        """Set up test class with database path validation."""
        cls.db_path = Path("data/raw/bmarket.db")
        if not cls.db_path.exists():
            pytest.skip(f"Database file not found: {cls.db_path}")

    def test_small_sample_loading_is_fast(self):
        """
        Performance Smoke Test: Small sample loading is fast.

        Efficiency: Ensure development workflow is not slowed by testing.
        Testing Philosophy: Quick verification for development efficiency.
        """
        loader = BankingDataLoader()

        start_time = time.time()
        df = loader.load_data(limit=100)
        load_time = time.time() - start_time

        # Should load 100 records very quickly (< 1 second)
        assert load_time < 1.0, f"Small sample loading too slow: {load_time:.2f} seconds"
        assert len(df) == 100

    def test_medium_sample_loading_is_reasonable(self):
        """
        Performance Smoke Test: Medium sample loading is reasonable.

        Business Value: Ensure data loading scales for analysis needs.
        """
        loader = BankingDataLoader()

        start_time = time.time()
        df = loader.load_data(limit=1000)
        load_time = time.time() - start_time

        # Should load 1000 records reasonably fast (< 2 seconds)
        assert load_time < 2.0, f"Medium sample loading too slow: {load_time:.2f} seconds"
        assert len(df) == 1000


@pytest.mark.smoke
class TestPhase2FixturesSmoke:
    """Smoke tests for Phase 2 test fixtures."""

    def test_small_sample_fixture_smoke(self, small_sample_dataframe):
        """
        Fixture Smoke Test: Small sample dataframe fixture works.

        Efficiency: Ensure test fixtures support fast testing.
        """
        df = small_sample_dataframe

        # Test size constraint for fast testing
        assert len(df) < 100, f"Sample too large for fast testing: {len(df)} rows"
        assert len(df) > 0, "Sample is empty"

        # Test has essential structure
        assert 'Client ID' in df.columns
        assert 'Subscription Status' in df.columns

    def test_validation_rules_fixture_smoke(self, data_validation_rules):
        """
        Fixture Smoke Test: Data validation rules fixture works.

        Testing Philosophy: Ensure validation infrastructure is ready.
        """
        assert isinstance(data_validation_rules, dict)
        assert 'subscription_values' in data_validation_rules
        assert 'min_rows_expected' in data_validation_rules

        # Test essential rules are defined
        assert len(data_validation_rules['subscription_values']) > 0
        assert data_validation_rules['min_rows_expected'] > 0

    def test_expected_columns_fixture_smoke(self, expected_columns):
        """
        Fixture Smoke Test: Expected columns fixture works.

        Data Validation: Ensure column validation infrastructure is ready.
        """
        assert isinstance(expected_columns, list)
        assert len(expected_columns) >= 10

        # Test essential columns are defined
        essential_columns = ['Client ID', 'Subscription Status']
        for col in essential_columns:
            assert col in expected_columns, f"Essential column not in fixture: {col}"


@pytest.mark.smoke
class TestPhase2IntegrationSmoke:
    """Integration smoke tests for Phase 2 functionality."""

    @classmethod
    def setup_class(cls):
        """Set up test class with database path validation."""
        cls.db_path = Path("data/raw/bmarket.db")
        if not cls.db_path.exists():
            pytest.skip(f"Database file not found: {cls.db_path}")

    def test_end_to_end_data_loading_smoke(self):
        """
        Integration Smoke Test: End-to-end data loading workflow.

        Critical Path: Complete Phase 2 data loading workflow.
        Business Value: Ensure complete data pipeline works.
        """
        loader = BankingDataLoader()

        # Test complete workflow
        # 1. Get data summary
        summary = loader.get_data_summary()
        assert 'table_info' in summary
        assert 'sample_shape' in summary

        # 2. Load sample data
        df = loader.load_data(limit=50)
        assert len(df) == 50
        assert len(df.columns) >= 10

        # 3. Validate data structure
        assert 'Client ID' in df.columns
        assert 'Subscription Status' in df.columns

        # 4. Check data quality
        assert df['Client ID'].nunique() == len(df)  # Unique IDs
        assert df['Subscription Status'].notna().all()  # No null targets

    def test_data_validation_workflow_smoke(self, expected_columns, data_validation_rules):
        """
        Integration Smoke Test: Data validation workflow.

        Testing Philosophy: Ensure validation workflow is complete.
        """
        loader = BankingDataLoader()
        df = loader.load_data(limit=100)

        # Test validation workflow
        # 1. Column validation
        missing_columns = set(expected_columns) - set(df.columns)
        assert len(missing_columns) == 0, f"Missing columns: {missing_columns}"

        # 2. Value validation
        if 'Subscription Status' in df.columns:
            subscription_values = set(df['Subscription Status'].unique())
            expected_values = set(data_validation_rules['subscription_values'])
            assert subscription_values.issubset(expected_values)

        # 3. Quality validation
        assert len(df) > 0
        assert df['Client ID'].nunique() == len(df)
