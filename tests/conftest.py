"""
Pytest Configuration and Lightweight Fixtures

This module provides shared fixtures and configuration for the test suite.
Efficiency: Minimal setup, maximum utility.
"""

import pytest
import pandas as pd
import numpy as np
import sqlite3
import tempfile
import os
from pathlib import Path


@pytest.fixture(scope="session")
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root):
    """Get the test data directory."""
    return project_root / "data" / "raw"


@pytest.fixture(scope="session")
def sample_database_path(test_data_dir):
    """Path to the sample database for testing."""
    db_path = test_data_dir / "bmarket.db"
    if db_path.exists():
        return str(db_path)
    else:
        pytest.skip("Sample database not found. Run data download first.")


@pytest.fixture
def sample_dataframe():
    """Create a minimal sample dataframe for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'age': np.random.randint(18, 80, 100),
        'job': np.random.choice(['admin.', 'technician', 'services'], 100),
        'marital': np.random.choice(['married', 'single', 'divorced'], 100),
        'education': np.random.choice(['primary', 'secondary', 'tertiary'], 100),
        'balance': np.random.randint(-1000, 5000, 100),
        'housing': np.random.choice(['yes', 'no'], 100),
        'loan': np.random.choice(['yes', 'no'], 100),
        'duration': np.random.randint(0, 1000, 100),
        'campaign': np.random.randint(1, 10, 100),
        'y': np.random.choice(['yes', 'no'], 100)
    })


@pytest.fixture
def temp_database():
    """Create a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name

    # Create a simple test table
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE bank_data (
            id INTEGER PRIMARY KEY,
            age INTEGER,
            job TEXT,
            marital TEXT,
            education TEXT,
            balance INTEGER,
            housing TEXT,
            loan TEXT,
            duration INTEGER,
            campaign INTEGER,
            y TEXT
        )
    ''')

    # Insert sample data
    sample_data = [
        (1, 30, 'admin.', 'married', 'secondary', 1500, 'yes', 'no', 200, 1, 'no'),
        (2, 45, 'technician', 'single', 'tertiary', 2500, 'no', 'yes', 300, 2, 'yes'),
        (3, 25, 'services', 'married', 'primary', 500, 'yes', 'no', 150, 1, 'no'),
    ]

    cursor.executemany('''
        INSERT INTO bank_data (id, age, job, marital, education, balance, housing, loan, duration, campaign, y)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', sample_data)

    conn.commit()
    conn.close()

    yield db_path

    # Cleanup - handle Windows file locking issues
    try:
        os.unlink(db_path)
    except (OSError, PermissionError):
        # On Windows, sometimes the file is still locked
        # This is acceptable for temporary test files
        pass


@pytest.fixture
def mock_config():
    """Provide a mock configuration for testing."""
    return {
        'data': {
            'database_path': 'data/raw/bmarket.db',
            'target_column': 'y',
            'test_size': 0.2,
            'random_state': 42
        },
        'preprocessing': {
            'handle_missing': True,
            'scale_features': True,
            'encode_categorical': True
        },
        'models': {
            'logistic_regression': {
                'C': 1.0,
                'random_state': 42
            },
            'random_forest': {
                'n_estimators': 100,
                'random_state': 42
            }
        },
        'evaluation': {
            'cv_folds': 5,
            'scoring': ['accuracy', 'precision', 'recall', 'f1']
        }
    }


# Test markers for different test categories
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (slower, dependencies)"
    )
    config.addinivalue_line(
        "markers", "smoke: marks tests as smoke tests (quick verification)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )


# Pytest collection hooks for better organization
def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Mark tests based on directory structure
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "smoke" in str(item.fspath):
            item.add_marker(pytest.mark.smoke)
