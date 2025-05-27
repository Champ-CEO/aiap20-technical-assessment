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
    return pd.DataFrame(
        {
            "age": np.random.randint(18, 80, 100),
            "job": np.random.choice(["admin.", "technician", "services"], 100),
            "marital": np.random.choice(["married", "single", "divorced"], 100),
            "education": np.random.choice(["primary", "secondary", "tertiary"], 100),
            "balance": np.random.randint(-1000, 5000, 100),
            "housing": np.random.choice(["yes", "no"], 100),
            "loan": np.random.choice(["yes", "no"], 100),
            "duration": np.random.randint(0, 1000, 100),
            "campaign": np.random.randint(1, 10, 100),
            "y": np.random.choice(["yes", "no"], 100),
        }
    )


@pytest.fixture
def small_sample_dataframe():
    """
    Create a small sample dataframe for fast testing (< 100 rows).

    Efficiency: Minimal dataset for Phase 2 streamlined testing.
    Business Value: Quick validation without performance overhead.
    """
    np.random.seed(42)
    return pd.DataFrame(
        {
            "Client ID": range(1, 51),  # 50 rows for fast testing
            "Age": np.random.choice(["25-35 years", "36-45 years", "46-55 years"], 50),
            "Occupation": np.random.choice(
                ["admin.", "technician", "services", "management"], 50
            ),
            "Marital Status": np.random.choice(["married", "single", "divorced"], 50),
            "Education Level": np.random.choice(
                ["primary", "secondary", "tertiary"], 50
            ),
            "Credit Default": np.random.choice(["yes", "no", "unknown"], 50),
            "Housing Loan": np.random.choice(["yes", "no"], 50),
            "Personal Loan": np.random.choice(["yes", "no"], 50),
            "Contact Method": np.random.choice(["cellular", "telephone"], 50),
            "Campaign Calls": np.random.randint(1, 10, 50),
            "Previous Contact Days": np.random.randint(-1, 999, 50),
            "Subscription Status": np.random.choice(["yes", "no"], 50),
        }
    )


@pytest.fixture
def expected_columns():
    """
    Define expected columns for banking marketing dataset validation.

    Business Value: Ensures data structure consistency for marketing analysis.
    """
    return [
        "Client ID",
        "Age",
        "Occupation",
        "Marital Status",
        "Education Level",
        "Credit Default",
        "Housing Loan",
        "Personal Loan",
        "Contact Method",
        "Campaign Calls",
        "Previous Contact Days",
        "Subscription Status",
    ]


@pytest.fixture
def expected_data_types():
    """
    Define expected data types for validation testing.

    Testing Philosophy: Quick verification of data integrity.
    """
    return {
        "Client ID": ["int64", "int32"],  # Allow both for flexibility
        "Campaign Calls": ["int64", "int32"],
        "Previous Contact Days": ["int64", "int32"],
        "Age": ["object", "str"],  # Text format in source data
        "Subscription Status": ["object", "str"],
    }


@pytest.fixture
def data_validation_rules():
    """
    Define business rules for data validation.

    Business Value: Ensure data quality for marketing decision making.
    """
    return {
        "subscription_values": ["yes", "no"],
        "age_categories": [
            "18-25 years",
            "25-35 years",
            "36-45 years",
            "46-55 years",
            "56-65 years",
            "65+ years",
        ],
        "campaign_calls_range": (
            -50,
            100,
        ),  # Allow some negative values as seen in real data
        "min_rows_expected": 1000,  # Minimum rows for meaningful analysis
        "required_columns_count": 12,
    }


@pytest.fixture
def temp_database():
    """Create a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        db_path = tmp_file.name

    # Create a simple test table
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
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
    """
    )

    # Insert sample data
    sample_data = [
        (1, 30, "admin.", "married", "secondary", 1500, "yes", "no", 200, 1, "no"),
        (2, 45, "technician", "single", "tertiary", 2500, "no", "yes", 300, 2, "yes"),
        (3, 25, "services", "married", "primary", 500, "yes", "no", 150, 1, "no"),
    ]

    cursor.executemany(
        """
        INSERT INTO bank_data (id, age, job, marital, education, balance, housing, loan, duration, campaign, y)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        sample_data,
    )

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
        "data": {
            "database_path": "data/raw/bmarket.db",
            "target_column": "y",
            "test_size": 0.2,
            "random_state": 42,
        },
        "preprocessing": {
            "handle_missing": True,
            "scale_features": True,
            "encode_categorical": True,
        },
        "models": {
            "logistic_regression": {"C": 1.0, "random_state": 42},
            "random_forest": {"n_estimators": 100, "random_state": 42},
        },
        "evaluation": {
            "cv_folds": 5,
            "scoring": ["accuracy", "precision", "recall", "f1"],
        },
    }


# Phase 3 Testing Fixtures - EDA-Informed Test Data


@pytest.fixture
def phase3_raw_sample_data():
    """
    Create sample data mimicking raw data issues identified in EDA.

    This fixture provides realistic test cases based on EDA findings:
    - Age in text format ('57 years', '55 years', etc.)
    - Missing values in Housing Loan (60.2%) and Personal Loan (10.1%)
    - 'Unknown' values in Credit Default (20.9%), Education Level (4.2%)
    - Contact method inconsistencies ('Cell' vs 'cellular')
    - Previous Contact Days with 999 values (96.3%)
    - Target variable as text ('yes'/'no')
    """
    np.random.seed(42)
    n_samples = 100

    # Age in text format (EDA finding)
    ages = [f"{age} years" for age in np.random.randint(18, 80, n_samples)]

    # Add some edge cases for age conversion testing
    ages[0] = "invalid age"
    ages[1] = "150 years"  # Outlier
    ages[2] = "17 years"  # Below minimum

    # Housing Loan with 60.2% missing (EDA finding)
    housing_loan = ["yes"] * 20 + ["no"] * 20 + [np.nan] * 60
    np.random.shuffle(housing_loan)

    # Personal Loan with 10.1% missing (EDA finding)
    personal_loan = ["yes"] * 45 + ["no"] * 45 + [np.nan] * 10
    np.random.shuffle(personal_loan)

    # Credit Default with 20.9% unknown (EDA finding)
    credit_default = ["yes"] * 30 + ["no"] * 49 + ["unknown"] * 21
    np.random.shuffle(credit_default)

    # Education Level with 4.2% unknown (EDA finding)
    education = (
        ["primary"] * 32 + ["secondary"] * 32 + ["tertiary"] * 32 + ["unknown"] * 4
    )
    np.random.shuffle(education)

    # Contact method inconsistencies (EDA finding)
    contact_methods = (
        ["Cell"] * 25 + ["cellular"] * 25 + ["Telephone"] * 25 + ["telephone"] * 25
    )

    # Previous Contact Days with 96.3% having 999 (EDA finding)
    previous_contact_days = [999] * 96 + list(np.random.randint(0, 30, 4))
    np.random.shuffle(previous_contact_days)

    # Target variable as text (EDA finding)
    target = ["yes"] * 11 + ["no"] * 89  # 11.3% subscription rate from EDA
    np.random.shuffle(target)

    return pd.DataFrame(
        {
            "Client ID": range(1, n_samples + 1),
            "Age": ages,
            "Occupation": np.random.choice(
                ["admin.", "technician", "services", "management"], n_samples
            ),
            "Marital Status": np.random.choice(
                ["married", "single", "divorced"], n_samples
            ),
            "Education Level": education,
            "Credit Default": credit_default,
            "Housing Loan": housing_loan,
            "Personal Loan": personal_loan,
            "Contact Method": contact_methods,
            "Campaign Calls": np.random.randint(
                -5, 60, n_samples
            ),  # Include negative values from EDA
            "Previous Contact Days": previous_contact_days,
            "Subscription Status": target,
        }
    )


@pytest.fixture
def phase3_expected_cleaned_schema():
    """
    Define expected schema after Phase 3 cleaning.

    Based on Phase 3 requirements:
    - Age: numeric (18-100 range)
    - Missing values: 0
    - Target variable: binary (1/0)
    - Contact methods: standardized
    - Previous Contact Days: 999 → binary flag
    """
    return {
        "expected_columns": [
            "Client ID",
            "Age",
            "Occupation",
            "Marital Status",
            "Education Level",
            "Credit Default",
            "Housing Loan",
            "Personal Loan",
            "Contact Method",
            "Campaign Calls",
            "Previous Contact Days",
            "Subscription Status",
            "No_Previous_Contact",  # New binary flag
        ],
        "expected_dtypes": {
            "Client ID": "int64",
            "Age": "float64",  # Numeric after conversion
            "Subscription Status": "int64",  # Binary encoded
            "Campaign Calls": "int64",
            "Previous Contact Days": "int64",
            "No_Previous_Contact": "int64",  # Binary flag
        },
        "expected_ranges": {
            "Age": (18, 100),
            "Campaign Calls": (0, 50),  # Capped at business realistic limits
            "Previous Contact Days": (0, 999),
            "Subscription Status": (0, 1),
            "No_Previous_Contact": (0, 1),
        },
        "expected_missing_values": 0,
        "standardized_contact_methods": [
            "cellular",
            "telephone",
        ],  # Standardized values
    }


@pytest.fixture
def phase3_validation_rules():
    """
    Business validation rules for Phase 3 testing.

    Based on EDA findings and business requirements.
    """
    return {
        "age_conversion_rules": {
            "valid_patterns": [r"\d+ years", r"\d+"],
            "invalid_patterns": ["invalid age", "unknown", ""],
            "min_age": 18,
            "max_age": 100,
        },
        "missing_value_rules": {
            "housing_loan_missing_threshold": 0.6,  # 60.2% from EDA
            "personal_loan_missing_threshold": 0.1,  # 10.1% from EDA
            "imputation_strategy": "Information Not Available",
        },
        "special_value_rules": {
            "unknown_categories": ["unknown", "Unknown", "UNKNOWN"],
            "credit_default_unknown_threshold": 0.2,  # 20.9% from EDA
            "education_unknown_threshold": 0.04,  # 4.2% from EDA
            "retention_strategy": "keep_as_category",
        },
        "standardization_rules": {
            "contact_method_mapping": {
                "Cell": "cellular",
                "cellular": "cellular",
                "Telephone": "telephone",
                "telephone": "telephone",
            },
            "previous_contact_999_flag": "No_Previous_Contact",
            "target_encoding": {"yes": 1, "no": 0},
        },
        "business_validation_rules": {
            "campaign_calls_max": 50,
            "campaign_calls_min": 0,
            "age_business_min": 18,
            "age_business_max": 100,
            "subscription_rate_expected": 0.113,  # 11.3% from EDA
        },
    }


# Test markers for different test categories
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests (fast, isolated)"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (slower, dependencies)",
    )
    config.addinivalue_line(
        "markers", "smoke: marks tests as smoke tests (quick verification)"
    )
    config.addinivalue_line("markers", "slow: marks tests as slow running")


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
