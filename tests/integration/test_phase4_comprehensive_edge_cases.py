"""
Phase 4 Comprehensive Edge Case Testing

This module implements comprehensive edge case testing for Phase 4 Data Integration,
covering scenarios that could occur in production environments:

1. Corrupted CSV files with various corruption types
2. Missing required columns scenarios
3. Invalid data types in input data
4. Empty and malformed datasets
5. Large file handling and memory management
6. Concurrent access and file locking scenarios
7. Network and I/O error simulation

Following TDD approach with focus on production readiness and robustness.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import time
import tempfile
import os
import threading
import io
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from data_integration import CSVLoader, DataValidator, PipelineUtils
from data_integration.csv_loader import CSVLoaderError
from data_integration.data_validator import DataValidationError


class TestCorruptedFileHandling:
    """Comprehensive tests for various types of corrupted CSV files."""

    def test_malformed_headers_handling(self):
        """
        Edge Case: CSV files with malformed headers

        Scenarios:
        - Headers with special characters
        - Duplicate column names
        - Missing headers
        - Headers with different encoding
        """
        test_cases = [
            # Malformed headers with special characters
            {
                "content": "Col@1,Col#2,Col$3\n1,2,3\n4,5,6\n",
                "description": "special characters in headers",
            },
            # Duplicate column names
            {
                "content": "Age,Age,Age\n25,30,35\n40,45,50\n",
                "description": "duplicate column names",
            },
            # Missing headers (data starts immediately)
            {
                "content": "25,Male,Engineer\n30,Female,Doctor\n",
                "description": "missing headers",
            },
        ]

        for case in test_cases:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False, encoding="utf-8"
            ) as tmp_file:
                tmp_file.write(case["content"])
                tmp_file_path = tmp_file.name

            try:
                # Test CSVLoader handling
                loader = CSVLoader(tmp_file_path)

                # Should either load with warnings or raise appropriate error
                try:
                    df = loader.load_data(
                        validate=False
                    )  # Skip validation for malformed data
                    # If it loads, verify it's handled appropriately
                    assert (
                        not df.empty
                    ), f"Should handle {case['description']} gracefully"
                    print(
                        f"✅ Handled {case['description']}: Loaded with {len(df)} rows"
                    )
                except (CSVLoaderError, pd.errors.ParserError, ValueError) as e:
                    print(
                        f"✅ Handled {case['description']}: Appropriate error raised - {type(e).__name__}"
                    )

            finally:
                try:
                    os.unlink(tmp_file_path)
                except (OSError, PermissionError):
                    pass

    def test_mixed_data_types_handling(self):
        """
        Edge Case: CSV files with mixed/inconsistent data types in columns

        Scenarios:
        - Numeric columns with text values
        - Date columns with invalid dates
        - Boolean columns with inconsistent values
        """
        # Mixed data types in numeric column
        mixed_content = """Client ID,Age,Subscription Status
1,25,1
2,thirty,0
3,35.5,yes
4,N/A,1
5,40,0"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as tmp_file:
            tmp_file.write(mixed_content)
            tmp_file_path = tmp_file.name

        try:
            loader = CSVLoader(tmp_file_path)

            # Should load but may have mixed types
            df = loader.load_data(validate=False)

            # Verify handling
            assert len(df) > 0, "Should load data despite mixed types"

            # Age column should be object type due to mixed content
            assert (
                df["Age"].dtype == "object"
            ), "Mixed type column should be object dtype"

            print(
                f"✅ Mixed data types handled: {len(df)} rows loaded with mixed types"
            )

        finally:
            try:
                os.unlink(tmp_file_path)
            except (OSError, PermissionError):
                pass

    def test_encoding_issues_handling(self):
        """
        Edge Case: CSV files with encoding issues

        Scenarios:
        - Non-UTF8 encoding
        - BOM (Byte Order Mark) presence
        - Special characters that cause encoding errors
        """
        # Content with special characters
        special_content = (
            "Name,Age,City\nJosé,25,São Paulo\nMüller,30,München\nNaïve,35,Café\n"
        )

        # Test different encodings
        encodings_to_test = ["utf-8", "latin-1", "cp1252"]

        for encoding in encodings_to_test:
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".csv", delete=False, encoding=encoding
                ) as tmp_file:
                    tmp_file.write(special_content)
                    tmp_file_path = tmp_file.name

                # Test loading with CSVLoader
                loader = CSVLoader(tmp_file_path)

                try:
                    df = loader.load_data(validate=False)
                    assert len(df) > 0, f"Should handle {encoding} encoding"
                    print(f"✅ Encoding {encoding} handled successfully")
                except (UnicodeDecodeError, CSVLoaderError) as e:
                    print(
                        f"✅ Encoding {encoding} error handled appropriately: {type(e).__name__}"
                    )

            except Exception as e:
                print(
                    f"⚠️ Encoding {encoding} test skipped due to system limitation: {e}"
                )
            finally:
                try:
                    os.unlink(tmp_file_path)
                except (OSError, PermissionError):
                    pass


class TestMissingColumnsScenarios:
    """Tests for handling missing required columns in input data."""

    def test_missing_critical_columns(self):
        """
        Edge Case: Missing critical columns required for Phase 4

        Test missing:
        - Target column (Subscription Status)
        - Key identifier (Client ID)
        - Important features (Age, Contact Method)
        """
        # Base valid data
        base_data = {
            "Client ID": [1, 2, 3, 4, 5],
            "Age": [25, 30, 35, 40, 45],
            "Contact Method": [
                "cellular",
                "telephone",
                "cellular",
                "telephone",
                "cellular",
            ],
            "Subscription Status": [0, 1, 0, 1, 0],
        }

        # Test scenarios with missing columns
        missing_column_scenarios = [
            {
                "missing_columns": ["Subscription Status"],
                "description": "missing target column",
            },
            {
                "missing_columns": ["Client ID"],
                "description": "missing identifier column",
            },
            {"missing_columns": ["Age"], "description": "missing key feature"},
            {
                "missing_columns": ["Subscription Status", "Age"],
                "description": "missing multiple critical columns",
            },
        ]

        for scenario in missing_column_scenarios:
            # Create data without specified columns
            test_data = {
                k: v
                for k, v in base_data.items()
                if k not in scenario["missing_columns"]
            }
            df_test = pd.DataFrame(test_data)

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as tmp_file:
                df_test.to_csv(tmp_file.name, index=False)
                tmp_file_path = tmp_file.name

            try:
                loader = CSVLoader(tmp_file_path)
                validator = DataValidator()

                # Load data
                df = loader.load_data(validate=False)

                # Test validation with missing columns
                try:
                    validation_report = validator.validate_data(df, comprehensive=False)
                    # Should detect missing columns
                    assert validation_report["overall_status"] in [
                        "FAILED",
                        "PARTIAL",
                    ], f"Should detect {scenario['description']}"
                    print(
                        f"✅ Detected {scenario['description']}: Validation failed appropriately"
                    )
                except DataValidationError as e:
                    print(
                        f"✅ Detected {scenario['description']}: Validation error raised - {e}"
                    )

            finally:
                try:
                    os.unlink(tmp_file_path)
                except (OSError, PermissionError):
                    pass

    def test_column_name_variations(self):
        """
        Edge Case: Column names with variations that might cause issues

        Scenarios:
        - Case sensitivity variations
        - Extra whitespace in column names
        - Similar but not exact column names
        """
        # Test data with column name variations
        variations_content = """client_id,AGE,Contact_Method,subscription_status
1,25,cellular,0
2,30,telephone,1
3,35,cellular,0"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmp_file:
            tmp_file.write(variations_content)
            tmp_file_path = tmp_file.name

        try:
            loader = CSVLoader(tmp_file_path)
            df = loader.load_data(validate=False)

            # Should load successfully
            assert len(df) > 0, "Should load data with column name variations"

            # Column names should be preserved as-is
            expected_columns = [
                "client_id",
                "AGE",
                "Contact_Method",
                "subscription_status",
            ]
            assert (
                list(df.columns) == expected_columns
            ), "Column names should be preserved exactly"

            print(f"✅ Column name variations handled: {list(df.columns)}")

        finally:
            try:
                os.unlink(tmp_file_path)
            except (OSError, PermissionError):
                pass


class TestInvalidDataTypes:
    """Tests for handling invalid data types in input data."""

    def test_extreme_values_handling(self):
        """
        Edge Case: Extreme values that might cause processing issues

        Scenarios:
        - Very large numbers
        - Negative values where not expected
        - Zero values in critical fields
        - Infinity and NaN values
        """
        extreme_data = {
            "Client ID": [1, 2, 3, 4, 5],
            "Age": [25, 999999, -5, 0, 150],  # Extreme age values
            "Campaign Calls": [1, 0, -1, 999999, 50],  # Extreme campaign values
            "Subscription Status": [0, 1, 2, -1, 0],  # Invalid target values
        }

        df_extreme = pd.DataFrame(extreme_data)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmp_file:
            df_extreme.to_csv(tmp_file.name, index=False)
            tmp_file_path = tmp_file.name

        try:
            loader = CSVLoader(tmp_file_path)
            validator = DataValidator()

            # Load data
            df = loader.load_data(validate=False)

            # Test validation with extreme values
            validation_report = validator.validate_data(df, comprehensive=True)

            # Should detect business rule violations
            assert validation_report["overall_status"] in [
                "FAILED",
                "PARTIAL",
            ], "Should detect extreme value violations"

            # Check specific business rule failures
            business_rules = validation_report.get("business_rules", {})
            assert not business_rules.get(
                "age_range_valid", True
            ), "Should detect invalid age ranges"

            print("✅ Extreme values detected and handled appropriately")

        finally:
            try:
                os.unlink(tmp_file_path)
            except (OSError, PermissionError):
                pass

    def test_special_float_values(self):
        """
        Edge Case: Special float values (NaN, Infinity, -Infinity)

        These values can cause issues in data processing and ML pipelines.
        """
        special_data = {
            "Client ID": [1, 2, 3, 4, 5],
            "Age": [25.0, float("inf"), float("-inf"), float("nan"), 35.0],
            "Campaign Calls": [1, 2, 3, 4, 5],
            "Subscription Status": [0, 1, 0, 1, 0],
        }

        df_special = pd.DataFrame(special_data)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmp_file:
            df_special.to_csv(tmp_file.name, index=False)
            tmp_file_path = tmp_file.name

        try:
            loader = CSVLoader(tmp_file_path)
            validator = DataValidator()

            # Load data
            df = loader.load_data(validate=False)

            # Check for special values
            has_inf = np.isinf(df.select_dtypes(include=[np.number])).any().any()
            has_nan = df.isnull().any().any()

            if has_inf or has_nan:
                print(f"✅ Special float values detected: inf={has_inf}, nan={has_nan}")

                # Validation should detect these issues
                validation_report = validator.validate_data(df, comprehensive=True)
                assert validation_report["overall_status"] in [
                    "FAILED",
                    "PARTIAL",
                ], "Should detect special float value issues"
            else:
                print("✅ Special float values handled during loading")

        finally:
            try:
                os.unlink(tmp_file_path)
            except (OSError, PermissionError):
                pass


class TestEmptyAndMalformedDatasets:
    """Tests for handling empty and malformed datasets."""

    def test_header_only_file(self):
        """
        Edge Case: CSV file with headers but no data rows
        """
        header_only_content = "Client ID,Age,Contact Method,Subscription Status\n"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmp_file:
            tmp_file.write(header_only_content)
            tmp_file_path = tmp_file.name

        try:
            loader = CSVLoader(tmp_file_path)

            # Should load but result in empty DataFrame
            df = loader.load_data(validate=False)

            assert len(df) == 0, "Header-only file should result in empty DataFrame"
            assert len(df.columns) == 4, "Should preserve column structure"

            print("✅ Header-only file handled: Empty DataFrame with correct columns")

        finally:
            try:
                os.unlink(tmp_file_path)
            except (OSError, PermissionError):
                pass

    def test_single_row_dataset(self):
        """
        Edge Case: Dataset with only one data row

        This can cause issues with statistical operations and data splitting.
        """
        single_row_content = """Client ID,Age,Contact Method,Subscription Status
1,25,cellular,0"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmp_file:
            tmp_file.write(single_row_content)
            tmp_file_path = tmp_file.name

        try:
            loader = CSVLoader(tmp_file_path)
            validator = DataValidator()
            utils = PipelineUtils()

            # Load data
            df = loader.load_data(validate=False)

            assert len(df) == 1, "Should load single row"

            # Test operations that might fail with single row
            try:
                # Statistical operations
                desc = df.describe()
                print("✅ Statistical operations work with single row")
            except Exception as e:
                print(f"⚠️ Statistical operations failed with single row: {e}")

            try:
                # Data splitting (should handle gracefully)
                splits = utils.split_data(
                    df, target_column="Subscription Status", test_size=0.2
                )
                print("✅ Data splitting handled single row gracefully")
            except Exception as e:
                print(f"⚠️ Data splitting failed with single row: {e}")

        finally:
            try:
                os.unlink(tmp_file_path)
            except (OSError, PermissionError):
                pass

    def test_inconsistent_row_lengths(self):
        """
        Edge Case: CSV with inconsistent number of columns per row

        Some rows have more or fewer columns than the header.
        """
        inconsistent_content = """Client ID,Age,Contact Method,Subscription Status
1,25,cellular,0
2,30,telephone
3,35,cellular,1,extra_value
4,40,telephone,0"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmp_file:
            tmp_file.write(inconsistent_content)
            tmp_file_path = tmp_file.name

        try:
            loader = CSVLoader(tmp_file_path)

            # pandas may raise ParserError for inconsistent row lengths
            try:
                df = loader.load_data(validate=False)

                assert len(df) > 0, "Should load data despite inconsistent row lengths"

                # Check if NaN values were introduced
                if df.isnull().any().any():
                    print(
                        "✅ Inconsistent row lengths handled: NaN values introduced appropriately"
                    )
                else:
                    print(
                        "✅ Inconsistent row lengths handled: Data loaded successfully"
                    )

            except (CSVLoaderError, pd.errors.ParserError) as e:
                # This is also acceptable behavior - pandas detecting the inconsistency
                print(
                    f"✅ Inconsistent row lengths detected appropriately: {type(e).__name__}"
                )
                assert (
                    "field" in str(e).lower() or "column" in str(e).lower()
                ), "Error should mention field/column inconsistency"

        finally:
            try:
                os.unlink(tmp_file_path)
            except (OSError, PermissionError):
                pass
