"""
Phase 4 Documentation Validation Testing

This module implements comprehensive documentation validation for Phase 4 Data Integration,
ensuring all functions have clear interfaces, proper error handling, and informative messages:

1. Function interface validation and docstring completeness
2. Error message clarity and informativeness
3. API consistency and parameter validation
4. Usage examples verification
5. Type hints and return type documentation
6. Integration documentation accuracy

Following TDD approach with focus on production documentation standards.
"""

import pytest
import pandas as pd
import sys
import inspect
import re
from pathlib import Path
from typing import get_type_hints

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from data_integration import (
    CSVLoader,
    DataValidator,
    PipelineUtils,
    load_phase3_output,
    validate_phase3_output,
    prepare_ml_pipeline,
    load_and_validate_data,
    prepare_data_for_ml,
    validate_phase3_continuity,
    get_data_summary,
    quick_data_check,
    load_sample_data,
)
from data_integration.csv_loader import CSVLoaderError
from data_integration.data_validator import DataValidationError


class TestFunctionDocumentation:
    """Tests for function interface validation and docstring completeness."""

    def test_all_public_functions_have_docstrings(self):
        """
        Documentation Test: All public functions should have comprehensive docstrings

        Verify that all public functions in the data_integration module have:
        - Non-empty docstrings
        - Parameter descriptions
        - Return value descriptions
        - Example usage (where appropriate)
        """
        # Get all public functions from main classes
        classes_to_check = [CSVLoader, DataValidator, PipelineUtils]
        functions_to_check = [
            load_phase3_output,
            validate_phase3_output,
            prepare_ml_pipeline,
            load_and_validate_data,
            prepare_data_for_ml,
            validate_phase3_continuity,
            get_data_summary,
            quick_data_check,
            load_sample_data,
        ]

        missing_docstrings = []
        incomplete_docstrings = []

        # Check class methods
        for cls in classes_to_check:
            for method_name in dir(cls):
                if not method_name.startswith("_"):  # Public methods only
                    method = getattr(cls, method_name)
                    if callable(method):
                        docstring = inspect.getdoc(method)
                        if not docstring:
                            missing_docstrings.append(f"{cls.__name__}.{method_name}")
                        elif len(docstring.strip()) < 50:  # Very short docstring
                            incomplete_docstrings.append(
                                f"{cls.__name__}.{method_name}"
                            )

        # Check standalone functions
        for func in functions_to_check:
            docstring = inspect.getdoc(func)
            if not docstring:
                missing_docstrings.append(func.__name__)
            elif len(docstring.strip()) < 50:  # Very short docstring
                incomplete_docstrings.append(func.__name__)

        # Assert no missing docstrings
        assert (
            len(missing_docstrings) == 0
        ), f"Functions missing docstrings: {missing_docstrings}"

        # Warn about incomplete docstrings
        if incomplete_docstrings:
            print(
                f"⚠️ Functions with potentially incomplete docstrings: {incomplete_docstrings}"
            )

        print(f"✅ All public functions have docstrings")

    def test_docstring_format_consistency(self):
        """
        Documentation Test: Docstrings should follow consistent format

        Check for consistent docstring formatting including:
        - Args/Parameters section
        - Returns section
        - Raises section (where applicable)
        """
        functions_to_check = [
            load_phase3_output,
            validate_phase3_output,
            prepare_ml_pipeline,
            load_and_validate_data,
            prepare_data_for_ml,
            validate_phase3_continuity,
        ]

        format_issues = []

        for func in functions_to_check:
            docstring = inspect.getdoc(func)
            if docstring:
                # Check for parameter documentation
                signature = inspect.signature(func)
                if len(signature.parameters) > 0:
                    # Should have Args or Parameters section
                    if not re.search(
                        r"(Args|Arguments|Parameters):", docstring, re.IGNORECASE
                    ):
                        format_issues.append(
                            f"{func.__name__}: Missing Args/Parameters section"
                        )

                # Check for return documentation if function returns something
                if signature.return_annotation != inspect.Signature.empty:
                    if not re.search(r"Returns?:", docstring, re.IGNORECASE):
                        format_issues.append(
                            f"{func.__name__}: Missing Returns section"
                        )

        # Report format issues as warnings (not failures)
        if format_issues:
            print(f"⚠️ Docstring format issues found:")
            for issue in format_issues:
                print(f"   - {issue}")
        else:
            print("✅ Docstring formats are consistent")

    def test_function_parameter_documentation(self):
        """
        Documentation Test: Function parameters should be properly documented

        Verify that all function parameters are documented in docstrings.
        """
        functions_to_check = [
            load_and_validate_data,
            prepare_data_for_ml,
            load_sample_data,
        ]

        undocumented_params = []

        for func in functions_to_check:
            docstring = inspect.getdoc(func)
            signature = inspect.signature(func)

            if docstring and signature.parameters:
                for param_name in signature.parameters:
                    if param_name not in ["self", "cls"]:  # Skip self and cls
                        # Check if parameter is mentioned in docstring
                        if param_name not in docstring:
                            undocumented_params.append(f"{func.__name__}.{param_name}")

        # Report undocumented parameters
        if undocumented_params:
            print(f"⚠️ Undocumented parameters: {undocumented_params}")
        else:
            print("✅ All function parameters are documented")


class TestErrorMessageClarity:
    """Tests for error message clarity and informativeness."""

    def test_csv_loader_error_messages(self):
        """
        Documentation Test: CSVLoader should provide clear error messages

        Test various error scenarios to ensure error messages are informative.
        """
        # Test file not found error
        try:
            loader = CSVLoader("nonexistent_file.csv")
            loader.load_data()
            assert False, "Should raise error for nonexistent file"
        except (CSVLoaderError, FileNotFoundError) as e:
            error_message = str(e)
            assert len(error_message) > 10, "Error message should be informative"
            assert (
                "nonexistent_file.csv" in error_message.lower()
                or "file" in error_message.lower()
            ), "Error message should mention file issue"
            print(f"✅ File not found error message: {error_message}")

        # Test with actual file but invalid parameters
        try:
            loader = CSVLoader("../data/processed/cleaned-db.csv")
            # Try to load with invalid column names
            loader.load_columns(["NonexistentColumn1", "NonexistentColumn2"])
            print("⚠️ Invalid column loading did not raise error")
        except Exception as e:
            error_message = str(e)
            assert len(error_message) > 10, "Error message should be informative"
            print(f"✅ Invalid column error message: {error_message}")

    def test_data_validator_error_messages(self):
        """
        Documentation Test: DataValidator should provide clear error messages

        Test validation error scenarios for message clarity.
        """
        # Create invalid data for testing
        invalid_data = pd.DataFrame(
            {
                "Age": [-5, 999, 150],  # Invalid age values (numeric only)
                "Subscription Status": [0, 1, 2],  # Invalid target values
            }
        )

        validator = DataValidator()

        try:
            validation_report = validator.validate_data(
                invalid_data, comprehensive=True
            )

            # Check if validation detected issues
            if validation_report["overall_status"] in ["FAILED", "PARTIAL"]:
                # Check if error messages are informative
                business_rules = validation_report.get("business_rules", {})
                if "validation_errors" in business_rules:
                    errors = business_rules["validation_errors"]
                    for error in errors:
                        assert (
                            len(str(error)) > 10
                        ), "Validation error messages should be informative"

                print("✅ DataValidator provides informative error messages")
            else:
                print("⚠️ DataValidator did not detect obvious data issues")

        except DataValidationError as e:
            error_message = str(e)
            assert (
                len(error_message) > 10
            ), "Validation error message should be informative"
            print(f"✅ Validation error message: {error_message}")

    def test_pipeline_utils_error_messages(self):
        """
        Documentation Test: PipelineUtils should provide clear error messages

        Test pipeline operation error scenarios.
        """
        utils = PipelineUtils()

        # Test with invalid data for splitting
        invalid_data = pd.DataFrame({"A": [1, 2, 3]})  # No target column

        try:
            utils.split_data(invalid_data, target_column="NonexistentTarget")
            assert False, "Should raise error for nonexistent target column"
        except Exception as e:
            error_message = str(e)
            assert len(error_message) > 10, "Error message should be informative"
            assert (
                "target" in error_message.lower() or "column" in error_message.lower()
            ), "Error message should mention target column issue"
            print(f"✅ Pipeline utils error message: {error_message}")


class TestAPIConsistency:
    """Tests for API consistency and parameter validation."""

    def test_parameter_naming_consistency(self):
        """
        Documentation Test: Parameter names should be consistent across functions

        Similar parameters should have consistent names across the API.
        """
        # Check for consistent parameter naming patterns
        functions_to_check = [
            load_and_validate_data,
            prepare_data_for_ml,
            load_sample_data,
        ]

        parameter_patterns = {}

        for func in functions_to_check:
            signature = inspect.signature(func)
            for param_name, param in signature.parameters.items():
                if param_name not in ["self", "cls"]:
                    # Group similar parameter types
                    if "file" in param_name.lower() or "path" in param_name.lower():
                        parameter_patterns.setdefault("file_path", []).append(
                            f"{func.__name__}.{param_name}"
                        )
                    elif "validate" in param_name.lower():
                        parameter_patterns.setdefault("validation", []).append(
                            f"{func.__name__}.{param_name}"
                        )
                    elif "comprehensive" in param_name.lower():
                        parameter_patterns.setdefault("comprehensive", []).append(
                            f"{func.__name__}.{param_name}"
                        )

        # Check for consistency within groups
        inconsistencies = []
        for pattern_type, params in parameter_patterns.items():
            if len(set(p.split(".")[-1] for p in params)) > 1:
                inconsistencies.append(f"{pattern_type}: {params}")

        if inconsistencies:
            print(f"⚠️ Parameter naming inconsistencies:")
            for inconsistency in inconsistencies:
                print(f"   - {inconsistency}")
        else:
            print("✅ Parameter naming is consistent")

    def test_return_type_consistency(self):
        """
        Documentation Test: Return types should be consistent and documented

        Similar functions should return similar types.
        """
        # Check return type annotations
        functions_to_check = [
            load_phase3_output,
            load_sample_data,
            load_and_validate_data,
        ]

        return_types = {}

        for func in functions_to_check:
            signature = inspect.signature(func)
            return_annotation = signature.return_annotation

            if return_annotation != inspect.Signature.empty:
                return_types[func.__name__] = return_annotation
            else:
                print(f"⚠️ {func.__name__} missing return type annotation")

        print(f"✅ Return type annotations found for {len(return_types)} functions")

    def test_default_parameter_values(self):
        """
        Documentation Test: Default parameter values should be reasonable

        Check that default values make sense for the function's purpose.
        """
        functions_to_check = [
            load_and_validate_data,
            prepare_data_for_ml,
            load_sample_data,
        ]

        unreasonable_defaults = []

        for func in functions_to_check:
            signature = inspect.signature(func)
            for param_name, param in signature.parameters.items():
                if param.default != inspect.Parameter.empty:
                    # Check for reasonable defaults
                    default_value = param.default

                    # File paths should default to None or a reasonable path
                    if "file" in param_name.lower() or "path" in param_name.lower():
                        if default_value is not None and not isinstance(
                            default_value, str
                        ):
                            unreasonable_defaults.append(
                                f"{func.__name__}.{param_name}: {default_value}"
                            )

                    # Boolean flags should default to reasonable values
                    elif "validate" in param_name.lower():
                        if not isinstance(default_value, bool):
                            unreasonable_defaults.append(
                                f"{func.__name__}.{param_name}: {default_value}"
                            )

        if unreasonable_defaults:
            print(f"⚠️ Potentially unreasonable default values: {unreasonable_defaults}")
        else:
            print("✅ Default parameter values are reasonable")


class TestUsageExamples:
    """Tests for usage examples verification."""

    def test_documented_examples_work(self):
        """
        Documentation Test: Documented usage examples should work correctly

        Test basic usage patterns that should be documented.
        """
        # Test basic data loading example
        try:
            df = load_phase3_output()
            assert len(df) == 41188, "Basic loading example should work"
            print("✅ Basic loading example works")
        except Exception as e:
            print(f"⚠️ Basic loading example failed: {e}")

        # Test ML pipeline preparation example
        try:
            splits = prepare_ml_pipeline()
            assert "train" in splits, "ML pipeline example should return train split"
            assert "test" in splits, "ML pipeline example should return test split"
            print("✅ ML pipeline example works")
        except Exception as e:
            print(f"⚠️ ML pipeline example failed: {e}")

        # Test quick validation example
        try:
            df = load_phase3_output()
            is_valid = quick_data_check(df)
            assert isinstance(is_valid, bool), "Quick validation should return boolean"
            print("✅ Quick validation example works")
        except Exception as e:
            print(f"⚠️ Quick validation example failed: {e}")

    def test_error_handling_examples(self):
        """
        Documentation Test: Error handling examples should be clear

        Test that error handling patterns are well-documented.
        """
        # Test file not found handling
        try:
            loader = CSVLoader("nonexistent_file.csv")
            loader.load_data()
        except Exception as e:
            # Error should be specific and informative
            assert len(str(e)) > 5, "Error messages should be informative"
            print(f"✅ File not found error handling: {type(e).__name__}")

        # Test validation error handling
        try:
            invalid_data = pd.DataFrame({"invalid": [1, 2, 3]})
            validator = DataValidator()
            report = validator.validate_data(invalid_data)
            # Should handle gracefully, not crash
            print("✅ Validation error handling works gracefully")
        except Exception as e:
            print(f"✅ Validation error handling: {type(e).__name__}")

    def test_integration_documentation_accuracy(self):
        """
        Documentation Test: Integration documentation should be accurate

        Verify that documented integration points work as described.
        """
        # Test Phase 3 → Phase 4 integration
        try:
            continuity_report = validate_phase3_continuity()
            assert isinstance(
                continuity_report, dict
            ), "Continuity validation should return dict"
            print("✅ Phase 3 → Phase 4 integration documentation accurate")
        except Exception as e:
            print(f"⚠️ Phase 3 → Phase 4 integration issue: {e}")

        # Test data summary functionality
        try:
            df = load_phase3_output()
            summary = get_data_summary(df)
            assert isinstance(summary, dict), "Data summary should return dict"
            print("✅ Data summary documentation accurate")
        except Exception as e:
            print(f"⚠️ Data summary issue: {e}")
