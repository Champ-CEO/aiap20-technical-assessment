"""
Phase 4 Data Integration Smoke Tests

Smoke tests for Phase 4 data integration core requirements:
1. Data loading smoke test: CSV file loads without errors
2. Schema validation smoke test: 33 features structure detected correctly  
3. Performance smoke test: Loading completes within reasonable time (<5 seconds for 41K records)
4. Critical path verification: Phase 3 → Phase 4 data flow works end-to-end

Following streamlined testing approach: critical path over exhaustive coverage.
"""

import pytest
import pandas as pd
import numpy as np
import time
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))


class TestPhase4DataIntegrationSmoke:
    """Smoke tests for Phase 4 data integration core requirements."""
    
    def test_data_loading_smoke_test(self):
        """
        Smoke Test: CSV file loads without errors (basic functionality verification)
        
        Requirements:
        - File exists and is accessible
        - Loads without exceptions
        - Returns valid DataFrame
        """
        # Arrange
        cleaned_data_path = Path("data/processed/cleaned-db.csv")
        
        # Act & Assert
        assert cleaned_data_path.exists(), f"Cleaned data file not found: {cleaned_data_path}"
        
        # Load data without errors
        df = pd.read_csv(cleaned_data_path)
        
        # Basic validation
        assert isinstance(df, pd.DataFrame), "Should return a valid DataFrame"
        assert len(df) > 0, "DataFrame should not be empty"
        assert len(df.columns) > 0, "DataFrame should have columns"
        
        print(f"✅ Data loading smoke test passed: {len(df)} rows, {len(df.columns)} columns loaded")
    
    def test_schema_validation_smoke_test(self):
        """
        Smoke Test: 33 features structure detected correctly
        
        Requirements:
        - Exactly 33 features as per Phase 3 output
        - Core columns from Phase 3 are present
        - Data types are preserved
        """
        # Arrange
        cleaned_data_path = Path("data/processed/cleaned-db.csv")
        expected_feature_count = 33
        
        # Core columns that must be present from Phase 3
        core_columns = [
            'Client ID', 'Age', 'Occupation', 'Marital Status', 'Education Level',
            'Credit Default', 'Housing Loan', 'Personal Loan', 'Contact Method',
            'Campaign Calls', 'Previous Contact Days', 'Subscription Status'
        ]
        
        # Act
        df = pd.read_csv(cleaned_data_path)
        
        # Assert
        assert len(df.columns) == expected_feature_count, \
            f"Expected {expected_feature_count} features, got {len(df.columns)}"
        
        # Verify core columns are present
        missing_core_columns = [col for col in core_columns if col not in df.columns]
        assert len(missing_core_columns) == 0, \
            f"Missing core columns from Phase 3: {missing_core_columns}"
        
        # Verify key data types are preserved
        assert pd.api.types.is_integer_dtype(df['Age']), "Age should be numeric"
        assert pd.api.types.is_integer_dtype(df['Subscription Status']), "Target should be binary encoded"
        
        print(f"✅ Schema validation smoke test passed: {len(df.columns)} features detected correctly")
    
    def test_performance_smoke_test(self):
        """
        Smoke Test: Loading completes within reasonable time (<5 seconds for 41K records)
        
        Requirements:
        - Load 41,188 records within 5 seconds
        - Performance meets Phase 4 requirements
        """
        # Arrange
        cleaned_data_path = Path("data/processed/cleaned-db.csv")
        max_load_time_seconds = 5.0
        expected_record_count = 41188
        
        # Act
        start_time = time.time()
        df = pd.read_csv(cleaned_data_path)
        load_time = time.time() - start_time
        
        # Assert
        assert load_time < max_load_time_seconds, \
            f"Loading took {load_time:.2f}s, should be < {max_load_time_seconds}s"
        
        assert len(df) == expected_record_count, \
            f"Expected {expected_record_count} records, got {len(df)}"
        
        # Calculate performance metrics
        records_per_second = len(df) / load_time if load_time > 0 else float('inf')
        
        print(f"✅ Performance smoke test passed: {len(df)} records loaded in {load_time:.3f}s")
        print(f"   Performance: {records_per_second:,.0f} records/second")
    
    def test_critical_path_verification_smoke_test(self):
        """
        Smoke Test: Phase 3 → Phase 4 data flow works end-to-end
        
        Requirements:
        - Phase 3 output is valid input for Phase 4
        - Data quality metrics from Phase 3 are preserved
        - No missing values (100% data quality score maintained)
        """
        # Arrange
        cleaned_data_path = Path("data/processed/cleaned-db.csv")
        
        # Act
        df = pd.read_csv(cleaned_data_path)
        
        # Assert Phase 3 → Phase 4 continuity
        # 1. Zero missing values (Phase 3 requirement)
        missing_values_count = df.isnull().sum().sum()
        assert missing_values_count == 0, \
            f"Expected 0 missing values from Phase 3, found {missing_values_count}"
        
        # 2. Target variable is properly encoded (Phase 3 requirement)
        unique_target_values = set(df['Subscription Status'].unique())
        expected_target_values = {0, 1}
        assert unique_target_values == expected_target_values, \
            f"Target should be binary encoded {expected_target_values}, got {unique_target_values}"
        
        # 3. Age is numeric (Phase 3 requirement)
        assert pd.api.types.is_numeric_dtype(df['Age']), "Age should be numeric from Phase 3"
        
        # 4. Age range validation (Phase 3 business rules)
        age_min, age_max = df['Age'].min(), df['Age'].max()
        assert 18 <= age_min <= age_max <= 100, \
            f"Age range should be 18-100 from Phase 3, got {age_min}-{age_max}"
        
        # 5. Record count preservation (Phase 3 requirement)
        assert len(df) == 41188, f"Should preserve 41,188 records from Phase 3, got {len(df)}"
        
        print(f"✅ Critical path verification passed: Phase 3 → Phase 4 data flow validated")
        print(f"   Records: {len(df)}, Features: {len(df.columns)}, Missing values: {missing_values_count}")


@pytest.mark.smoke
class TestPhase4QuickValidation:
    """Quick validation tests for Phase 4 readiness."""
    
    def test_phase4_input_file_exists(self):
        """Quick check that Phase 4 input file exists and is accessible."""
        cleaned_data_path = Path("data/processed/cleaned-db.csv")
        assert cleaned_data_path.exists(), "Phase 4 input file must exist"
        assert cleaned_data_path.stat().st_size > 0, "Phase 4 input file must not be empty"
    
    def test_phase4_basic_data_structure(self):
        """Quick validation of basic data structure for Phase 4."""
        df = pd.read_csv("data/processed/cleaned-db.csv")
        
        # Basic structure checks
        assert len(df) > 40000, "Should have approximately 41K records"
        assert len(df.columns) >= 30, "Should have 30+ features from Phase 3"
        assert 'Subscription Status' in df.columns, "Target variable must be present"
        
        print(f"✅ Phase 4 basic structure validated: {len(df)} rows, {len(df.columns)} columns")
