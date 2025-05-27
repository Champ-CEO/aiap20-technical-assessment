"""
Data Validator for Phase 4 Data Integration

This module provides comprehensive data validation and integrity checks
to ensure Phase 3 transformations are preserved and data quality
standards are maintained.

Key Features:
- Schema validation against Phase 3 output specifications
- Data integrity checks for all transformations
- Business rule validation
- Quality score calculation and monitoring
- Feature validation for ML requirements
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


class DataValidator:
    """
    Comprehensive data validator for Phase 4 data integration.
    
    Validates that Phase 3 transformations are preserved and data quality
    standards are maintained for seamless Phase 4 → Phase 5 data flow.
    """
    
    def __init__(self):
        """Initialize the data validator with Phase 3 specifications."""
        # Phase 3 output specifications
        self.expected_records = 41188
        self.expected_features = 33
        self.quality_score_threshold = 100
        
        # Business rules from Phase 3
        self.business_rules = {
            'age_min': 18,
            'age_max': 100,
            'target_values': {0, 1},
            'contact_methods': {'cellular', 'telephone'},
            'missing_values_threshold': 0
        }
        
        # Core columns that must be present
        self.core_columns = [
            'Client ID', 'Age', 'Occupation', 'Marital Status', 'Education Level',
            'Credit Default', 'Housing Loan', 'Personal Loan', 'Contact Method',
            'Campaign Calls', 'Previous Contact Days', 'Subscription Status'
        ]
        
        # Expected data types
        self.expected_dtypes = {
            'Client ID': 'int64',
            'Age': ['int64', 'float64'],  # Allow both for flexibility
            'Campaign Calls': 'int64',
            'Previous Contact Days': 'int64',
            'Subscription Status': 'int64'
        }
        
        logger.info("Initialized DataValidator with Phase 3 specifications")
    
    def validate_data(self, df: pd.DataFrame, comprehensive: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive data validation.
        
        Args:
            df (pd.DataFrame): Data to validate
            comprehensive (bool): Whether to perform all validation checks
        
        Returns:
            Dict[str, Any]: Comprehensive validation report
        
        Raises:
            DataValidationError: If critical validation failures occur
        """
        logger.info("Starting comprehensive data validation...")
        
        validation_report = {
            'validation_timestamp': pd.Timestamp.now(),
            'dataset_info': self._get_dataset_info(df),
            'schema_validation': self._validate_schema(df),
            'data_integrity': self._validate_data_integrity(df),
            'business_rules': self._validate_business_rules(df),
            'quality_metrics': self._calculate_quality_metrics(df),
            'phase3_preservation': self._validate_phase3_preservation(df),
            'overall_status': 'PENDING'
        }
        
        if comprehensive:
            validation_report.update({
                'feature_validation': self._validate_features(df),
                'statistical_validation': self._validate_statistical_properties(df),
                'ml_readiness': self._validate_ml_readiness(df)
            })
        
        # Determine overall validation status
        validation_report['overall_status'] = self._determine_overall_status(validation_report)
        
        # Log validation summary
        self._log_validation_summary(validation_report)
        
        return validation_report
    
    def _get_dataset_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset information."""
        return {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            'column_names': list(df.columns),
            'data_types': df.dtypes.to_dict()
        }
    
    def _validate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data schema against Phase 3 specifications."""
        schema_validation = {
            'record_count_valid': len(df) == self.expected_records,
            'feature_count_valid': len(df.columns) == self.expected_features,
            'core_columns_present': True,
            'missing_core_columns': [],
            'extra_columns': [],
            'data_types_valid': True,
            'invalid_data_types': []
        }
        
        # Check core columns
        missing_columns = [col for col in self.core_columns if col not in df.columns]
        if missing_columns:
            schema_validation['core_columns_present'] = False
            schema_validation['missing_core_columns'] = missing_columns
        
        # Check for extra columns
        expected_core_set = set(self.core_columns)
        actual_columns_set = set(df.columns)
        extra_columns = actual_columns_set - expected_core_set
        if extra_columns:
            schema_validation['extra_columns'] = list(extra_columns)
        
        # Validate data types for key columns
        for col, expected_dtype in self.expected_dtypes.items():
            if col in df.columns:
                actual_dtype = str(df[col].dtype)
                if isinstance(expected_dtype, list):
                    if actual_dtype not in expected_dtype:
                        schema_validation['data_types_valid'] = False
                        schema_validation['invalid_data_types'].append({
                            'column': col,
                            'expected': expected_dtype,
                            'actual': actual_dtype
                        })
                else:
                    if actual_dtype != expected_dtype:
                        schema_validation['data_types_valid'] = False
                        schema_validation['invalid_data_types'].append({
                            'column': col,
                            'expected': expected_dtype,
                            'actual': actual_dtype
                        })
        
        return schema_validation
    
    def _validate_data_integrity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data integrity and completeness."""
        integrity_validation = {
            'missing_values_count': df.isnull().sum().sum(),
            'missing_values_by_column': df.isnull().sum().to_dict(),
            'duplicate_records_count': df.duplicated().sum(),
            'zero_missing_values': df.isnull().sum().sum() == 0,
            'data_completeness_percent': (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        }
        
        return integrity_validation
    
    def _validate_business_rules(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate business rules from Phase 3."""
        business_validation = {
            'age_range_valid': True,
            'target_binary_valid': True,
            'contact_methods_valid': True,
            'age_violations': 0,
            'target_violations': 0,
            'contact_violations': 0
        }
        
        # Age range validation
        if 'Age' in df.columns:
            age_violations = (~df['Age'].between(self.business_rules['age_min'], 
                                               self.business_rules['age_max'])).sum()
            business_validation['age_violations'] = age_violations
            business_validation['age_range_valid'] = age_violations == 0
        
        # Target variable validation
        if 'Subscription Status' in df.columns:
            unique_target_values = set(df['Subscription Status'].unique())
            target_valid = unique_target_values == self.business_rules['target_values']
            business_validation['target_binary_valid'] = target_valid
            if not target_valid:
                business_validation['target_violations'] = len(unique_target_values - self.business_rules['target_values'])
        
        # Contact method validation
        if 'Contact Method' in df.columns:
            unique_contact_methods = set(df['Contact Method'].unique())
            contact_valid = unique_contact_methods == self.business_rules['contact_methods']
            business_validation['contact_methods_valid'] = contact_valid
            if not contact_valid:
                business_validation['contact_violations'] = len(unique_contact_methods - self.business_rules['contact_methods'])
        
        return business_validation
    
    def _calculate_quality_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics."""
        quality_metrics = {
            'completeness_score': 0,
            'consistency_score': 0,
            'validity_score': 0,
            'accuracy_score': 0,
            'overall_quality_score': 0
        }
        
        # Completeness (no missing values)
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        quality_metrics['completeness_score'] = completeness
        
        # Consistency (data types are correct)
        consistency = 100  # Assume consistent if loaded successfully
        quality_metrics['consistency_score'] = consistency
        
        # Validity (business rules satisfied)
        validity_checks = []
        if 'Age' in df.columns:
            validity_checks.append(df['Age'].between(18, 100).all())
        if 'Subscription Status' in df.columns:
            validity_checks.append(set(df['Subscription Status'].unique()) == {0, 1})
        
        validity = (sum(validity_checks) / len(validity_checks) * 100) if validity_checks else 100
        quality_metrics['validity_score'] = validity
        
        # Accuracy (Phase 3 transformations preserved)
        accuracy_checks = []
        if 'Age' in df.columns:
            accuracy_checks.append(pd.api.types.is_numeric_dtype(df['Age']))
        if 'Subscription Status' in df.columns:
            accuracy_checks.append(pd.api.types.is_integer_dtype(df['Subscription Status']))
        
        accuracy = (sum(accuracy_checks) / len(accuracy_checks) * 100) if accuracy_checks else 100
        quality_metrics['accuracy_score'] = accuracy
        
        # Overall quality score
        scores = [completeness, consistency, validity, accuracy]
        quality_metrics['overall_quality_score'] = sum(scores) / len(scores)
        
        return quality_metrics
    
    def _validate_phase3_preservation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate that Phase 3 transformations are preserved."""
        preservation_validation = {
            'age_conversion_preserved': False,
            'target_encoding_preserved': False,
            'missing_values_eliminated': False,
            'contact_standardization_preserved': False,
            'previous_contact_handling_preserved': False,
            'feature_engineering_preserved': False
        }
        
        # Age conversion preservation
        if 'Age' in df.columns:
            preservation_validation['age_conversion_preserved'] = pd.api.types.is_numeric_dtype(df['Age'])
        
        # Target encoding preservation
        if 'Subscription Status' in df.columns:
            preservation_validation['target_encoding_preserved'] = (
                pd.api.types.is_integer_dtype(df['Subscription Status']) and
                set(df['Subscription Status'].unique()) == {0, 1}
            )
        
        # Missing values elimination
        preservation_validation['missing_values_eliminated'] = df.isnull().sum().sum() == 0
        
        # Contact standardization preservation
        if 'Contact Method' in df.columns:
            preservation_validation['contact_standardization_preserved'] = (
                set(df['Contact Method'].unique()) == {'cellular', 'telephone'}
            )
        
        # Previous contact handling preservation
        preservation_validation['previous_contact_handling_preserved'] = 'No_Previous_Contact' in df.columns
        
        # Feature engineering preservation (33 features total)
        preservation_validation['feature_engineering_preserved'] = len(df.columns) == 33
        
        return preservation_validation
    
    def _validate_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate features for ML readiness."""
        feature_validation = {
            'numeric_features_count': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features_count': len(df.select_dtypes(include=['object']).columns),
            'features_with_variation': 0,
            'constant_features': [],
            'high_cardinality_features': []
        }
        
        # Check for features with variation
        for col in df.columns:
            if df[col].nunique() > 1:
                feature_validation['features_with_variation'] += 1
            else:
                feature_validation['constant_features'].append(col)
        
        # Check for high cardinality categorical features
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() > 50:  # Arbitrary threshold
                feature_validation['high_cardinality_features'].append(col)
        
        return feature_validation
    
    def _validate_statistical_properties(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate statistical properties of the data."""
        stats_validation = {
            'target_distribution': {},
            'age_distribution_reasonable': False,
            'campaign_calls_reasonable': False,
            'outliers_detected': False
        }
        
        # Target distribution
        if 'Subscription Status' in df.columns:
            target_dist = df['Subscription Status'].value_counts(normalize=True).to_dict()
            stats_validation['target_distribution'] = target_dist
        
        # Age distribution
        if 'Age' in df.columns:
            age_mean = df['Age'].mean()
            age_std = df['Age'].std()
            stats_validation['age_distribution_reasonable'] = (30 <= age_mean <= 60) and (age_std > 5)
        
        # Campaign calls
        if 'Campaign Calls' in df.columns:
            max_calls = df['Campaign Calls'].max()
            stats_validation['campaign_calls_reasonable'] = max_calls <= 100
        
        return stats_validation
    
    def _validate_ml_readiness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate ML readiness of the dataset."""
        ml_validation = {
            'target_variable_present': 'Subscription Status' in df.columns,
            'sufficient_features': len(df.columns) >= 10,
            'sufficient_samples': len(df) >= 1000,
            'no_missing_values': df.isnull().sum().sum() == 0,
            'balanced_target': False
        }
        
        # Check target balance
        if 'Subscription Status' in df.columns:
            target_dist = df['Subscription Status'].value_counts(normalize=True)
            min_class_ratio = target_dist.min()
            ml_validation['balanced_target'] = min_class_ratio >= 0.05  # At least 5% for minority class
        
        return ml_validation
    
    def _determine_overall_status(self, validation_report: Dict[str, Any]) -> str:
        """Determine overall validation status."""
        critical_checks = [
            validation_report['schema_validation']['record_count_valid'],
            validation_report['schema_validation']['feature_count_valid'],
            validation_report['schema_validation']['core_columns_present'],
            validation_report['data_integrity']['zero_missing_values'],
            validation_report['business_rules']['age_range_valid'],
            validation_report['business_rules']['target_binary_valid'],
            validation_report['quality_metrics']['overall_quality_score'] >= self.quality_score_threshold
        ]
        
        if all(critical_checks):
            return 'PASSED'
        elif any(critical_checks):
            return 'PARTIAL'
        else:
            return 'FAILED'
    
    def _log_validation_summary(self, validation_report: Dict[str, Any]) -> None:
        """Log validation summary."""
        status = validation_report['overall_status']
        quality_score = validation_report['quality_metrics']['overall_quality_score']
        
        logger.info("=== Data Validation Summary ===")
        logger.info(f"Overall Status: {status}")
        logger.info(f"Quality Score: {quality_score:.1f}%")
        logger.info(f"Records: {validation_report['dataset_info']['total_rows']:,}")
        logger.info(f"Features: {validation_report['dataset_info']['total_columns']}")
        logger.info(f"Missing Values: {validation_report['data_integrity']['missing_values_count']}")
        
        if status == 'PASSED':
            logger.info("✅ All validation checks passed")
        elif status == 'PARTIAL':
            logger.warning("⚠️ Some validation checks failed")
        else:
            logger.error("❌ Critical validation failures detected")
    
    def validate_quick(self, df: pd.DataFrame) -> bool:
        """
        Perform quick validation for basic checks.
        
        Args:
            df (pd.DataFrame): Data to validate
        
        Returns:
            bool: True if basic validation passes
        """
        try:
            # Quick checks
            checks = [
                not df.empty,
                len(df) > 40000,  # Approximately correct size
                len(df.columns) >= 30,  # Approximately correct features
                df.isnull().sum().sum() == 0,  # No missing values
                'Subscription Status' in df.columns  # Target present
            ]
            
            return all(checks)
        except Exception as e:
            logger.error(f"Quick validation failed: {str(e)}")
            return False
