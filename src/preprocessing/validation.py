"""
Data Validation Module for Banking Marketing Dataset

This module provides comprehensive data validation functions to ensure
data quality and business rule compliance based on Phase 2 EDA findings.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    """
    Comprehensive data validator for banking marketing dataset.
    
    Based on Phase 2 EDA findings, this class validates:
    - Age ranges (18-100 years)
    - Campaign calls limits (0-50 realistic range)
    - Previous contact days (0-999 range)
    - Data type consistency
    - Business rule compliance
    """
    
    def __init__(self):
        """Initialize the data validator with business rules."""
        self.business_rules = {
            'age_min': 18,
            'age_max': 100,
            'campaign_calls_max': 50,
            'campaign_calls_min': 0,
            'previous_contact_days_max': 999,
            'previous_contact_days_min': 0
        }
        
        self.expected_columns = [
            'Client ID', 'Age', 'Occupation', 'Marital Status', 'Education Level',
            'Credit Default', 'Housing Loan', 'Personal Loan', 'Contact Method',
            'Campaign Calls', 'Previous Contact Days', 'Subscription Status'
        ]
        
        self.categorical_columns = [
            'Occupation', 'Marital Status', 'Education Level', 'Credit Default',
            'Housing Loan', 'Personal Loan', 'Contact Method', 'Subscription Status'
        ]
        
        self.numerical_columns = ['Age', 'Campaign Calls', 'Previous Contact Days']
    
    def validate_data_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate basic data structure and schema.
        
        Args:
            df: Input DataFrame to validate
            
        Returns:
            Dict containing validation results
        """
        validation_results = {
            'structure_valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Check column presence
        missing_columns = set(self.expected_columns) - set(df.columns)
        if missing_columns:
            validation_results['structure_valid'] = False
            validation_results['issues'].append(f"Missing columns: {missing_columns}")
        
        # Check for extra columns
        extra_columns = set(df.columns) - set(self.expected_columns)
        if extra_columns:
            validation_results['warnings'].append(f"Extra columns found: {extra_columns}")
        
        # Check data types
        for col in self.numerical_columns:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                validation_results['warnings'].append(f"Column '{col}' should be numeric but is {df[col].dtype}")
        
        logger.info(f"Data structure validation completed. Valid: {validation_results['structure_valid']}")
        return validation_results
    
    def validate_age_column(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate age column for business rules compliance.
        
        Args:
            df: DataFrame with Age column
            
        Returns:
            Dict containing age validation results
        """
        validation_results = {
            'age_valid': True,
            'issues': [],
            'outliers': [],
            'statistics': {}
        }
        
        if 'Age' not in df.columns:
            validation_results['age_valid'] = False
            validation_results['issues'].append("Age column not found")
            return validation_results
        
        # Check for non-numeric ages (if still in text format)
        if df['Age'].dtype == 'object':
            validation_results['issues'].append("Age column is still in text format - needs conversion")
        
        # For numeric ages, validate ranges
        if pd.api.types.is_numeric_dtype(df['Age']):
            age_stats = df['Age'].describe()
            validation_results['statistics'] = age_stats.to_dict()
            
            # Check for outliers
            outliers_low = df[df['Age'] < self.business_rules['age_min']]
            outliers_high = df[df['Age'] > self.business_rules['age_max']]
            
            if len(outliers_low) > 0:
                validation_results['outliers'].append(f"{len(outliers_low)} ages below {self.business_rules['age_min']}")
            
            if len(outliers_high) > 0:
                validation_results['outliers'].append(f"{len(outliers_high)} ages above {self.business_rules['age_max']}")
            
            # Check for impossible values (like 150 years from EDA)
            impossible_ages = df[df['Age'] > 120]
            if len(impossible_ages) > 0:
                validation_results['issues'].append(f"{len(impossible_ages)} impossible age values (>120 years)")
        
        logger.info(f"Age validation completed. Issues: {len(validation_results['issues'])}")
        return validation_results
    
    def validate_campaign_calls(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate campaign calls for realistic business ranges.
        
        Args:
            df: DataFrame with Campaign Calls column
            
        Returns:
            Dict containing campaign calls validation results
        """
        validation_results = {
            'campaign_calls_valid': True,
            'issues': [],
            'outliers': [],
            'statistics': {}
        }
        
        if 'Campaign Calls' not in df.columns:
            validation_results['campaign_calls_valid'] = False
            validation_results['issues'].append("Campaign Calls column not found")
            return validation_results
        
        campaign_calls = df['Campaign Calls']
        validation_results['statistics'] = campaign_calls.describe().to_dict()
        
        # Check for negative values (found in EDA)
        negative_calls = df[campaign_calls < 0]
        if len(negative_calls) > 0:
            validation_results['issues'].append(f"{len(negative_calls)} negative campaign call values")
        
        # Check for unrealistic high values
        high_calls = df[campaign_calls > self.business_rules['campaign_calls_max']]
        if len(high_calls) > 0:
            validation_results['outliers'].append(f"{len(high_calls)} campaign calls above {self.business_rules['campaign_calls_max']}")
        
        logger.info(f"Campaign calls validation completed. Issues: {len(validation_results['issues'])}")
        return validation_results
    
    def validate_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate missing values patterns and counts.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dict containing missing values analysis
        """
        missing_analysis = {
            'total_missing': 0,
            'missing_by_column': {},
            'missing_percentage': {},
            'high_missing_columns': []
        }
        
        missing_counts = df.isnull().sum()
        missing_analysis['total_missing'] = missing_counts.sum()
        missing_analysis['missing_by_column'] = missing_counts.to_dict()
        
        # Calculate percentages
        total_rows = len(df)
        for col, count in missing_counts.items():
            percentage = (count / total_rows) * 100
            missing_analysis['missing_percentage'][col] = round(percentage, 2)
            
            # Flag high missing columns (>20%)
            if percentage > 20:
                missing_analysis['high_missing_columns'].append(col)
        
        logger.info(f"Missing values validation completed. Total missing: {missing_analysis['total_missing']}")
        return missing_analysis
    
    def validate_special_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate special values like 'unknown', '999', etc.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dict containing special values analysis
        """
        special_values_analysis = {
            'unknown_values': {},
            'special_numeric_values': {},
            'total_special_values': 0
        }
        
        # Check for 'unknown' in categorical columns
        for col in self.categorical_columns:
            if col in df.columns:
                unknown_count = (df[col] == 'unknown').sum()
                if unknown_count > 0:
                    special_values_analysis['unknown_values'][col] = unknown_count
        
        # Check for special numeric values (999 in Previous Contact Days)
        if 'Previous Contact Days' in df.columns:
            special_999 = (df['Previous Contact Days'] == 999).sum()
            if special_999 > 0:
                special_values_analysis['special_numeric_values']['Previous Contact Days (999)'] = special_999
        
        # Calculate total
        total_unknown = sum(special_values_analysis['unknown_values'].values())
        total_special_numeric = sum(special_values_analysis['special_numeric_values'].values())
        special_values_analysis['total_special_values'] = total_unknown + total_special_numeric
        
        logger.info(f"Special values validation completed. Total: {special_values_analysis['total_special_values']}")
        return special_values_analysis
    
    def generate_validation_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dict containing complete validation report
        """
        logger.info("Starting comprehensive data validation...")
        
        validation_report = {
            'dataset_info': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2)
            },
            'structure_validation': self.validate_data_structure(df),
            'age_validation': self.validate_age_column(df),
            'campaign_calls_validation': self.validate_campaign_calls(df),
            'missing_values_analysis': self.validate_missing_values(df),
            'special_values_analysis': self.validate_special_values(df),
            'overall_quality_score': 0
        }
        
        # Calculate overall quality score
        issues_count = (
            len(validation_report['structure_validation']['issues']) +
            len(validation_report['age_validation']['issues']) +
            len(validation_report['campaign_calls_validation']['issues'])
        )
        
        # Simple quality score (100 - issues penalty)
        validation_report['overall_quality_score'] = max(0, 100 - (issues_count * 10))
        
        logger.info(f"Validation report generated. Quality score: {validation_report['overall_quality_score']}")
        return validation_report
