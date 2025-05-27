"""
Data Access Functions for Phase 4 Data Integration

This module provides high-level data access functions that combine
CSV loading, validation, and pipeline utilities for seamless
Phase 4 data integration workflow.

Key Features:
- High-level data loading with automatic validation
- Integrated quality checks and performance monitoring
- Seamless Phase 3 → Phase 4 → Phase 5 data flow support
- Comprehensive error handling and recovery
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path

from .csv_loader import CSVLoader, CSVLoaderError
from .data_validator import DataValidator, DataValidationError
from .pipeline_utils import PipelineUtils, PipelineUtilsError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataAccessError(Exception):
    """Custom exception for data access errors."""
    pass


def load_and_validate_data(file_path: Optional[str] = None, 
                          comprehensive_validation: bool = True,
                          performance_monitoring: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load and validate data with comprehensive checks.
    
    Args:
        file_path (Optional[str]): Path to data file. Defaults to Phase 3 output.
        comprehensive_validation (bool): Whether to perform comprehensive validation
        performance_monitoring (bool): Whether to monitor performance
    
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Loaded data and validation report
    
    Raises:
        DataAccessError: If loading or validation fails
    """
    try:
        logger.info("Starting data loading and validation...")
        
        # Initialize components
        loader = CSVLoader(file_path)
        validator = DataValidator()
        utils = PipelineUtils() if performance_monitoring else None
        
        # Load data with performance monitoring
        if performance_monitoring:
            df = utils.monitor_performance('data_loading', loader.load_data)
        else:
            df = loader.load_data()
        
        # Validate data
        validation_report = validator.validate_data(df, comprehensive=comprehensive_validation)
        
        # Check validation status
        if validation_report['overall_status'] == 'FAILED':
            raise DataAccessError("Data validation failed with critical errors")
        elif validation_report['overall_status'] == 'PARTIAL':
            logger.warning("Data validation passed with some warnings")
        else:
            logger.info("✅ Data validation passed successfully")
        
        # Add performance metrics to report
        if performance_monitoring:
            validation_report['performance_metrics'] = {
                'loader_metrics': loader.get_performance_metrics(),
                'pipeline_metrics': utils.get_performance_report()
            }
        
        logger.info(f"Data loaded successfully: {len(df)} records, {len(df.columns)} features")
        return df, validation_report
        
    except Exception as e:
        logger.error(f"Data loading and validation failed: {str(e)}")
        raise DataAccessError(f"Data loading and validation failed: {str(e)}")


def prepare_data_for_ml(df: pd.DataFrame, 
                       target_column: str = 'Subscription Status',
                       test_size: float = 0.2,
                       validation_size: float = 0.2,
                       optimize_memory: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Prepare data for machine learning with splitting and optimization.
    
    Args:
        df (pd.DataFrame): Input data
        target_column (str): Name of target column
        test_size (float): Proportion for test set
        validation_size (float): Proportion for validation set
        optimize_memory (bool): Whether to optimize memory usage
    
    Returns:
        Dict[str, pd.DataFrame]: Dictionary with train/validation/test splits
    
    Raises:
        DataAccessError: If preparation fails
    """
    try:
        logger.info("Preparing data for machine learning...")
        
        utils = PipelineUtils()
        
        # Optimize memory if requested
        if optimize_memory:
            df = utils.monitor_performance('memory_optimization', utils.optimize_memory_usage, df)
        
        # Split data
        splits = utils.monitor_performance(
            'data_splitting', 
            utils.split_data, 
            df, 
            target_column=target_column,
            test_size=test_size,
            validation_size=validation_size
        )
        
        logger.info("Data preparation completed successfully")
        logger.info(f"Train: {len(splits['train'])} records")
        logger.info(f"Test: {len(splits['test'])} records")
        if 'validation' in splits:
            logger.info(f"Validation: {len(splits['validation'])} records")
        
        return splits
        
    except Exception as e:
        logger.error(f"Data preparation failed: {str(e)}")
        raise DataAccessError(f"Data preparation failed: {str(e)}")


def validate_phase3_continuity(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate Phase 3 → Phase 4 data flow continuity.
    
    Args:
        df (pd.DataFrame): Data to validate
    
    Returns:
        Dict[str, Any]: Continuity validation report
    
    Raises:
        DataAccessError: If continuity validation fails
    """
    try:
        logger.info("Validating Phase 3 → Phase 4 data flow continuity...")
        
        validator = DataValidator()
        
        # Perform focused validation on Phase 3 preservation
        validation_report = validator.validate_data(df, comprehensive=False)
        
        # Extract continuity-specific metrics
        continuity_report = {
            'phase3_preservation': validation_report['phase3_preservation'],
            'data_integrity': validation_report['data_integrity'],
            'business_rules': validation_report['business_rules'],
            'schema_validation': validation_report['schema_validation'],
            'quality_score': validation_report['quality_metrics']['overall_quality_score'],
            'continuity_status': 'PASSED' if validation_report['overall_status'] == 'PASSED' else 'FAILED'
        }
        
        # Log continuity status
        if continuity_report['continuity_status'] == 'PASSED':
            logger.info("✅ Phase 3 → Phase 4 continuity validated successfully")
        else:
            logger.error("❌ Phase 3 → Phase 4 continuity validation failed")
        
        return continuity_report
        
    except Exception as e:
        logger.error(f"Continuity validation failed: {str(e)}")
        raise DataAccessError(f"Continuity validation failed: {str(e)}")


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive data summary for Phase 4.
    
    Args:
        df (pd.DataFrame): Data to summarize
    
    Returns:
        Dict[str, Any]: Comprehensive data summary
    """
    try:
        logger.info("Generating data summary...")
        
        summary = {
            'basic_info': {
                'total_records': len(df),
                'total_features': len(df.columns),
                'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2),
                'file_size_estimate_mb': round(len(df) * len(df.columns) * 8 / 1024**2, 2)  # Rough estimate
            },
            'data_quality': {
                'missing_values': df.isnull().sum().sum(),
                'duplicate_records': df.duplicated().sum(),
                'completeness_percent': round((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 2)
            },
            'feature_types': {
                'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
                'categorical_features': len(df.select_dtypes(include=['object']).columns),
                'datetime_features': len(df.select_dtypes(include=['datetime']).columns)
            },
            'target_analysis': {},
            'phase4_readiness': {
                'ml_ready': True,
                'sufficient_samples': len(df) >= 1000,
                'sufficient_features': len(df.columns) >= 10,
                'no_missing_values': df.isnull().sum().sum() == 0
            }
        }
        
        # Target analysis if present
        if 'Subscription Status' in df.columns:
            target_dist = df['Subscription Status'].value_counts(normalize=True)
            summary['target_analysis'] = {
                'target_distribution': target_dist.to_dict(),
                'class_balance': target_dist.min() / target_dist.max(),
                'minority_class_ratio': target_dist.min()
            }
        
        # Update ML readiness based on target analysis
        if summary['target_analysis']:
            summary['phase4_readiness']['balanced_target'] = summary['target_analysis']['minority_class_ratio'] >= 0.05
        
        logger.info("Data summary generated successfully")
        return summary
        
    except Exception as e:
        logger.error(f"Data summary generation failed: {str(e)}")
        raise DataAccessError(f"Data summary generation failed: {str(e)}")


def quick_data_check(file_path: Optional[str] = None) -> bool:
    """
    Perform quick data check for basic validation.
    
    Args:
        file_path (Optional[str]): Path to data file
    
    Returns:
        bool: True if basic checks pass
    """
    try:
        logger.info("Performing quick data check...")
        
        # Quick load and validate
        loader = CSVLoader(file_path)
        df = loader.load_data(validate=False)  # Skip detailed validation for speed
        
        validator = DataValidator()
        is_valid = validator.validate_quick(df)
        
        if is_valid:
            logger.info("✅ Quick data check passed")
        else:
            logger.warning("⚠️ Quick data check failed")
        
        return is_valid
        
    except Exception as e:
        logger.error(f"Quick data check failed: {str(e)}")
        return False


def load_sample_data(n_rows: int = 1000, 
                    columns: Optional[List[str]] = None,
                    file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load sample data for exploration or testing.
    
    Args:
        n_rows (int): Number of rows to load
        columns (Optional[List[str]]): Specific columns to load
        file_path (Optional[str]): Path to data file
    
    Returns:
        pd.DataFrame: Sample data
    
    Raises:
        DataAccessError: If sample loading fails
    """
    try:
        logger.info(f"Loading sample data: {n_rows} rows")
        
        loader = CSVLoader(file_path)
        
        if columns:
            df = loader.load_columns(columns)
            df = df.head(n_rows)
        else:
            df = loader.load_sample(n_rows)
        
        logger.info(f"Sample data loaded: {len(df)} rows, {len(df.columns)} columns")
        return df
        
    except Exception as e:
        logger.error(f"Sample data loading failed: {str(e)}")
        raise DataAccessError(f"Sample data loading failed: {str(e)}")


# Convenience functions for common operations
def load_phase3_output() -> pd.DataFrame:
    """Load Phase 3 output data with default settings."""
    df, _ = load_and_validate_data()
    return df


def validate_phase3_output() -> Dict[str, Any]:
    """Validate Phase 3 output data."""
    df, validation_report = load_and_validate_data()
    return validation_report


def prepare_ml_pipeline() -> Dict[str, pd.DataFrame]:
    """Prepare complete ML pipeline with default settings."""
    df = load_phase3_output()
    return prepare_data_for_ml(df)
