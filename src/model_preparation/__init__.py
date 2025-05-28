"""
Model Preparation Module

Phase 6 Model Preparation implementation following TDD approach.
Provides comprehensive model preparation functionality with business logic integration.

This module implements:
1. Data loading and validation with Phase 5 integration
2. Stratified splitting with customer segment awareness
3. Cross-validation with class balance preservation
4. Business metrics calculation with segment analysis
5. Model serialization with 45-feature schema support
6. Performance monitoring (>97K records/second standard)

Key Components:
- DataLoader: Phase 5 data loading with fallback support
- DataSplitter: Stratified splitting preserving segments and class distribution
- CrossValidator: 5-fold CV with segment awareness
- BusinessMetrics: ROI calculation with customer segment analysis
- ModelManager: Serialization and model factory pattern
- PerformanceMonitor: Processing speed validation

Integration Points:
- Phase 5 feature engineering pipeline
- Phase 4 data integration fallback
- Customer segment business logic
- Engineered features (age_bin, customer_value_segment, etc.)
"""

from .data_loader import DataLoader, ModelPreparationError
from .data_splitter import DataSplitter, StratifiedSplitter
from .cross_validator import CrossValidator, SegmentAwareCrossValidator
from .business_metrics import BusinessMetrics, SegmentMetrics, ROICalculator
from .model_manager import ModelManager, ModelFactory, ModelSerializer
from .performance_monitor import PerformanceMonitor, ProcessingTimer

# Module constants
EXPECTED_RECORD_COUNT = 41188
EXPECTED_TOTAL_FEATURES = 45
EXPECTED_SUBSCRIPTION_RATE = 0.113
PERFORMANCE_STANDARD = 97000  # records per second

# Customer segment distributions from Phase 5
CUSTOMER_SEGMENT_RATES = {
    'Premium': 0.316,  # 31.6%
    'Standard': 0.577,  # 57.7%
    'Basic': 0.107     # 10.7%
}

# Expected engineered features from Phase 5
ENGINEERED_FEATURES = [
    'age_bin',
    'education_job_segment', 
    'customer_value_segment',
    'recent_contact_flag',
    'campaign_intensity',
    'contact_effectiveness_score',
    'financial_risk_score',
    'risk_category',
    'is_high_risk',
    'high_intensity_flag',
    'is_premium_customer',
    'contact_recency'
]

# High-level convenience functions
def load_phase5_data(use_fallback=True, validate_schema=True):
    """
    Load Phase 5 featured data with automatic fallback.
    
    Args:
        use_fallback (bool): Whether to use Phase 4 fallback if Phase 5 data unavailable
        validate_schema (bool): Whether to validate feature schema
    
    Returns:
        pd.DataFrame: Loaded and validated dataset
    """
    loader = DataLoader()
    return loader.load_data(use_fallback=use_fallback, validate_schema=validate_schema)

def prepare_model_data(df, target_column='Subscription Status', test_size=0.2, 
                      validation_size=0.2, preserve_segments=True):
    """
    Prepare data for model training with stratified splitting.
    
    Args:
        df (pd.DataFrame): Input dataset
        target_column (str): Name of target column
        test_size (float): Proportion for test set
        validation_size (float): Proportion for validation set
        preserve_segments (bool): Whether to preserve customer segment distributions
    
    Returns:
        dict: Dictionary with train/validation/test splits
    """
    splitter = DataSplitter()
    return splitter.split_data(
        df, target_column=target_column, test_size=test_size,
        validation_size=validation_size, preserve_segments=preserve_segments
    )

def setup_cross_validation(n_splits=5, preserve_segments=True, random_state=42):
    """
    Set up cross-validation with segment awareness.
    
    Args:
        n_splits (int): Number of CV folds
        preserve_segments (bool): Whether to preserve segment balance
        random_state (int): Random state for reproducibility
    
    Returns:
        CrossValidator: Configured cross-validator
    """
    return CrossValidator(
        n_splits=n_splits, preserve_segments=preserve_segments, 
        random_state=random_state
    )

def calculate_business_metrics(y_true, y_pred, segments=None, campaign_intensity=None):
    """
    Calculate business metrics with segment awareness.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        segments (array-like, optional): Customer segments
        campaign_intensity (array-like, optional): Campaign intensity levels
    
    Returns:
        dict: Business metrics including ROI by segment
    """
    metrics = BusinessMetrics()
    return metrics.calculate_comprehensive_metrics(
        y_true, y_pred, segments=segments, campaign_intensity=campaign_intensity
    )

__version__ = "1.0.0"
__author__ = "AI-Vive Banking Team - Phase 6"

# Export all main classes and functions
__all__ = [
    # Main classes
    'DataLoader', 'DataSplitter', 'CrossValidator', 'BusinessMetrics', 
    'ModelManager', 'PerformanceMonitor',
    
    # Specialized classes
    'StratifiedSplitter', 'SegmentAwareCrossValidator', 'SegmentMetrics', 
    'ROICalculator', 'ModelFactory', 'ModelSerializer', 'ProcessingTimer',
    
    # Convenience functions
    'load_phase5_data', 'prepare_model_data', 'setup_cross_validation', 
    'calculate_business_metrics',
    
    # Constants
    'EXPECTED_RECORD_COUNT', 'EXPECTED_TOTAL_FEATURES', 'EXPECTED_SUBSCRIPTION_RATE',
    'PERFORMANCE_STANDARD', 'CUSTOMER_SEGMENT_RATES', 'ENGINEERED_FEATURES',
    
    # Exceptions
    'ModelPreparationError'
]
