"""
Phase 5 Feature Engineering Module

This module provides production-ready feature engineering capabilities for the banking marketing dataset,
building upon Phase 4's validated data integration infrastructure.

Key Features:
1. Age binning for optimal model performance (young/middle/senior categories)
2. Education-occupation interactions for high-value customer segments
3. Contact recency features leveraging Phase 3's No_Previous_Contact flag
4. Campaign intensity features for optimal contact frequency analysis
5. Phase 4 integration with continuous validation and quality monitoring

Performance Standards:
- >97K records/second processing (Phase 4 achieved 437K+ records/second)
- Memory optimization for large feature sets
- Continuous validation after each feature engineering step
- Error handling patterns from Phase 4

Business Value:
- Features directly impact subscription prediction accuracy
- Leverages Phase 3 cleaned data through Phase 4 production-ready integration
- Clear business rationale for each transformation

Usage:
    from feature_engineering import FeatureEngineer, engineer_features_pipeline
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Load data using Phase 4 integration
    from data_integration import prepare_ml_pipeline, validate_phase3_continuity
    
    # Get prepared data splits
    splits = prepare_ml_pipeline()
    train_df = splits['train']
    
    # Validate continuity before feature engineering
    continuity_report = validate_phase3_continuity(train_df)
    assert continuity_report['continuity_status'] == 'PASSED'
    
    # Engineer features
    featured_df = engineer.engineer_features(train_df)
    
    # Or use high-level pipeline
    featured_df = engineer_features_pipeline()
"""

from .feature_engineer import FeatureEngineer
from .business_features import BusinessFeatureCreator
from .transformations import FeatureTransformer
from .pipeline import engineer_features_pipeline, save_featured_data

__version__ = "1.0.0"
__author__ = "AI-Vive Banking Team"

# Module-level constants
FEATURED_OUTPUT_PATH = "data/featured/featured-db.csv"
PERFORMANCE_STANDARD = 97000  # records per second
AGE_BINS = [18, 35, 55, 100]
AGE_LABELS = [1, 2, 3]  # young, middle, senior (numeric for model performance)
CAMPAIGN_INTENSITY_BINS = [0, 2, 5, 50]
CAMPAIGN_INTENSITY_LABELS = ["low", "medium", "high"]

# Business feature specifications
BUSINESS_FEATURES = {
    "age_bin": {
        "description": "Age categories for optimal model performance",
        "bins": AGE_BINS,
        "labels": AGE_LABELS,
        "business_rationale": "Young (18-35), Middle (36-55), Senior (56-100) segments for targeted marketing"
    },
    "education_job_segment": {
        "description": "High-value customer segments from education-occupation interactions",
        "business_rationale": "Identify premium customer segments for focused campaigns"
    },
    "recent_contact_flag": {
        "description": "Recent contact effect on subscription likelihood",
        "business_rationale": "Leverage contact history for improved targeting"
    },
    "campaign_intensity": {
        "description": "Optimal contact frequency patterns",
        "bins": CAMPAIGN_INTENSITY_BINS,
        "labels": CAMPAIGN_INTENSITY_LABELS,
        "business_rationale": "Low (1-2), Medium (3-5), High (6+) contact strategies"
    },
    "high_intensity_flag": {
        "description": "Binary flag for intensive campaign contacts",
        "threshold": 6,
        "business_rationale": "Identify customers receiving intensive contact for analysis"
    }
}

# Quality monitoring thresholds
QUALITY_THRESHOLDS = {
    "missing_values_max": 0,
    "performance_min_records_per_second": PERFORMANCE_STANDARD,
    "data_integrity_score_min": 100,
    "feature_creation_success_rate_min": 100
}

__all__ = [
    "FeatureEngineer",
    "BusinessFeatureCreator", 
    "FeatureTransformer",
    "engineer_features_pipeline",
    "save_featured_data",
    "FEATURED_OUTPUT_PATH",
    "PERFORMANCE_STANDARD",
    "BUSINESS_FEATURES",
    "QUALITY_THRESHOLDS"
]
