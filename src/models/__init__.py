"""
Phase 7 Model Implementation Module

Implements 5 classifiers for term deposit subscription prediction following TDD approach.
Provides comprehensive model training functionality with business logic integration.

This module implements:
1. Logistic Regression (interpretable baseline for marketing insights)
2. Random Forest (top Phase 6 performer with feature importance)
3. Gradient Boosting/XGBoost (advanced patterns with categorical support)
4. Naive Bayes (probabilistic estimates for marketing)
5. Support Vector Machine (clear decision boundaries)

Key Features:
- Phase 6 model preparation pipeline integration
- 45-feature dataset compatibility (33 original + 12 engineered)
- Categorical encoding with LabelEncoder pipeline
- Business metrics with customer segment awareness
- Cross-validation with segment preservation
- Performance monitoring (>97K records/second standard)
- Model serialization with feature importance analysis

Integration Points:
- Phase 5 feature engineering pipeline (data/featured/featured-db.csv)
- Phase 6 model preparation (DataLoader, DataSplitter, CrossValidator, etc.)
- Customer segment business logic (Premium, Standard, Basic)
- Engineered features (age_bin, customer_value_segment, campaign_intensity, etc.)
"""

from .classifier1 import LogisticRegressionClassifier
from .classifier2 import RandomForestClassifier
from .classifier3 import GradientBoostingClassifier
from .classifier4 import NaiveBayesClassifier
from .classifier5 import SVMClassifier
from .train_model import train_model, ModelTrainer

# Model types mapping for factory pattern
MODEL_TYPES = {
    'LogisticRegression': LogisticRegressionClassifier,
    'RandomForest': RandomForestClassifier,
    'GradientBoosting': GradientBoostingClassifier,
    'NaiveBayes': NaiveBayesClassifier,
    'SVM': SVMClassifier,
}

# Expected constants from Phase 6
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
    'customer_value_segment', 
    'campaign_intensity',
    'education_job_segment',
    'recent_contact_flag',
    'contact_effectiveness_score',
    'financial_risk_score',
    'risk_category',
    'is_high_risk',
    'high_intensity_flag',
    'is_premium_customer',
    'contact_recency',
]

__version__ = "1.0.0"
__author__ = "AI-Vive Banking Team - Phase 7"

# Export all main classes and functions
__all__ = [
    # Classifier classes
    'LogisticRegressionClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier',
    'NaiveBayesClassifier', 'SVMClassifier',
    
    # Training functions
    'train_model', 'ModelTrainer',
    
    # Constants
    'MODEL_TYPES', 'EXPECTED_RECORD_COUNT', 'EXPECTED_TOTAL_FEATURES', 
    'EXPECTED_SUBSCRIPTION_RATE', 'PERFORMANCE_STANDARD', 'CUSTOMER_SEGMENT_RATES',
    'ENGINEERED_FEATURES'
]
