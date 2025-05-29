"""
Phase 8 Model Evaluation Module

Comprehensive model evaluation framework for Phase 8 implementation.
Provides business-relevant evaluation, model comparison, and production deployment validation.

Key Components:
- ModelEvaluator: Performance metrics calculation
- ModelComparator: Model ranking and comparison
- BusinessMetricsCalculator: ROI and business value analysis
- ProductionDeploymentValidator: 3-tier deployment strategy
- ModelVisualizer: Performance charts and feature importance plots
- ReportGenerator: Evaluation report generation
- EvaluationPipeline: End-to-end evaluation workflow

Integration:
- Phase 7 trained models from trained_models/ directory
- Phase 5 featured data from data/featured/featured-db.csv
- Customer segment awareness (Premium: 31.6%, Standard: 57.7%, Basic: 10.7%)
- Performance standard: >97K records/second
"""

from .evaluator import ModelEvaluator
from .comparator import ModelComparator
from .business_calculator import BusinessMetricsCalculator
from .deployment_validator import ProductionDeploymentValidator
from .visualizer import ModelVisualizer
from .reporter import ReportGenerator
from .pipeline import EvaluationPipeline
from .feature_analyzer import FeatureImportanceAnalyzer
from .performance_monitor import PerformanceMonitor
from .ensemble_evaluator import EnsembleEvaluator

# Main evaluation function
from .pipeline import evaluate_models

__all__ = [
    'ModelEvaluator',
    'ModelComparator', 
    'BusinessMetricsCalculator',
    'ProductionDeploymentValidator',
    'ModelVisualizer',
    'ReportGenerator',
    'EvaluationPipeline',
    'FeatureImportanceAnalyzer',
    'PerformanceMonitor',
    'EnsembleEvaluator',
    'evaluate_models'
]

__version__ = '1.0.0'
__author__ = 'Phase 8 Model Evaluation Team'
