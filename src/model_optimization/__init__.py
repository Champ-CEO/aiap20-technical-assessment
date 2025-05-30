"""
Phase 9 Model Optimization Module

Provides comprehensive model optimization capabilities including ensemble methods,
hyperparameter optimization, business criteria optimization, and performance monitoring.

Key Components:
- EnsembleOptimizer: Model combination and ensemble methods
- HyperparameterOptimizer: Parameter tuning for enhanced performance
- BusinessCriteriaOptimizer: ROI optimization with customer segments
- PerformanceMonitor: Drift detection and monitoring systems
- ProductionReadinessValidator: Production deployment validation
- EnsembleValidator: Ensemble performance validation
- FeatureOptimizer: Feature selection and optimization
- DeploymentFeasibilityValidator: Deployment feasibility assessment

Integration:
- Phase 8 evaluation results and trained models
- Customer segment awareness (Premium: 6,977% ROI, Standard: 5,421% ROI, Basic: 3,279% ROI)
- Performance standard: >97K records/second
"""

from .ensemble_optimizer import EnsembleOptimizer

# Import other modules as they are implemented
try:
    from .hyperparameter_optimizer import HyperparameterOptimizer
except ImportError:
    HyperparameterOptimizer = None

try:
    from .business_criteria_optimizer import BusinessCriteriaOptimizer
except ImportError:
    BusinessCriteriaOptimizer = None

try:
    from .performance_monitor import PerformanceMonitor
except ImportError:
    PerformanceMonitor = None

try:
    from .production_readiness_validator import ProductionReadinessValidator
except ImportError:
    ProductionReadinessValidator = None

try:
    from .ensemble_validator import EnsembleValidator
except ImportError:
    EnsembleValidator = None

try:
    from .feature_optimizer import FeatureOptimizer
except ImportError:
    FeatureOptimizer = None

try:
    from .deployment_feasibility_validator import DeploymentFeasibilityValidator
except ImportError:
    DeploymentFeasibilityValidator = None

__all__ = [
    "EnsembleOptimizer",
    "HyperparameterOptimizer",
    "BusinessCriteriaOptimizer",
    "PerformanceMonitor",
    "ProductionReadinessValidator",
    "EnsembleValidator",
    "FeatureOptimizer",
    "DeploymentFeasibilityValidator",
]
