"""
Production Deployment Validator Module

Implements 3-tier production deployment strategy validation.
Validates Primary: GradientBoosting (89.8%), Secondary: RandomForest, Tertiary: NaiveBayes (255K records/sec).

Key Features:
- 3-tier deployment strategy validation
- Performance baseline validation (89.8% accuracy)
- Speed standard validation (255K records/second)
- Model drift detection
- Production readiness assessment
"""

import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Production deployment strategy from Phase 7
PRODUCTION_STRATEGY = {
    "Primary": "GradientBoosting",    # 89.8% accuracy
    "Secondary": "RandomForest",      # 84.6% balanced
    "Tertiary": "NaiveBayes",        # 255K records/sec speed
}

# Performance standards
ACCURACY_BASELINE = 0.898  # 89.8% from GradientBoosting
SPEED_STANDARD = 255000    # 255K records/second from NaiveBayes
DRIFT_THRESHOLD = 0.05     # 5% accuracy drift threshold

# Phase 7 baseline performance
PHASE7_BASELINES = {
    "GradientBoosting": 0.8978,
    "RandomForest": 0.8460,
    "NaiveBayes": 0.8954,
    "LogisticRegression": 0.7147,
    "SVM": 0.7879
}


class ProductionDeploymentValidator:
    """
    Production deployment validator for Phase 8 implementation.
    
    Validates model readiness for production deployment based on
    3-tier strategy and performance standards.
    """
    
    def __init__(self):
        """Initialize ProductionDeploymentValidator."""
        self.validation_results = {}
        self.deployment_recommendations = {}
        
    def validate_deployment_strategy(self, evaluation_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate the 3-tier production deployment strategy.
        
        Args:
            evaluation_results (Dict): Model evaluation results
            
        Returns:
            Dict[str, Any]: Deployment strategy validation results
        """
        start_time = time.time()
        
        validation_results = {}
        
        # Validate each tier of the deployment strategy
        for tier, expected_model in PRODUCTION_STRATEGY.items():
            tier_validation = self._validate_tier(tier, expected_model, evaluation_results)
            validation_results[tier] = tier_validation
        
        # Overall strategy assessment
        strategy_assessment = self._assess_overall_strategy(validation_results)
        
        # Alternative strategy recommendations
        alternative_strategy = self._recommend_alternative_strategy(evaluation_results)
        
        # Compile results
        results = {
            'tier_validation': validation_results,
            'strategy_assessment': strategy_assessment,
            'alternative_strategy': alternative_strategy,
            'deployment_readiness': self._assess_deployment_readiness(validation_results),
            'validation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Performance tracking
        validation_time = time.time() - start_time
        results['validation_performance'] = {
            'validation_time': validation_time,
            'tiers_validated': len(PRODUCTION_STRATEGY)
        }
        
        self.validation_results = results
        logger.info(f"Deployment strategy validation completed in {validation_time:.2f}s")
        
        return results
    
    def _validate_tier(self, tier: str, model_name: str, 
                      evaluation_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Validate a specific tier of the deployment strategy."""
        
        if model_name not in evaluation_results or evaluation_results[model_name] is None:
            return {
                'model': model_name,
                'available': False,
                'validation_status': 'FAILED',
                'reason': 'Model not available or evaluation failed'
            }
        
        results = evaluation_results[model_name]
        
        # Extract metrics
        accuracy = results.get('accuracy', 0)
        performance_metrics = results.get('performance', {})
        records_per_second = performance_metrics.get('records_per_second', 0)
        
        # Tier-specific validation criteria
        validation_criteria = self._get_tier_criteria(tier)
        
        # Validate against criteria
        validation_checks = {}
        
        # Accuracy validation
        validation_checks['accuracy_check'] = {
            'required': validation_criteria['min_accuracy'],
            'actual': accuracy,
            'passed': accuracy >= validation_criteria['min_accuracy']
        }
        
        # Speed validation
        validation_checks['speed_check'] = {
            'required': validation_criteria['min_speed'],
            'actual': records_per_second,
            'passed': records_per_second >= validation_criteria['min_speed']
        }
        
        # Drift validation (compared to Phase 7 baseline)
        if model_name in PHASE7_BASELINES:
            baseline_accuracy = PHASE7_BASELINES[model_name]
            accuracy_drift = abs(accuracy - baseline_accuracy)
            
            validation_checks['drift_check'] = {
                'baseline': baseline_accuracy,
                'current': accuracy,
                'drift': accuracy_drift,
                'threshold': DRIFT_THRESHOLD,
                'passed': accuracy_drift <= DRIFT_THRESHOLD
            }
        
        # Overall tier validation
        all_checks_passed = all(check.get('passed', False) for check in validation_checks.values())
        
        tier_result = {
            'model': model_name,
            'tier': tier,
            'available': True,
            'validation_checks': validation_checks,
            'validation_status': 'PASSED' if all_checks_passed else 'FAILED',
            'tier_criteria': validation_criteria,
            'deployment_ready': all_checks_passed
        }
        
        return tier_result
    
    def _get_tier_criteria(self, tier: str) -> Dict[str, float]:
        """Get validation criteria for each tier."""
        
        criteria = {
            'Primary': {
                'min_accuracy': ACCURACY_BASELINE,      # Must meet baseline
                'min_speed': 50000,                     # Moderate speed requirement
                'priority': 'accuracy'
            },
            'Secondary': {
                'min_accuracy': ACCURACY_BASELINE * 0.95,  # 95% of baseline
                'min_speed': 75000,                        # Higher speed requirement
                'priority': 'balanced'
            },
            'Tertiary': {
                'min_accuracy': ACCURACY_BASELINE * 0.90,  # 90% of baseline
                'min_speed': SPEED_STANDARD,               # Must meet speed standard
                'priority': 'speed'
            }
        }
        
        return criteria.get(tier, {
            'min_accuracy': 0.7,
            'min_speed': 10000,
            'priority': 'unknown'
        })
    
    def _assess_overall_strategy(self, tier_validations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Assess the overall deployment strategy."""
        
        # Count successful tiers
        successful_tiers = sum(1 for validation in tier_validations.values() 
                             if validation.get('validation_status') == 'PASSED')
        total_tiers = len(tier_validations)
        
        # Determine strategy status
        if successful_tiers == total_tiers:
            strategy_status = 'FULLY_VALIDATED'
            strategy_health = 'EXCELLENT'
        elif successful_tiers >= 2:
            strategy_status = 'PARTIALLY_VALIDATED'
            strategy_health = 'GOOD'
        elif successful_tiers >= 1:
            strategy_status = 'MINIMALLY_VALIDATED'
            strategy_health = 'ACCEPTABLE'
        else:
            strategy_status = 'VALIDATION_FAILED'
            strategy_health = 'POOR'
        
        # Identify critical issues
        critical_issues = []
        for tier, validation in tier_validations.items():
            if validation.get('validation_status') == 'FAILED':
                if tier == 'Primary':
                    critical_issues.append(f"Primary model ({validation.get('model')}) validation failed")
                else:
                    critical_issues.append(f"{tier} model ({validation.get('model')}) validation failed")
        
        assessment = {
            'strategy_status': strategy_status,
            'strategy_health': strategy_health,
            'successful_tiers': successful_tiers,
            'total_tiers': total_tiers,
            'success_rate': successful_tiers / total_tiers,
            'critical_issues': critical_issues,
            'deployment_recommendation': self._get_deployment_recommendation(strategy_status)
        }
        
        return assessment
    
    def _recommend_alternative_strategy(self, evaluation_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Recommend alternative deployment strategy based on current results."""
        
        # Rank models by overall performance
        model_scores = {}
        
        for model_name, results in evaluation_results.items():
            if results is None:
                continue
                
            accuracy = results.get('accuracy', 0)
            performance_metrics = results.get('performance', {})
            records_per_second = performance_metrics.get('records_per_second', 0)
            
            # Calculate composite score (weighted: 70% accuracy, 30% speed)
            max_speed = 300000  # Normalize speed to reasonable max
            normalized_speed = min(records_per_second / max_speed, 1.0)
            
            composite_score = 0.7 * accuracy + 0.3 * normalized_speed
            model_scores[model_name] = {
                'composite_score': composite_score,
                'accuracy': accuracy,
                'speed': records_per_second
            }
        
        # Sort by composite score
        ranked_models = sorted(model_scores.items(), key=lambda x: x[1]['composite_score'], reverse=True)
        
        # Recommend new strategy
        alternative_strategy = {}
        if len(ranked_models) >= 1:
            alternative_strategy['Primary'] = ranked_models[0][0]
        if len(ranked_models) >= 2:
            alternative_strategy['Secondary'] = ranked_models[1][0]
        if len(ranked_models) >= 3:
            alternative_strategy['Tertiary'] = ranked_models[2][0]
        
        # Compare with current strategy
        strategy_changed = alternative_strategy != PRODUCTION_STRATEGY
        
        return {
            'recommended_strategy': alternative_strategy,
            'current_strategy': PRODUCTION_STRATEGY,
            'strategy_changed': strategy_changed,
            'model_rankings': ranked_models,
            'recommendation_rationale': 'Based on composite score (70% accuracy, 30% speed)'
        }
    
    def _assess_deployment_readiness(self, tier_validations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Assess overall deployment readiness."""
        
        # Check if at least Primary tier is ready
        primary_ready = tier_validations.get('Primary', {}).get('deployment_ready', False)
        
        # Count ready tiers
        ready_tiers = sum(1 for validation in tier_validations.values() 
                         if validation.get('deployment_ready', False))
        
        # Determine readiness level
        if ready_tiers >= 3:
            readiness_level = 'FULLY_READY'
            deployment_confidence = 'HIGH'
        elif ready_tiers >= 2 and primary_ready:
            readiness_level = 'READY_WITH_BACKUP'
            deployment_confidence = 'MEDIUM'
        elif primary_ready:
            readiness_level = 'MINIMAL_READY'
            deployment_confidence = 'LOW'
        else:
            readiness_level = 'NOT_READY'
            deployment_confidence = 'NONE'
        
        readiness_assessment = {
            'readiness_level': readiness_level,
            'deployment_confidence': deployment_confidence,
            'ready_tiers': ready_tiers,
            'primary_ready': primary_ready,
            'can_deploy': primary_ready,
            'recommended_action': self._get_readiness_action(readiness_level)
        }
        
        return readiness_assessment
    
    def _get_deployment_recommendation(self, strategy_status: str) -> str:
        """Get deployment recommendation based on strategy status."""
        
        recommendations = {
            'FULLY_VALIDATED': 'PROCEED_WITH_DEPLOYMENT',
            'PARTIALLY_VALIDATED': 'PROCEED_WITH_CAUTION',
            'MINIMALLY_VALIDATED': 'DELAY_DEPLOYMENT',
            'VALIDATION_FAILED': 'DO_NOT_DEPLOY'
        }
        
        return recommendations.get(strategy_status, 'REVIEW_REQUIRED')
    
    def _get_readiness_action(self, readiness_level: str) -> str:
        """Get recommended action based on readiness level."""
        
        actions = {
            'FULLY_READY': 'Deploy with full 3-tier strategy',
            'READY_WITH_BACKUP': 'Deploy with primary and secondary models',
            'MINIMAL_READY': 'Deploy primary model only, monitor closely',
            'NOT_READY': 'Do not deploy, retrain models'
        }
        
        return actions.get(readiness_level, 'Review deployment criteria')
    
    def validate_model_drift(self, current_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate model performance drift compared to Phase 7 baselines.
        
        Args:
            current_results (Dict): Current model evaluation results
            
        Returns:
            Dict[str, Any]: Drift validation results
        """
        drift_results = {}
        
        for model_name, baseline_accuracy in PHASE7_BASELINES.items():
            if model_name in current_results and current_results[model_name] is not None:
                current_accuracy = current_results[model_name].get('accuracy', 0)
                accuracy_drift = current_accuracy - baseline_accuracy
                drift_magnitude = abs(accuracy_drift)
                
                drift_results[model_name] = {
                    'baseline_accuracy': baseline_accuracy,
                    'current_accuracy': current_accuracy,
                    'accuracy_drift': accuracy_drift,
                    'drift_magnitude': drift_magnitude,
                    'drift_threshold': DRIFT_THRESHOLD,
                    'drift_detected': drift_magnitude > DRIFT_THRESHOLD,
                    'drift_direction': 'improvement' if accuracy_drift > 0 else 'degradation',
                    'drift_severity': self._assess_drift_severity(drift_magnitude)
                }
        
        return drift_results
    
    def _assess_drift_severity(self, drift_magnitude: float) -> str:
        """Assess the severity of model drift."""
        
        if drift_magnitude <= 0.01:
            return 'MINIMAL'
        elif drift_magnitude <= 0.03:
            return 'MODERATE'
        elif drift_magnitude <= 0.05:
            return 'SIGNIFICANT'
        else:
            return 'SEVERE'
