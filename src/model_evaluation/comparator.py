"""
Model Comparator Module

Implements model comparison and ranking logic with Phase 7 performance integration.
Provides comprehensive model comparison based on multiple criteria.

Key Features:
- Model ranking based on accuracy and business metrics
- Phase 7 performance integration (89.8%, 89.5%, 84.6%, 78.8%, 71.4%)
- Multi-criteria comparison (accuracy, speed, interpretability)
- Production deployment recommendations
- Customer segment-aware comparison
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

# Phase 7 actual performance results
PHASE7_PERFORMANCE_RESULTS = {
    "GradientBoosting": 0.8978,  # 89.8% - Best performer
    "NaiveBayes": 0.8954,        # 89.5% - Fast performer  
    "RandomForest": 0.8460,      # 84.6% - Balanced performer
    "SVM": 0.7879,               # 78.8% - Support vector
    "LogisticRegression": 0.7147, # 71.4% - Interpretable baseline
}

# Performance standards
ACCURACY_BASELINE = 0.898  # 89.8% from GradientBoosting
SPEED_STANDARD = 255000    # 255K records/second from NaiveBayes

# Production deployment strategy
PRODUCTION_STRATEGY = {
    "Primary": "GradientBoosting",    # 89.8% accuracy
    "Secondary": "RandomForest",      # 84.6% balanced
    "Tertiary": "NaiveBayes",        # 255K records/sec speed
}


class ModelComparator:
    """
    Comprehensive model comparator for Phase 8 implementation.
    
    Handles model ranking, comparison analysis, and production deployment
    recommendations based on multiple evaluation criteria.
    """
    
    def __init__(self):
        """Initialize ModelComparator."""
        self.comparison_results = {}
        self.ranking_results = {}
        self.performance_metrics = {}
        
    def compare_models(self, evaluation_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare all models based on evaluation results.
        
        Args:
            evaluation_results (Dict): Results from ModelEvaluator
            
        Returns:
            Dict[str, Any]: Comprehensive comparison analysis
        """
        start_time = time.time()
        
        # Extract performance metrics for comparison
        model_metrics = {}
        
        for model_name, results in evaluation_results.items():
            if results is not None:
                model_metrics[model_name] = {
                    'accuracy': results.get('accuracy', 0),
                    'precision': results.get('precision', 0),
                    'recall': results.get('recall', 0),
                    'f1_score': results.get('f1_score', 0),
                    'auc_score': results.get('auc_score', 0),
                    'records_per_second': results.get('performance', {}).get('records_per_second', 0)
                }
            else:
                model_metrics[model_name] = {
                    'accuracy': 0, 'precision': 0, 'recall': 0, 
                    'f1_score': 0, 'auc_score': 0, 'records_per_second': 0
                }
        
        # Perform ranking analysis
        rankings = self._rank_models(model_metrics)
        
        # Generate comparison summary
        comparison_summary = self._generate_comparison_summary(model_metrics, rankings)
        
        # Validate against Phase 7 results
        phase7_validation = self._validate_against_phase7(model_metrics)
        
        # Production deployment analysis
        deployment_analysis = self._analyze_deployment_strategy(model_metrics, rankings)
        
        # Compile comprehensive comparison results
        comparison_results = {
            'model_metrics': model_metrics,
            'rankings': rankings,
            'comparison_summary': comparison_summary,
            'phase7_validation': phase7_validation,
            'deployment_analysis': deployment_analysis,
            'performance_analysis': self._analyze_performance_standards(model_metrics)
        }
        
        # Performance tracking
        comparison_time = time.time() - start_time
        self.performance_metrics['comparison'] = {
            'comparison_time': comparison_time,
            'models_compared': len(model_metrics),
            'rankings_generated': len(rankings)
        }
        
        self.comparison_results = comparison_results
        logger.info(f"Model comparison completed: {len(model_metrics)} models in {comparison_time:.2f}s")
        
        return comparison_results
    
    def _rank_models(self, model_metrics: Dict[str, Dict[str, float]]) -> Dict[str, List[Tuple[str, float]]]:
        """
        Rank models based on different criteria.
        
        Args:
            model_metrics (Dict): Model performance metrics
            
        Returns:
            Dict[str, List[Tuple]]: Rankings by different criteria
        """
        rankings = {}
        
        # Rank by accuracy
        accuracy_ranking = sorted(
            [(name, metrics['accuracy']) for name, metrics in model_metrics.items()],
            key=lambda x: x[1], reverse=True
        )
        rankings['accuracy'] = accuracy_ranking
        
        # Rank by F1 score
        f1_ranking = sorted(
            [(name, metrics['f1_score']) for name, metrics in model_metrics.items()],
            key=lambda x: x[1], reverse=True
        )
        rankings['f1_score'] = f1_ranking
        
        # Rank by speed (records per second)
        speed_ranking = sorted(
            [(name, metrics['records_per_second']) for name, metrics in model_metrics.items()],
            key=lambda x: x[1], reverse=True
        )
        rankings['speed'] = speed_ranking
        
        # Rank by AUC (handle None values)
        auc_ranking = sorted(
            [(name, metrics['auc_score'] or 0) for name, metrics in model_metrics.items()],
            key=lambda x: x[1], reverse=True
        )
        rankings['auc'] = auc_ranking
        
        # Overall ranking (weighted combination)
        overall_scores = {}
        for name, metrics in model_metrics.items():
            # Weighted score: 40% accuracy, 30% F1, 20% AUC, 10% speed (normalized)
            max_speed = max(m['records_per_second'] for m in model_metrics.values()) or 1
            normalized_speed = metrics['records_per_second'] / max_speed
            
            overall_score = (
                0.4 * metrics['accuracy'] +
                0.3 * metrics['f1_score'] +
                0.2 * (metrics['auc_score'] or 0) +
                0.1 * normalized_speed
            )
            overall_scores[name] = overall_score
        
        overall_ranking = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        rankings['overall'] = overall_ranking
        
        return rankings
    
    def _generate_comparison_summary(self, model_metrics: Dict[str, Dict[str, float]], 
                                   rankings: Dict[str, List[Tuple[str, float]]]) -> Dict[str, Any]:
        """Generate summary of model comparison."""
        
        # Best performers by category
        best_performers = {}
        for category, ranking in rankings.items():
            if ranking:
                best_performers[category] = {
                    'model': ranking[0][0],
                    'score': ranking[0][1]
                }
        
        # Performance statistics
        accuracy_scores = [metrics['accuracy'] for metrics in model_metrics.values()]
        speed_scores = [metrics['records_per_second'] for metrics in model_metrics.values()]
        
        summary = {
            'best_performers': best_performers,
            'performance_statistics': {
                'accuracy': {
                    'mean': np.mean(accuracy_scores),
                    'std': np.std(accuracy_scores),
                    'min': np.min(accuracy_scores),
                    'max': np.max(accuracy_scores)
                },
                'speed': {
                    'mean': np.mean(speed_scores),
                    'std': np.std(speed_scores),
                    'min': np.min(speed_scores),
                    'max': np.max(speed_scores)
                }
            },
            'models_meeting_standards': {
                'accuracy_baseline': sum(1 for acc in accuracy_scores if acc >= ACCURACY_BASELINE),
                'speed_standard': sum(1 for speed in speed_scores if speed >= SPEED_STANDARD)
            }
        }
        
        return summary
    
    def _validate_against_phase7(self, model_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Validate current results against Phase 7 performance."""
        
        validation_results = {}
        
        for model_name, phase7_accuracy in PHASE7_PERFORMANCE_RESULTS.items():
            if model_name in model_metrics:
                current_accuracy = model_metrics[model_name]['accuracy']
                accuracy_diff = current_accuracy - phase7_accuracy
                
                validation_results[model_name] = {
                    'phase7_accuracy': phase7_accuracy,
                    'current_accuracy': current_accuracy,
                    'accuracy_difference': accuracy_diff,
                    'performance_maintained': abs(accuracy_diff) <= 0.01,  # Within 1%
                    'performance_improved': accuracy_diff > 0.01
                }
        
        return validation_results
    
    def _analyze_deployment_strategy(self, model_metrics: Dict[str, Dict[str, float]], 
                                   rankings: Dict[str, List[Tuple[str, float]]]) -> Dict[str, Any]:
        """Analyze production deployment strategy."""
        
        # Validate current production strategy
        strategy_validation = {}
        
        for tier, expected_model in PRODUCTION_STRATEGY.items():
            if expected_model in model_metrics:
                metrics = model_metrics[expected_model]
                strategy_validation[tier] = {
                    'model': expected_model,
                    'accuracy': metrics['accuracy'],
                    'speed': metrics['records_per_second'],
                    'meets_accuracy_baseline': metrics['accuracy'] >= ACCURACY_BASELINE,
                    'meets_speed_standard': metrics['records_per_second'] >= SPEED_STANDARD
                }
        
        # Recommend alternative strategy based on current results
        if rankings.get('overall'):
            recommended_strategy = {
                'Primary': rankings['overall'][0][0] if len(rankings['overall']) > 0 else None,
                'Secondary': rankings['overall'][1][0] if len(rankings['overall']) > 1 else None,
                'Tertiary': rankings['overall'][2][0] if len(rankings['overall']) > 2 else None
            }
        else:
            recommended_strategy = PRODUCTION_STRATEGY.copy()
        
        return {
            'current_strategy_validation': strategy_validation,
            'recommended_strategy': recommended_strategy,
            'strategy_alignment': recommended_strategy == PRODUCTION_STRATEGY
        }
    
    def _analyze_performance_standards(self, model_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Analyze performance against established standards."""
        
        performance_analysis = {}
        
        for model_name, metrics in model_metrics.items():
            performance_analysis[model_name] = {
                'meets_accuracy_baseline': metrics['accuracy'] >= ACCURACY_BASELINE,
                'meets_speed_standard': metrics['records_per_second'] >= SPEED_STANDARD,
                'accuracy_ratio': metrics['accuracy'] / ACCURACY_BASELINE,
                'speed_ratio': metrics['records_per_second'] / SPEED_STANDARD,
                'overall_compliance': (
                    metrics['accuracy'] >= ACCURACY_BASELINE and 
                    metrics['records_per_second'] >= SPEED_STANDARD
                )
            }
        
        return performance_analysis
    
    def get_top_models(self, n: int = 3, criterion: str = 'overall') -> List[Tuple[str, float]]:
        """
        Get top N models based on specified criterion.
        
        Args:
            n (int): Number of top models to return
            criterion (str): Ranking criterion ('overall', 'accuracy', 'speed', etc.)
            
        Returns:
            List[Tuple[str, float]]: Top N models with scores
        """
        if not self.ranking_results or criterion not in self.ranking_results:
            return []
        
        return self.ranking_results[criterion][:n]
