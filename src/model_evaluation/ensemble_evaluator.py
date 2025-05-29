"""
Ensemble Evaluator Module

Implements ensemble evaluation combining top 3 models for enhanced accuracy.
Provides voting classifiers and weighted ensemble strategies.

Key Features:
- Ensemble creation from top performing models
- Voting classifier implementation (hard and soft voting)
- Weighted ensemble based on individual model performance
- Ensemble performance evaluation
- Production ensemble recommendations
"""

import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Performance standard
PERFORMANCE_STANDARD = 97000  # >97K records/second


class EnsembleEvaluator:
    """
    Ensemble evaluator for Phase 8 implementation.
    
    Creates and evaluates ensemble models combining top performers
    for enhanced accuracy and robustness.
    """
    
    def __init__(self):
        """Initialize EnsembleEvaluator."""
        self.ensemble_results = {}
        self.ensemble_models = {}
        
    def create_ensemble(self, models: Dict[str, Any], model_rankings: List[Tuple[str, float]], 
                       top_n: int = 3) -> Dict[str, Any]:
        """
        Create ensemble from top N models.
        
        Args:
            models (Dict): Trained models
            model_rankings (List): Model rankings by performance
            top_n (int): Number of top models to include
            
        Returns:
            Dict[str, Any]: Ensemble creation results
        """
        start_time = time.time()
        
        # Select top N models
        top_models = model_rankings[:top_n]
        selected_models = []
        model_weights = []
        
        for model_name, score in top_models:
            if model_name in models and models[model_name] is not None:
                selected_models.append((model_name, models[model_name]))
                model_weights.append(score)
        
        if len(selected_models) < 2:
            logger.warning("Insufficient models for ensemble creation")
            return {'error': 'Insufficient models for ensemble'}
        
        # Normalize weights
        total_weight = sum(model_weights)
        normalized_weights = [w / total_weight for w in model_weights] if total_weight > 0 else None
        
        # Create ensemble configurations
        ensemble_configs = self._create_ensemble_configs(selected_models, normalized_weights)
        
        creation_time = time.time() - start_time
        
        ensemble_info = {
            'selected_models': [name for name, _ in selected_models],
            'model_scores': dict(top_models[:len(selected_models)]),
            'normalized_weights': normalized_weights,
            'ensemble_configs': ensemble_configs,
            'creation_time': creation_time
        }
        
        logger.info(f"Ensemble created with {len(selected_models)} models in {creation_time:.2f}s")
        return ensemble_info
    
    def _create_ensemble_configs(self, selected_models: List[Tuple[str, Any]], 
                                weights: Optional[List[float]]) -> Dict[str, Dict[str, Any]]:
        """Create different ensemble configurations."""
        
        configs = {}
        
        # Hard voting ensemble
        hard_voting_estimators = [(name, model) for name, model in selected_models]
        configs['hard_voting'] = {
            'type': 'VotingClassifier',
            'voting': 'hard',
            'estimators': hard_voting_estimators,
            'weights': None
        }
        
        # Soft voting ensemble (if all models support predict_proba)
        soft_voting_supported = all(
            hasattr(model, 'predict_proba') for _, model in selected_models
        )
        
        if soft_voting_supported:
            configs['soft_voting'] = {
                'type': 'VotingClassifier',
                'voting': 'soft',
                'estimators': hard_voting_estimators,
                'weights': None
            }
            
            # Weighted soft voting
            if weights:
                configs['weighted_soft_voting'] = {
                    'type': 'VotingClassifier',
                    'voting': 'soft',
                    'estimators': hard_voting_estimators,
                    'weights': weights
                }
        
        # Simple averaging ensemble (for models with predict_proba)
        if soft_voting_supported:
            configs['simple_averaging'] = {
                'type': 'SimpleAveraging',
                'models': selected_models,
                'weights': None
            }
            
            # Weighted averaging
            if weights:
                configs['weighted_averaging'] = {
                    'type': 'WeightedAveraging',
                    'models': selected_models,
                    'weights': weights
                }
        
        return configs
    
    def evaluate_ensemble(self, ensemble_config: Dict[str, Any], X_test: pd.DataFrame, 
                         y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate a specific ensemble configuration.
        
        Args:
            ensemble_config (Dict): Ensemble configuration
            X_test (DataFrame): Test features
            y_test (Series): Test labels
            
        Returns:
            Dict[str, Any]: Ensemble evaluation results
        """
        start_time = time.time()
        
        try:
            # Create ensemble based on configuration
            if ensemble_config['type'] == 'VotingClassifier':
                ensemble = self._create_voting_classifier(ensemble_config)
                
                # Fit ensemble (already trained individual models)
                # VotingClassifier doesn't need fitting if base models are fitted
                
                # Make predictions
                y_pred = ensemble.predict(X_test)
                
                # Get probabilities if available
                try:
                    y_pred_proba = ensemble.predict_proba(X_test)[:, 1]
                except:
                    y_pred_proba = None
                    
            elif ensemble_config['type'] in ['SimpleAveraging', 'WeightedAveraging']:
                y_pred, y_pred_proba = self._evaluate_averaging_ensemble(
                    ensemble_config, X_test
                )
            else:
                raise ValueError(f"Unknown ensemble type: {ensemble_config['type']}")
            
            # Calculate metrics
            metrics = self._calculate_ensemble_metrics(y_test, y_pred, y_pred_proba)
            
            # Performance tracking
            eval_time = time.time() - start_time
            records_per_second = len(X_test) / eval_time if eval_time > 0 else 0
            
            metrics['performance'] = {
                'evaluation_time': eval_time,
                'records_per_second': records_per_second,
                'meets_performance_standard': records_per_second >= PERFORMANCE_STANDARD,
                'test_samples': len(X_test)
            }
            
            logger.info(f"Ensemble evaluated: {metrics['accuracy']:.4f} accuracy, {records_per_second:,.0f} records/sec")
            return metrics
            
        except Exception as e:
            logger.error(f"Ensemble evaluation failed: {str(e)}")
            return {'error': str(e)}
    
    def _create_voting_classifier(self, config: Dict[str, Any]) -> VotingClassifier:
        """Create VotingClassifier from configuration."""
        
        return VotingClassifier(
            estimators=config['estimators'],
            voting=config['voting'],
            weights=config.get('weights')
        )
    
    def _evaluate_averaging_ensemble(self, config: Dict[str, Any], 
                                   X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate averaging ensemble."""
        
        models = config['models']
        weights = config.get('weights')
        
        # Collect predictions from all models
        predictions = []
        probabilities = []
        
        for model_name, model in models:
            try:
                pred = model.predict(X_test)
                predictions.append(pred)
                
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_test)[:, 1]
                    probabilities.append(proba)
                    
            except Exception as e:
                logger.warning(f"Error getting predictions from {model_name}: {str(e)}")
        
        if not predictions:
            raise ValueError("No valid predictions obtained from ensemble models")
        
        # Average predictions
        if weights and len(weights) == len(predictions):
            # Weighted averaging
            weighted_proba = np.average(probabilities, axis=0, weights=weights) if probabilities else None
            
            # For hard predictions, use weighted voting
            if weighted_proba is not None:
                y_pred = (weighted_proba >= 0.5).astype(int)
            else:
                # Fallback to simple majority voting
                y_pred = np.round(np.average(predictions, axis=0, weights=weights)).astype(int)
        else:
            # Simple averaging
            if probabilities:
                weighted_proba = np.mean(probabilities, axis=0)
                y_pred = (weighted_proba >= 0.5).astype(int)
            else:
                weighted_proba = None
                y_pred = np.round(np.mean(predictions, axis=0)).astype(int)
        
        return y_pred, weighted_proba
    
    def _calculate_ensemble_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                                  y_pred_proba: Optional[np.ndarray]) -> Dict[str, Any]:
        """Calculate comprehensive metrics for ensemble."""
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Add AUC if probabilities available
        if y_pred_proba is not None:
            try:
                metrics['auc_score'] = roc_auc_score(y_true, y_pred_proba)
            except:
                metrics['auc_score'] = None
        else:
            metrics['auc_score'] = None
        
        return metrics
    
    def evaluate_all_ensembles(self, models: Dict[str, Any], model_rankings: List[Tuple[str, float]], 
                             X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate all ensemble configurations.
        
        Args:
            models (Dict): Trained models
            model_rankings (List): Model rankings
            X_test (DataFrame): Test features
            y_test (Series): Test labels
            
        Returns:
            Dict[str, Any]: Complete ensemble evaluation results
        """
        start_time = time.time()
        
        # Create ensemble
        ensemble_info = self.create_ensemble(models, model_rankings)
        
        if 'error' in ensemble_info:
            return ensemble_info
        
        # Evaluate each ensemble configuration
        ensemble_results = {}
        
        for config_name, config in ensemble_info['ensemble_configs'].items():
            logger.info(f"Evaluating ensemble: {config_name}")
            
            try:
                result = self.evaluate_ensemble(config, X_test, y_test)
                ensemble_results[config_name] = result
                
            except Exception as e:
                logger.error(f"Failed to evaluate {config_name}: {str(e)}")
                ensemble_results[config_name] = {'error': str(e)}
        
        # Find best ensemble
        best_ensemble = self._find_best_ensemble(ensemble_results)
        
        # Compare with individual models
        individual_comparison = self._compare_with_individuals(
            ensemble_results, model_rankings
        )
        
        # Performance summary
        total_time = time.time() - start_time
        
        complete_results = {
            'ensemble_info': ensemble_info,
            'ensemble_results': ensemble_results,
            'best_ensemble': best_ensemble,
            'individual_comparison': individual_comparison,
            'evaluation_summary': {
                'total_ensembles_evaluated': len(ensemble_results),
                'successful_evaluations': len([r for r in ensemble_results.values() if 'error' not in r]),
                'total_evaluation_time': total_time
            }
        }
        
        self.ensemble_results = complete_results
        logger.info(f"Ensemble evaluation completed in {total_time:.2f}s")
        
        return complete_results
    
    def _find_best_ensemble(self, ensemble_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Find the best performing ensemble."""
        
        best_ensemble = None
        best_score = -1
        best_config = None
        
        for config_name, results in ensemble_results.items():
            if 'error' in results:
                continue
                
            # Use F1 score as primary metric, accuracy as secondary
            f1_score = results.get('f1_score', 0)
            accuracy = results.get('accuracy', 0)
            
            # Composite score (70% F1, 30% accuracy)
            composite_score = 0.7 * f1_score + 0.3 * accuracy
            
            if composite_score > best_score:
                best_score = composite_score
                best_ensemble = results
                best_config = config_name
        
        if best_ensemble:
            return {
                'config_name': best_config,
                'metrics': best_ensemble,
                'composite_score': best_score
            }
        else:
            return {'error': 'No successful ensemble evaluations'}
    
    def _compare_with_individuals(self, ensemble_results: Dict[str, Dict[str, Any]], 
                                model_rankings: List[Tuple[str, float]]) -> Dict[str, Any]:
        """Compare ensemble performance with individual models."""
        
        # Get best individual model performance
        if model_rankings:
            best_individual_name, best_individual_score = model_rankings[0]
        else:
            best_individual_name, best_individual_score = "Unknown", 0
        
        # Get best ensemble performance
        best_ensemble_info = self._find_best_ensemble(ensemble_results)
        
        if 'error' in best_ensemble_info:
            return {'error': 'No valid ensemble for comparison'}
        
        best_ensemble_score = best_ensemble_info['composite_score']
        
        # Calculate improvement
        improvement = best_ensemble_score - best_individual_score
        improvement_percentage = (improvement / best_individual_score * 100) if best_individual_score > 0 else 0
        
        comparison = {
            'best_individual': {
                'model': best_individual_name,
                'score': best_individual_score
            },
            'best_ensemble': {
                'config': best_ensemble_info['config_name'],
                'score': best_ensemble_score,
                'metrics': best_ensemble_info['metrics']
            },
            'improvement': {
                'absolute': improvement,
                'percentage': improvement_percentage,
                'ensemble_better': improvement > 0
            },
            'recommendation': self._generate_ensemble_recommendation(improvement_percentage)
        }
        
        return comparison
    
    def _generate_ensemble_recommendation(self, improvement_percentage: float) -> str:
        """Generate ensemble deployment recommendation."""
        
        if improvement_percentage >= 5:
            return "STRONGLY_RECOMMEND_ENSEMBLE"
        elif improvement_percentage >= 2:
            return "RECOMMEND_ENSEMBLE"
        elif improvement_percentage >= 0:
            return "CONSIDER_ENSEMBLE"
        else:
            return "USE_INDIVIDUAL_MODEL"
    
    def get_production_ensemble_recommendation(self) -> Dict[str, Any]:
        """Get production deployment recommendation for ensemble."""
        
        if not self.ensemble_results:
            return {'error': 'No ensemble evaluation results available'}
        
        best_ensemble = self.ensemble_results.get('best_ensemble', {})
        individual_comparison = self.ensemble_results.get('individual_comparison', {})
        
        if 'error' in best_ensemble:
            return {'recommendation': 'USE_INDIVIDUAL_MODEL', 'reason': 'No successful ensemble'}
        
        recommendation = {
            'deployment_strategy': individual_comparison.get('recommendation', 'USE_INDIVIDUAL_MODEL'),
            'best_ensemble_config': best_ensemble.get('config_name'),
            'expected_improvement': individual_comparison.get('improvement', {}),
            'production_considerations': [
                'Ensemble increases computational complexity',
                'Requires all base models to be deployed',
                'May have higher latency than individual models',
                'Provides better robustness and accuracy'
            ]
        }
        
        return recommendation
