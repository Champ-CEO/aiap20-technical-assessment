"""
Feature Importance Analyzer Module

Implements feature importance analysis with Phase 7 findings validation.
Analyzes feature importance across models and validates against expected top features.

Key Features:
- Feature importance extraction from all models
- Phase 7 top features validation (Client ID, Previous Contact Days, contact_effectiveness_score)
- Cross-model feature importance comparison
- Business feature impact analysis
- Feature ranking and selection recommendations
"""

import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Phase 7 expected top features
PHASE7_TOP_FEATURES = [
    "Client ID",
    "Previous Contact Days",
    "contact_effectiveness_score"
]

# Business features from Phase 5
BUSINESS_FEATURES = [
    "age_bin",
    "customer_value_segment", 
    "contact_effectiveness_score",
    "campaign_intensity",
    "recent_contact_flag",
    "education_job_segment",
    "financial_risk_score"
]


class FeatureImportanceAnalyzer:
    """
    Feature importance analyzer for Phase 8 implementation.
    
    Analyzes feature importance across models and validates against
    Phase 7 findings and business feature expectations.
    """
    
    def __init__(self):
        """Initialize FeatureImportanceAnalyzer."""
        self.analysis_results = {}
        self.feature_rankings = {}
        
    def analyze_feature_importance(self, models: Dict[str, Any], 
                                 feature_names: List[str]) -> Dict[str, Any]:
        """
        Analyze feature importance across all models.
        
        Args:
            models (Dict): Trained models
            feature_names (List): List of feature names
            
        Returns:
            Dict[str, Any]: Comprehensive feature importance analysis
        """
        start_time = time.time()
        
        # Extract feature importance from each model
        model_importances = {}
        
        for model_name, model in models.items():
            if model is None:
                continue
                
            importance_data = self._extract_model_importance(model, feature_names)
            if importance_data:
                model_importances[model_name] = importance_data
        
        # Aggregate feature importance across models
        aggregated_importance = self._aggregate_importance(model_importances, feature_names)
        
        # Validate against Phase 7 findings
        phase7_validation = self._validate_phase7_features(aggregated_importance)
        
        # Analyze business features
        business_analysis = self._analyze_business_features(aggregated_importance)
        
        # Generate feature rankings
        feature_rankings = self._generate_feature_rankings(model_importances, aggregated_importance)
        
        # Compile results
        analysis_results = {
            'model_importances': model_importances,
            'aggregated_importance': aggregated_importance,
            'phase7_validation': phase7_validation,
            'business_analysis': business_analysis,
            'feature_rankings': feature_rankings,
            'top_features_summary': self._generate_top_features_summary(aggregated_importance),
            'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Performance tracking
        analysis_time = time.time() - start_time
        analysis_results['performance'] = {
            'analysis_time': analysis_time,
            'models_analyzed': len(model_importances),
            'features_analyzed': len(feature_names)
        }
        
        self.analysis_results = analysis_results
        logger.info(f"Feature importance analysis completed in {analysis_time:.2f}s")
        
        return analysis_results
    
    def _extract_model_importance(self, model: Any, feature_names: List[str]) -> Optional[Dict[str, float]]:
        """Extract feature importance from a single model."""
        
        try:
            # Try different methods to extract feature importance
            if hasattr(model, 'feature_importances_'):
                # Tree-based models (RandomForest, GradientBoosting)
                importances = model.feature_importances_
                return dict(zip(feature_names, importances))
                
            elif hasattr(model, 'coef_'):
                # Linear models (LogisticRegression, SVM)
                coef = model.coef_
                if len(coef.shape) > 1:
                    # Multi-class case, take absolute mean
                    importances = np.mean(np.abs(coef), axis=0)
                else:
                    importances = np.abs(coef)
                return dict(zip(feature_names, importances))
                
            elif hasattr(model, 'feature_log_prob_'):
                # Naive Bayes
                # Use variance of log probabilities as importance proxy
                log_probs = model.feature_log_prob_
                importances = np.var(log_probs, axis=0)
                return dict(zip(feature_names, importances))
                
            else:
                logger.warning(f"Cannot extract feature importance from model type: {type(model)}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting feature importance: {str(e)}")
            return None
    
    def _aggregate_importance(self, model_importances: Dict[str, Dict[str, float]], 
                            feature_names: List[str]) -> Dict[str, float]:
        """Aggregate feature importance across models."""
        
        if not model_importances:
            return {}
        
        # Initialize aggregated importance
        aggregated = {feature: 0.0 for feature in feature_names}
        
        # Calculate mean importance across models
        for feature in feature_names:
            importances = []
            for model_name, importance_dict in model_importances.items():
                if feature in importance_dict:
                    importances.append(importance_dict[feature])
            
            if importances:
                # Normalize importances within each model first, then average
                normalized_importances = []
                for model_name, importance_dict in model_importances.items():
                    if feature in importance_dict:
                        total_importance = sum(importance_dict.values())
                        if total_importance > 0:
                            normalized_importance = importance_dict[feature] / total_importance
                            normalized_importances.append(normalized_importance)
                
                if normalized_importances:
                    aggregated[feature] = np.mean(normalized_importances)
        
        return aggregated
    
    def _validate_phase7_features(self, aggregated_importance: Dict[str, float]) -> Dict[str, Any]:
        """Validate against Phase 7 expected top features."""
        
        # Sort features by importance
        sorted_features = sorted(aggregated_importance.items(), key=lambda x: x[1], reverse=True)
        top_10_features = [feature for feature, _ in sorted_features[:10]]
        
        validation_results = {}
        
        for expected_feature in PHASE7_TOP_FEATURES:
            # Check if feature exists in current analysis
            if expected_feature in aggregated_importance:
                importance = aggregated_importance[expected_feature]
                rank = top_10_features.index(expected_feature) + 1 if expected_feature in top_10_features else None
                
                validation_results[expected_feature] = {
                    'found': True,
                    'importance': importance,
                    'rank_in_top_10': rank,
                    'in_top_10': rank is not None,
                    'validation_status': 'VALIDATED' if rank and rank <= 10 else 'NOT_IN_TOP_10'
                }
            else:
                validation_results[expected_feature] = {
                    'found': False,
                    'importance': 0.0,
                    'rank_in_top_10': None,
                    'in_top_10': False,
                    'validation_status': 'NOT_FOUND'
                }
        
        # Overall validation summary
        validated_features = sum(1 for result in validation_results.values() 
                               if result['validation_status'] == 'VALIDATED')
        total_expected = len(PHASE7_TOP_FEATURES)
        
        validation_summary = {
            'individual_validations': validation_results,
            'validated_features': validated_features,
            'total_expected': total_expected,
            'validation_rate': validated_features / total_expected,
            'overall_status': 'PASSED' if validated_features >= total_expected * 0.7 else 'FAILED'
        }
        
        return validation_summary
    
    def _analyze_business_features(self, aggregated_importance: Dict[str, float]) -> Dict[str, Any]:
        """Analyze business features impact."""
        
        business_feature_analysis = {}
        
        # Analyze each business feature
        for business_feature in BUSINESS_FEATURES:
            if business_feature in aggregated_importance:
                importance = aggregated_importance[business_feature]
                
                # Determine impact level
                if importance >= 0.05:  # 5% or more
                    impact_level = 'HIGH'
                elif importance >= 0.02:  # 2-5%
                    impact_level = 'MEDIUM'
                elif importance >= 0.01:  # 1-2%
                    impact_level = 'LOW'
                else:
                    impact_level = 'MINIMAL'
                
                business_feature_analysis[business_feature] = {
                    'importance': importance,
                    'impact_level': impact_level,
                    'found': True
                }
            else:
                business_feature_analysis[business_feature] = {
                    'importance': 0.0,
                    'impact_level': 'NOT_FOUND',
                    'found': False
                }
        
        # Summary statistics
        found_features = [f for f, data in business_feature_analysis.items() if data['found']]
        high_impact_features = [f for f, data in business_feature_analysis.items() 
                              if data['impact_level'] == 'HIGH']
        
        business_summary = {
            'feature_analysis': business_feature_analysis,
            'found_features': found_features,
            'high_impact_features': high_impact_features,
            'business_feature_coverage': len(found_features) / len(BUSINESS_FEATURES),
            'high_impact_rate': len(high_impact_features) / len(found_features) if found_features else 0
        }
        
        return business_summary
    
    def _generate_feature_rankings(self, model_importances: Dict[str, Dict[str, float]], 
                                 aggregated_importance: Dict[str, float]) -> Dict[str, Any]:
        """Generate feature rankings across different criteria."""
        
        rankings = {}
        
        # Overall ranking (aggregated importance)
        overall_ranking = sorted(aggregated_importance.items(), key=lambda x: x[1], reverse=True)
        rankings['overall'] = overall_ranking
        
        # Model-specific rankings
        for model_name, importance_dict in model_importances.items():
            model_ranking = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            rankings[f'{model_name}_ranking'] = model_ranking
        
        # Business features ranking
        business_features_importance = {
            feature: aggregated_importance.get(feature, 0) 
            for feature in BUSINESS_FEATURES 
            if feature in aggregated_importance
        }
        business_ranking = sorted(business_features_importance.items(), key=lambda x: x[1], reverse=True)
        rankings['business_features'] = business_ranking
        
        # Phase 7 features ranking
        phase7_features_importance = {
            feature: aggregated_importance.get(feature, 0) 
            for feature in PHASE7_TOP_FEATURES 
            if feature in aggregated_importance
        }
        phase7_ranking = sorted(phase7_features_importance.items(), key=lambda x: x[1], reverse=True)
        rankings['phase7_features'] = phase7_ranking
        
        return rankings
    
    def _generate_top_features_summary(self, aggregated_importance: Dict[str, float]) -> Dict[str, Any]:
        """Generate summary of top features."""
        
        # Sort by importance
        sorted_features = sorted(aggregated_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Top features at different levels
        top_5 = sorted_features[:5]
        top_10 = sorted_features[:10]
        top_20 = sorted_features[:20]
        
        # Calculate cumulative importance
        total_importance = sum(aggregated_importance.values())
        
        top_5_cumulative = sum(importance for _, importance in top_5) / total_importance if total_importance > 0 else 0
        top_10_cumulative = sum(importance for _, importance in top_10) / total_importance if total_importance > 0 else 0
        top_20_cumulative = sum(importance for _, importance in top_20) / total_importance if total_importance > 0 else 0
        
        summary = {
            'top_5_features': top_5,
            'top_10_features': top_10,
            'top_20_features': top_20,
            'cumulative_importance': {
                'top_5': top_5_cumulative,
                'top_10': top_10_cumulative,
                'top_20': top_20_cumulative
            },
            'feature_concentration': {
                'top_5_dominance': top_5_cumulative,
                'feature_diversity': 1 - top_5_cumulative,  # How distributed importance is
                'effective_features': len([f for f, imp in sorted_features if imp >= 0.01])  # Features with >1% importance
            }
        }
        
        return summary
    
    def get_feature_recommendations(self) -> Dict[str, Any]:
        """Generate feature selection recommendations."""
        
        if not self.analysis_results:
            return {}
        
        top_features_summary = self.analysis_results.get('top_features_summary', {})
        phase7_validation = self.analysis_results.get('phase7_validation', {})
        business_analysis = self.analysis_results.get('business_analysis', {})
        
        recommendations = {
            'feature_selection': {
                'recommended_features': [],
                'features_to_investigate': [],
                'features_to_consider_removing': []
            },
            'model_optimization': {
                'high_impact_features': [],
                'feature_engineering_opportunities': []
            },
            'business_alignment': {
                'validated_business_features': [],
                'missing_business_features': []
            }
        }
        
        # Recommended features (top performers + validated Phase 7 features)
        if 'top_10_features' in top_features_summary:
            top_10 = [feature for feature, _ in top_features_summary['top_10_features']]
            recommendations['feature_selection']['recommended_features'] = top_10
        
        # High impact features
        if 'high_impact_features' in business_analysis:
            recommendations['model_optimization']['high_impact_features'] = business_analysis['high_impact_features']
        
        # Validated business features
        if 'individual_validations' in phase7_validation:
            validated = [feature for feature, result in phase7_validation['individual_validations'].items()
                        if result['validation_status'] == 'VALIDATED']
            recommendations['business_alignment']['validated_business_features'] = validated
        
        return recommendations
