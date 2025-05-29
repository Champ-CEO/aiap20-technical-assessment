"""
Classifier 3: Gradient Boosting/XGBoost

Phase 6 recommended - advanced gradient boosting with categorical support for complex patterns.
Sequential ensemble method that builds models iteratively to correct previous errors.

Key Features:
- Advanced gradient boosting for complex patterns
- Excellent handling of categorical features
- Built-in regularization to prevent overfitting
- Feature importance through gain/split analysis
- High performance on structured data
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as SklearnGradientBoostingClassifier
from .base_classifier import BaseClassifier


class GradientBoostingClassifier(BaseClassifier):
    """
    Gradient Boosting classifier for term deposit subscription prediction.
    
    Optimized for complex pattern recognition with sequential ensemble learning.
    Provides advanced feature importance analysis and handles categorical features effectively.
    """
    
    def __init__(self, 
                 n_estimators=100, 
                 learning_rate=0.1, 
                 max_depth=3,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 subsample=1.0,
                 max_features=None,
                 random_state=42):
        """
        Initialize Gradient Boosting classifier.
        
        Args:
            n_estimators (int): Number of boosting stages
            learning_rate (float): Learning rate shrinks contribution of each tree
            max_depth (int): Maximum depth of individual trees
            min_samples_split (int): Minimum samples required to split node
            min_samples_leaf (int): Minimum samples required at leaf node
            subsample (float): Fraction of samples for fitting individual trees
            max_features (str/int): Number of features for best split
            random_state (int): Random seed for reproducibility
        """
        super().__init__("Gradient Boosting")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.random_state = random_state
        
    def _create_model(self):
        """
        Create Gradient Boosting model instance.
        
        Returns:
            GradientBoostingClassifier: Configured sklearn model
        """
        return SklearnGradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            subsample=self.subsample,
            max_features=self.max_features,
            random_state=self.random_state,
            validation_fraction=0.1,  # For early stopping
            n_iter_no_change=10,  # Early stopping patience
            tol=1e-4  # Tolerance for early stopping
        )
    
    def get_training_progress(self):
        """
        Get training progress and convergence information.
        
        Returns:
            dict: Training progress analysis
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting training progress")
        
        # Get training scores
        train_scores = self.model.train_score_
        
        # Calculate improvement metrics
        initial_score = train_scores[0]
        final_score = train_scores[-1]
        improvement = final_score - initial_score
        
        # Find convergence point (where improvement becomes minimal)
        convergence_point = len(train_scores)
        for i in range(10, len(train_scores)):
            recent_improvement = train_scores[i] - train_scores[i-10]
            if recent_improvement < 0.001:  # Minimal improvement threshold
                convergence_point = i
                break
        
        return {
            'total_iterations': len(train_scores),
            'convergence_iteration': convergence_point,
            'initial_score': initial_score,
            'final_score': final_score,
            'total_improvement': improvement,
            'training_scores': train_scores,
            'early_stopping_used': convergence_point < len(train_scores)
        }
    
    def get_feature_importance_by_stage(self, n_stages=None):
        """
        Get feature importance evolution across boosting stages.
        
        Args:
            n_stages (int): Number of stages to analyze (None for all)
            
        Returns:
            dict: Feature importance evolution
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before analyzing feature importance")
        
        if n_stages is None:
            n_stages = min(10, self.model.n_estimators)
        
        # Get feature importance at different stages
        stage_importances = {}
        
        # Sample stages evenly across training
        stage_indices = np.linspace(0, self.model.n_estimators - 1, n_stages, dtype=int)
        
        for stage_idx in stage_indices:
            # Create a temporary model with limited estimators
            temp_model = SklearnGradientBoostingClassifier(
                n_estimators=stage_idx + 1,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
            
            # Copy the trained estimators
            temp_model.estimators_ = self.model.estimators_[:stage_idx + 1]
            temp_model.train_score_ = self.model.train_score_[:stage_idx + 1]
            temp_model.n_estimators = stage_idx + 1
            
            # Get feature importance at this stage
            if hasattr(temp_model, 'feature_importances_'):
                stage_importances[f'stage_{stage_idx + 1}'] = dict(
                    zip(self.feature_names, temp_model.feature_importances_)
                )
        
        return stage_importances
    
    def analyze_boosting_contribution(self):
        """
        Analyze how each boosting stage contributes to the final prediction.
        
        Returns:
            dict: Boosting contribution analysis
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before analyzing boosting")
        
        # Get staged predictions (cumulative predictions at each stage)
        # Note: This would require test data, so we'll provide the framework
        
        analysis = {
            'total_stages': self.model.n_estimators,
            'learning_rate': self.learning_rate,
            'effective_stages': len(self.model.estimators_),
            'feature_importance_final': self.get_feature_importance(),
            'training_progress': self.get_training_progress(),
            'regularization_effect': {
                'max_depth': self.max_depth,
                'subsample': self.subsample,
                'min_samples_split': self.min_samples_split
            }
        }
        
        return analysis
    
    def get_prediction_explanation(self, X, n_top_features=5):
        """
        Explain predictions using feature contributions.
        
        Args:
            X (pd.DataFrame): Features for explanation
            n_top_features (int): Number of top contributing features to show
            
        Returns:
            dict: Prediction explanations
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before explaining predictions")
        
        # Get predictions and probabilities
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        # Get feature importance
        feature_importance = self.get_feature_importance()
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:n_top_features]
        
        # Encode features for analysis
        X_encoded = self._encode_categorical_features(X, fit=False)
        
        explanations = []
        for i in range(len(X)):
            explanation = {
                'prediction': predictions[i],
                'probability': probabilities[i],
                'top_contributing_features': []
            }
            
            # Analyze top features for this instance
            for feature, importance in top_features:
                if feature in X.columns:
                    feature_value = X.iloc[i][feature]
                    explanation['top_contributing_features'].append({
                        'feature': feature,
                        'value': feature_value,
                        'importance': importance,
                        'contribution': importance * X_encoded.iloc[i][X_encoded.columns.get_loc(feature)]
                    })
            
            explanations.append(explanation)
        
        return explanations
    
    def get_business_insights(self):
        """
        Generate business insights from the Gradient Boosting model.
        
        Returns:
            dict: Business insights and recommendations
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before generating insights")
        
        # Get comprehensive analysis
        feature_importance = self.get_feature_importance()
        training_progress = self.get_training_progress()
        boosting_analysis = self.analyze_boosting_contribution()
        
        # Identify pattern complexity
        complexity_indicators = {
            'convergence_speed': training_progress['convergence_iteration'] / training_progress['total_iterations'],
            'improvement_magnitude': training_progress['total_improvement'],
            'effective_depth': self.max_depth,
            'regularization_strength': 1 - self.subsample
        }
        
        insights = {
            'model_type': 'Gradient Boosting',
            'interpretability': 'Medium - sequential feature importance and staged predictions',
            'pattern_complexity': self._assess_pattern_complexity(complexity_indicators),
            'top_features': dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]),
            'training_efficiency': {
                'convergence_iteration': training_progress['convergence_iteration'],
                'total_iterations': training_progress['total_iterations'],
                'early_stopping': training_progress['early_stopping_used']
            },
            'business_recommendations': self._generate_gb_recommendations(feature_importance, complexity_indicators)
        }
        
        return insights
    
    def _assess_pattern_complexity(self, indicators):
        """
        Assess the complexity of patterns learned by the model.
        
        Args:
            indicators (dict): Complexity indicators
            
        Returns:
            dict: Pattern complexity assessment
        """
        complexity_score = 0
        
        # Fast convergence suggests simpler patterns
        if indicators['convergence_speed'] < 0.5:
            complexity_score += 1
        
        # Large improvement suggests complex patterns
        if indicators['improvement_magnitude'] > 0.1:
            complexity_score += 1
        
        # Deep trees suggest complex interactions
        if indicators['effective_depth'] > 5:
            complexity_score += 1
        
        complexity_levels = ['Simple', 'Moderate', 'Complex', 'Very Complex']
        complexity_level = complexity_levels[min(complexity_score, 3)]
        
        return {
            'level': complexity_level,
            'score': complexity_score,
            'indicators': indicators
        }
    
    def _generate_gb_recommendations(self, feature_importance, complexity_indicators):
        """
        Generate business recommendations based on Gradient Boosting insights.
        
        Args:
            feature_importance (dict): Feature importance scores
            complexity_indicators (dict): Model complexity indicators
            
        Returns:
            list: Business recommendations
        """
        recommendations = []
        
        # Analyze convergence patterns
        if complexity_indicators['convergence_speed'] > 0.8:
            recommendations.append("Model converged quickly - patterns are relatively simple and stable")
        else:
            recommendations.append("Model required many iterations - complex patterns detected, consider feature engineering")
        
        # Analyze top features
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        for feature, importance in top_features:
            if importance > 0.15:  # High importance threshold
                if 'campaign' in feature.lower():
                    recommendations.append(f"Campaign feature {feature} is critical - focus on campaign optimization")
                elif 'contact' in feature.lower():
                    recommendations.append(f"Contact strategy {feature} significantly impacts outcomes")
                elif any(eng_feat in feature.lower() for eng_feat in ['age_bin', 'customer_value', 'intensity']):
                    recommendations.append(f"Engineered feature {feature} is highly predictive - validate business logic")
        
        # Model complexity recommendations
        if complexity_indicators['improvement_magnitude'] > 0.2:
            recommendations.append("High model improvement suggests strong predictive patterns - consider ensemble approaches")
        
        return recommendations
    
    def __str__(self):
        """String representation with model parameters."""
        params = f"n_estimators={self.n_estimators}, learning_rate={self.learning_rate}, max_depth={self.max_depth}"
        status = "Trained" if self.is_trained else "Not Trained"
        return f"Gradient Boosting ({params}) - {status}"
