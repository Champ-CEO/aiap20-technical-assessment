"""
Classifier 1: Logistic Regression

Phase 6 proven performer - interpretable baseline for marketing insights and coefficient analysis.
Provides clear feature importance through coefficients and excellent interpretability for business users.

Key Features:
- Interpretable coefficients for marketing insights
- Fast training and prediction
- Probability estimates for customer targeting
- Regularization to prevent overfitting
- Handles categorical features through encoding
"""

import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from .base_classifier import BaseClassifier


class LogisticRegressionClassifier(BaseClassifier):
    """
    Logistic Regression classifier for term deposit subscription prediction.
    
    Optimized for interpretability and marketing insights with coefficient analysis.
    Provides baseline performance with clear business interpretation.
    """
    
    def __init__(self, 
                 C=1.0, 
                 penalty='l2', 
                 solver='lbfgs', 
                 max_iter=1000,
                 random_state=42):
        """
        Initialize Logistic Regression classifier.
        
        Args:
            C (float): Regularization strength (default: 1.0)
            penalty (str): Regularization type ('l1', 'l2', 'elasticnet', 'none')
            solver (str): Optimization algorithm ('lbfgs', 'liblinear', 'newton-cg', etc.)
            max_iter (int): Maximum iterations for convergence
            random_state (int): Random seed for reproducibility
        """
        super().__init__("Logistic Regression")
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state
        
    def _create_model(self):
        """
        Create Logistic Regression model instance.
        
        Returns:
            LogisticRegression: Configured sklearn model
        """
        return SklearnLogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state,
            class_weight='balanced'  # Handle class imbalance
        )
    
    def get_coefficients(self):
        """
        Get model coefficients for interpretability.
        
        Returns:
            dict: Feature coefficients mapping
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting coefficients")
        
        if self.feature_names is None:
            raise ValueError("Feature names not available")
        
        coefficients = self.model.coef_[0]
        return dict(zip(self.feature_names, coefficients))
    
    def get_odds_ratios(self):
        """
        Get odds ratios for business interpretation.
        
        Returns:
            dict: Feature odds ratios mapping
        """
        coefficients = self.get_coefficients()
        return {feature: np.exp(coef) for feature, coef in coefficients.items()}
    
    def get_top_features(self, n=10, by_importance=True):
        """
        Get top features by coefficient magnitude or importance.
        
        Args:
            n (int): Number of top features to return
            by_importance (bool): Sort by absolute coefficient value
            
        Returns:
            list: Top feature names
        """
        if by_importance:
            importance = self.get_feature_importance()
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        else:
            coefficients = self.get_coefficients()
            sorted_features = sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)
        
        return [feature for feature, _ in sorted_features[:n]]
    
    def interpret_prediction(self, X, feature_names=None):
        """
        Provide interpretation for predictions.
        
        Args:
            X (pd.DataFrame): Features for interpretation
            feature_names (list): Specific features to interpret
            
        Returns:
            dict: Interpretation results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before interpretation")
        
        # Get predictions and probabilities
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        # Get coefficients
        coefficients = self.get_coefficients()
        
        # Select features for interpretation
        if feature_names is None:
            feature_names = self.get_top_features(n=5)
        
        interpretation = {
            'predictions': predictions,
            'probabilities': probabilities,
            'top_features': feature_names,
            'feature_contributions': {
                feature: coefficients.get(feature, 0) 
                for feature in feature_names
            }
        }
        
        return interpretation
    
    def get_business_insights(self):
        """
        Generate business insights from the trained model.
        
        Returns:
            dict: Business insights and recommendations
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before generating insights")
        
        # Get feature importance and coefficients
        importance = self.get_feature_importance()
        coefficients = self.get_coefficients()
        odds_ratios = self.get_odds_ratios()
        
        # Identify top positive and negative predictors
        positive_predictors = {k: v for k, v in coefficients.items() if v > 0}
        negative_predictors = {k: v for k, v in coefficients.items() if v < 0}
        
        # Sort by magnitude
        top_positive = sorted(positive_predictors.items(), key=lambda x: x[1], reverse=True)[:5]
        top_negative = sorted(negative_predictors.items(), key=lambda x: x[1])[:5]
        
        insights = {
            'model_type': 'Logistic Regression',
            'interpretability': 'High - coefficients show direct feature impact',
            'top_positive_predictors': top_positive,
            'top_negative_predictors': top_negative,
            'feature_importance': dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]),
            'business_recommendations': self._generate_recommendations(top_positive, top_negative)
        }
        
        return insights
    
    def _generate_recommendations(self, top_positive, top_negative):
        """
        Generate business recommendations based on model insights.
        
        Args:
            top_positive (list): Top positive predictors
            top_negative (list): Top negative predictors
            
        Returns:
            list: Business recommendations
        """
        recommendations = []
        
        # Analyze positive predictors
        for feature, coef in top_positive:
            if 'age' in feature.lower():
                recommendations.append(f"Target customers in {feature} category (positive impact: {coef:.3f})")
            elif 'education' in feature.lower():
                recommendations.append(f"Focus on {feature} segment (increases subscription likelihood)")
            elif 'campaign' in feature.lower():
                recommendations.append(f"Optimize {feature} strategy (positive coefficient: {coef:.3f})")
            elif 'contact' in feature.lower():
                recommendations.append(f"Leverage {feature} approach for better results")
        
        # Analyze negative predictors
        for feature, coef in top_negative:
            if 'loan' in feature.lower():
                recommendations.append(f"Consider special offers for {feature} customers (negative impact: {coef:.3f})")
            elif 'default' in feature.lower():
                recommendations.append(f"Address concerns related to {feature} (reduces subscription)")
        
        return recommendations
    
    def __str__(self):
        """String representation with model parameters."""
        params = f"C={self.C}, penalty={self.penalty}, solver={self.solver}"
        status = "Trained" if self.is_trained else "Not Trained"
        return f"Logistic Regression ({params}) - {status}"
