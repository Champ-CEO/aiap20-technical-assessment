"""
Classifier 5: Support Vector Machine

Strong performance on structured data with clear decision boundaries.
Excellent for finding optimal separating hyperplanes in high-dimensional feature spaces.

Key Features:
- Strong performance on structured data
- Clear decision boundaries
- Kernel trick for non-linear patterns
- Robust to outliers
- Good generalization with proper regularization
- Effective in high-dimensional spaces
"""

import numpy as np
import time
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from .base_classifier import BaseClassifier


class SVMClassifier(BaseClassifier):
    """
    Support Vector Machine classifier for term deposit subscription prediction.
    
    Optimized for finding optimal decision boundaries with kernel support for non-linear patterns.
    Provides robust classification with good generalization properties.
    """
    
    def __init__(self, 
                 C=1.0, 
                 kernel='rbf', 
                 degree=3,
                 gamma='scale',
                 coef0=0.0,
                 probability=True,
                 class_weight='balanced',
                 random_state=42):
        """
        Initialize SVM classifier.
        
        Args:
            C (float): Regularization parameter
            kernel (str): Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
            degree (int): Degree for polynomial kernel
            gamma (str/float): Kernel coefficient ('scale', 'auto', or float)
            coef0 (float): Independent term in kernel function
            probability (bool): Whether to enable probability estimates
            class_weight (str/dict): Class weight handling
            random_state (int): Random seed for reproducibility
        """
        super().__init__("Support Vector Machine")
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.probability = probability
        self.class_weight = class_weight
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.is_scaled = False
        
    def _create_model(self):
        """
        Create SVM model instance.
        
        Returns:
            SVC: Configured sklearn model
        """
        return SVC(
            C=self.C,
            kernel=self.kernel,
            degree=self.degree,
            gamma=self.gamma,
            coef0=self.coef0,
            probability=self.probability,
            class_weight=self.class_weight,
            random_state=self.random_state,
            cache_size=1000  # Increase cache for better performance
        )
    
    def fit(self, X, y):
        """
        Train the SVM classifier with feature scaling.
        
        Args:
            X (pd.DataFrame): Training features
            y (pd.Series): Training target
            
        Returns:
            self: Trained classifier
        """
        start_time = time.time()
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Encode categorical features
        X_encoded = self._encode_categorical_features(X, fit=True)
        
        # Scale features for SVM
        X_scaled = self.scaler.fit_transform(X_encoded)
        self.is_scaled = True
        
        # Create and train model
        if self.model is None:
            self.model = self._create_model()
        
        # Train the model
        self.model.fit(X_scaled, y)
        
        # Record training time
        self.training_time = time.time() - start_time
        self.is_trained = True
        
        return self
    
    def predict(self, X):
        """
        Make predictions with feature scaling.
        
        Args:
            X (pd.DataFrame): Features for prediction
            
        Returns:
            np.array: Predictions
        """
        if not self.is_trained:
            raise ValueError(f"{self.name} must be trained before making predictions")
        
        # Encode and scale features
        X_encoded = self._encode_categorical_features(X, fit=False)
        X_scaled = self.scaler.transform(X_encoded)
        
        # Make predictions
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """
        Predict class probabilities with feature scaling.
        
        Args:
            X (pd.DataFrame): Features for prediction
            
        Returns:
            np.array: Class probabilities
        """
        if not self.is_trained:
            raise ValueError(f"{self.name} must be trained before making predictions")
        
        if not self.probability:
            raise ValueError("Probability estimation not enabled. Set probability=True during initialization.")
        
        # Encode and scale features
        X_encoded = self._encode_categorical_features(X, fit=False)
        X_scaled = self.scaler.transform(X_encoded)
        
        # Make probability predictions
        return self.model.predict_proba(X_scaled)
    
    def get_support_vectors_info(self):
        """
        Get information about support vectors.
        
        Returns:
            dict: Support vector analysis
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting support vector info")
        
        n_support = self.model.n_support_
        support_vectors = self.model.support_vectors_
        
        info = {
            'n_support_vectors': n_support,
            'total_support_vectors': np.sum(n_support),
            'support_vector_ratio': np.sum(n_support) / len(self.model.support_),
            'support_vectors_per_class': dict(zip(self.model.classes_, n_support)),
            'dual_coefficients_shape': self.model.dual_coef_.shape if hasattr(self.model, 'dual_coef_') else None
        }
        
        return info
    
    def get_decision_boundary_analysis(self):
        """
        Analyze decision boundary characteristics.
        
        Returns:
            dict: Decision boundary analysis
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before analyzing decision boundary")
        
        analysis = {
            'kernel_type': self.kernel,
            'regularization_strength': self.C,
            'support_vector_info': self.get_support_vectors_info(),
            'model_complexity': self._assess_model_complexity()
        }
        
        # Add kernel-specific analysis
        if self.kernel == 'linear':
            analysis['linear_coefficients'] = self._analyze_linear_coefficients()
        elif self.kernel == 'rbf':
            analysis['rbf_characteristics'] = self._analyze_rbf_characteristics()
        elif self.kernel == 'poly':
            analysis['polynomial_characteristics'] = self._analyze_polynomial_characteristics()
        
        return analysis
    
    def _assess_model_complexity(self):
        """
        Assess model complexity based on support vectors and parameters.
        
        Returns:
            dict: Model complexity assessment
        """
        sv_info = self.get_support_vectors_info()
        sv_ratio = sv_info['support_vector_ratio']
        
        # Complexity indicators
        complexity_score = 0
        
        if sv_ratio > 0.5:  # High support vector ratio indicates complex boundary
            complexity_score += 2
        elif sv_ratio > 0.2:
            complexity_score += 1
        
        if self.C > 10:  # High C indicates complex model
            complexity_score += 1
        elif self.C < 0.1:  # Low C indicates simple model
            complexity_score -= 1
        
        if self.kernel in ['poly', 'rbf']:  # Non-linear kernels add complexity
            complexity_score += 1
        
        complexity_levels = ['Very Simple', 'Simple', 'Moderate', 'Complex', 'Very Complex']
        complexity_level = complexity_levels[max(0, min(complexity_score + 1, 4))]
        
        return {
            'complexity_level': complexity_level,
            'complexity_score': complexity_score,
            'support_vector_ratio': sv_ratio,
            'regularization_strength': self.C,
            'kernel_complexity': self.kernel
        }
    
    def _analyze_linear_coefficients(self):
        """
        Analyze linear SVM coefficients for feature importance.
        
        Returns:
            dict: Linear coefficient analysis
        """
        if self.kernel != 'linear':
            return {}
        
        coefficients = self.model.coef_[0]
        feature_importance = dict(zip(self.feature_names, np.abs(coefficients)))
        
        return {
            'coefficients': dict(zip(self.feature_names, coefficients)),
            'feature_importance': feature_importance,
            'top_features': sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        }
    
    def _analyze_rbf_characteristics(self):
        """
        Analyze RBF kernel characteristics.
        
        Returns:
            dict: RBF kernel analysis
        """
        if self.kernel != 'rbf':
            return {}
        
        return {
            'gamma_value': self.model.gamma,
            'kernel_interpretation': 'RBF kernel creates local decision boundaries',
            'gamma_effect': 'Higher gamma = more complex, localized boundaries',
            'support_vector_influence': 'Each support vector creates a local influence region'
        }
    
    def _analyze_polynomial_characteristics(self):
        """
        Analyze polynomial kernel characteristics.
        
        Returns:
            dict: Polynomial kernel analysis
        """
        if self.kernel != 'poly':
            return {}
        
        return {
            'degree': self.degree,
            'coef0': self.coef0,
            'kernel_interpretation': f'Polynomial kernel of degree {self.degree}',
            'complexity_note': 'Higher degree = more complex feature interactions'
        }
    
    def get_feature_importance(self):
        """
        Get feature importance (only available for linear kernel).
        
        Returns:
            dict: Feature importance mapping
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        if self.kernel == 'linear':
            coefficients = self.model.coef_[0]
            return dict(zip(self.feature_names, np.abs(coefficients)))
        else:
            # For non-linear kernels, feature importance is not directly available
            return {}
    
    def get_prediction_confidence(self, X):
        """
        Get prediction confidence based on distance from decision boundary.
        
        Args:
            X (pd.DataFrame): Features for confidence estimation
            
        Returns:
            dict: Prediction confidence analysis
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting confidence")
        
        # Encode and scale features
        X_encoded = self._encode_categorical_features(X, fit=False)
        X_scaled = self.scaler.transform(X_encoded)
        
        # Get decision function values (distance from boundary)
        decision_values = self.model.decision_function(X_scaled)
        predictions = self.model.predict(X_scaled)
        
        # Calculate confidence based on distance from boundary
        confidence_scores = np.abs(decision_values)
        
        # Normalize confidence scores
        max_confidence = np.max(confidence_scores)
        if max_confidence > 0:
            normalized_confidence = confidence_scores / max_confidence
        else:
            normalized_confidence = np.ones_like(confidence_scores)
        
        return {
            'predictions': predictions,
            'decision_values': decision_values,
            'confidence_scores': normalized_confidence,
            'mean_confidence': np.mean(normalized_confidence),
            'high_confidence_threshold': 0.7,
            'high_confidence_predictions': np.sum(normalized_confidence > 0.7)
        }
    
    def get_business_insights(self):
        """
        Generate business insights from the SVM model.
        
        Returns:
            dict: Business insights and recommendations
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before generating insights")
        
        # Get comprehensive analysis
        boundary_analysis = self.get_decision_boundary_analysis()
        sv_info = self.get_support_vectors_info()
        
        insights = {
            'model_type': 'Support Vector Machine',
            'interpretability': self._get_interpretability_level(),
            'decision_boundary': boundary_analysis,
            'support_vectors': sv_info,
            'model_characteristics': self._get_model_characteristics(),
            'business_recommendations': self._generate_svm_recommendations(boundary_analysis, sv_info)
        }
        
        # Add feature importance for linear kernel
        if self.kernel == 'linear':
            insights['feature_importance'] = self.get_feature_importance()
        
        return insights
    
    def _get_interpretability_level(self):
        """
        Determine interpretability level based on kernel type.
        
        Returns:
            str: Interpretability description
        """
        if self.kernel == 'linear':
            return 'High - linear coefficients show direct feature impact'
        elif self.kernel == 'poly':
            return 'Medium - polynomial interactions can be analyzed'
        elif self.kernel == 'rbf':
            return 'Low-Medium - local decision boundaries, support vector analysis'
        else:
            return 'Low - complex kernel, focus on performance metrics'
    
    def _get_model_characteristics(self):
        """
        Get key model characteristics for business understanding.
        
        Returns:
            dict: Model characteristics
        """
        sv_info = self.get_support_vectors_info()
        
        return {
            'kernel_type': self.kernel,
            'regularization': f'C={self.C}',
            'support_vector_dependency': f"{sv_info['support_vector_ratio']:.1%} of training data",
            'decision_boundary_type': 'Linear' if self.kernel == 'linear' else 'Non-linear',
            'probability_estimates': 'Available' if self.probability else 'Not available'
        }
    
    def _generate_svm_recommendations(self, boundary_analysis, sv_info):
        """
        Generate business recommendations based on SVM insights.
        
        Args:
            boundary_analysis (dict): Decision boundary analysis
            sv_info (dict): Support vector information
            
        Returns:
            list: Business recommendations
        """
        recommendations = []
        
        # Support vector ratio insights
        sv_ratio = sv_info['support_vector_ratio']
        if sv_ratio > 0.5:
            recommendations.append("High support vector ratio suggests complex patterns - consider feature engineering")
        elif sv_ratio < 0.1:
            recommendations.append("Low support vector ratio indicates clear separation - model is confident")
        
        # Kernel-specific recommendations
        if self.kernel == 'linear':
            recommendations.append("Linear kernel provides interpretable feature weights for business insights")
        elif self.kernel == 'rbf':
            recommendations.append("RBF kernel captures local patterns - good for complex customer behaviors")
        
        # Regularization insights
        if self.C > 10:
            recommendations.append("High regularization parameter - model focuses on training accuracy")
        elif self.C < 0.1:
            recommendations.append("Low regularization parameter - model prioritizes generalization")
        
        # Model complexity recommendations
        complexity = boundary_analysis['model_complexity']['complexity_level']
        if complexity in ['Complex', 'Very Complex']:
            recommendations.append("Complex decision boundary - validate on holdout data to ensure generalization")
        else:
            recommendations.append("Simple decision boundary - model should generalize well to new data")
        
        return recommendations
    
    def __str__(self):
        """String representation with model parameters."""
        params = f"C={self.C}, kernel={self.kernel}"
        if self.kernel == 'poly':
            params += f", degree={self.degree}"
        elif self.kernel == 'rbf':
            params += f", gamma={self.gamma}"
        
        status = "Trained" if self.is_trained else "Not Trained"
        return f"SVM ({params}) - {status}"
