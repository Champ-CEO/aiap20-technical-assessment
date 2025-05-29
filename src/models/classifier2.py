"""
Classifier 2: Random Forest

Phase 6 top performer - excellent with categorical features and provides feature importance.
Robust ensemble method with built-in feature selection and handling of mixed data types.

Key Features:
- Excellent performance with categorical features
- Built-in feature importance ranking
- Robust to outliers and missing values
- Handles feature interactions naturally
- Provides confidence estimates through voting
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from .base_classifier import BaseClassifier


class RandomForestClassifier(BaseClassifier):
    """
    Random Forest classifier for term deposit subscription prediction.
    
    Optimized for performance with categorical features and feature importance analysis.
    Provides robust predictions with ensemble voting and natural feature selection.
    """
    
    def __init__(self, 
                 n_estimators=100, 
                 max_depth=None, 
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_features='sqrt',
                 bootstrap=True,
                 random_state=42):
        """
        Initialize Random Forest classifier.
        
        Args:
            n_estimators (int): Number of trees in the forest
            max_depth (int): Maximum depth of trees (None for unlimited)
            min_samples_split (int): Minimum samples required to split node
            min_samples_leaf (int): Minimum samples required at leaf node
            max_features (str/int): Number of features for best split
            bootstrap (bool): Whether to use bootstrap sampling
            random_state (int): Random seed for reproducibility
        """
        super().__init__("Random Forest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        
    def _create_model(self):
        """
        Create Random Forest model instance.
        
        Returns:
            RandomForestClassifier: Configured sklearn model
        """
        return SklearnRandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            class_weight='balanced',  # Handle class imbalance
            n_jobs=-1  # Use all available cores
        )
    
    def get_feature_importance_detailed(self):
        """
        Get detailed feature importance with statistics.
        
        Returns:
            dict: Detailed feature importance analysis
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        # Get feature importances from all trees
        importances = self.model.feature_importances_
        
        # Calculate standard deviation across trees
        tree_importances = [tree.feature_importances_ for tree in self.model.estimators_]
        importances_std = np.std(tree_importances, axis=0)
        
        # Create detailed analysis
        detailed_importance = {}
        for i, feature in enumerate(self.feature_names):
            detailed_importance[feature] = {
                'importance': importances[i],
                'std': importances_std[i],
                'rank': None  # Will be filled below
            }
        
        # Add rankings
        sorted_features = sorted(detailed_importance.items(), 
                               key=lambda x: x[1]['importance'], reverse=True)
        for rank, (feature, info) in enumerate(sorted_features, 1):
            detailed_importance[feature]['rank'] = rank
        
        return detailed_importance
    
    def get_top_features_by_importance(self, n=10, include_engineered=True):
        """
        Get top features by importance with focus on engineered features.
        
        Args:
            n (int): Number of top features to return
            include_engineered (bool): Whether to highlight engineered features
            
        Returns:
            dict: Top features with importance scores and types
        """
        detailed_importance = self.get_feature_importance_detailed()
        
        # Define engineered features from Phase 5
        engineered_features = [
            'age_bin', 'customer_value_segment', 'campaign_intensity',
            'education_job_segment', 'recent_contact_flag', 'contact_effectiveness_score',
            'financial_risk_score', 'risk_category', 'is_high_risk',
            'high_intensity_flag', 'is_premium_customer', 'contact_recency'
        ]
        
        # Sort by importance
        sorted_features = sorted(detailed_importance.items(), 
                               key=lambda x: x[1]['importance'], reverse=True)
        
        top_features = {}
        for i, (feature, info) in enumerate(sorted_features[:n]):
            feature_type = 'engineered' if feature in engineered_features else 'original'
            top_features[feature] = {
                'importance': info['importance'],
                'rank': i + 1,
                'type': feature_type,
                'std': info['std']
            }
        
        return top_features
    
    def analyze_feature_interactions(self, X_sample=None, n_features=5):
        """
        Analyze feature interactions using tree paths.
        
        Args:
            X_sample (pd.DataFrame): Sample data for analysis
            n_features (int): Number of top features to analyze
            
        Returns:
            dict: Feature interaction analysis
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before analyzing interactions")
        
        # Get top features
        top_features = list(self.get_top_features_by_importance(n_features).keys())
        
        # Analyze tree structures for interactions
        interaction_counts = {}
        
        for tree in self.model.estimators_:
            # Get tree structure
            tree_structure = tree.tree_
            
            # Analyze splits for feature co-occurrence
            for node_id in range(tree_structure.node_count):
                if tree_structure.children_left[node_id] != tree_structure.children_right[node_id]:
                    # This is a split node
                    feature_idx = tree_structure.feature[node_id]
                    if feature_idx < len(self.feature_names):
                        feature_name = self.feature_names[feature_idx]
                        if feature_name in top_features:
                            # Count this feature's usage
                            if feature_name not in interaction_counts:
                                interaction_counts[feature_name] = 0
                            interaction_counts[feature_name] += 1
        
        return {
            'top_features_analyzed': top_features,
            'feature_usage_in_trees': interaction_counts,
            'interaction_strength': {
                feature: count / self.n_estimators 
                for feature, count in interaction_counts.items()
            }
        }
    
    def get_prediction_confidence(self, X):
        """
        Get prediction confidence based on tree voting.
        
        Args:
            X (pd.DataFrame): Features for prediction
            
        Returns:
            dict: Prediction confidence analysis
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting confidence")
        
        # Get predictions from all trees
        tree_predictions = np.array([tree.predict(self._encode_categorical_features(X, fit=False)) 
                                   for tree in self.model.estimators_])
        
        # Calculate voting statistics
        final_predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        # Calculate confidence metrics
        confidence_scores = []
        for i in range(len(X)):
            # Count votes for each class
            votes = tree_predictions[:, i]
            vote_counts = np.bincount(votes, minlength=2)
            
            # Confidence is the proportion of trees voting for the predicted class
            predicted_class = final_predictions[i]
            confidence = vote_counts[predicted_class] / self.n_estimators
            confidence_scores.append(confidence)
        
        return {
            'predictions': final_predictions,
            'probabilities': probabilities,
            'confidence_scores': np.array(confidence_scores),
            'mean_confidence': np.mean(confidence_scores),
            'high_confidence_threshold': 0.8,
            'high_confidence_predictions': np.sum(np.array(confidence_scores) > 0.8)
        }
    
    def get_business_insights(self):
        """
        Generate business insights from the Random Forest model.
        
        Returns:
            dict: Business insights and recommendations
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before generating insights")
        
        # Get feature importance analysis
        top_features = self.get_top_features_by_importance(n=10)
        detailed_importance = self.get_feature_importance_detailed()
        
        # Separate engineered vs original features
        engineered_features = {k: v for k, v in top_features.items() if v['type'] == 'engineered'}
        original_features = {k: v for k, v in top_features.items() if v['type'] == 'original'}
        
        insights = {
            'model_type': 'Random Forest',
            'interpretability': 'Medium-High - feature importance and tree-based insights',
            'total_trees': self.n_estimators,
            'top_engineered_features': engineered_features,
            'top_original_features': original_features,
            'feature_importance_stability': self._analyze_importance_stability(detailed_importance),
            'business_recommendations': self._generate_rf_recommendations(top_features)
        }
        
        return insights
    
    def _analyze_importance_stability(self, detailed_importance):
        """
        Analyze stability of feature importance across trees.
        
        Args:
            detailed_importance (dict): Detailed importance analysis
            
        Returns:
            dict: Stability analysis
        """
        # Calculate coefficient of variation for top features
        stability_metrics = {}
        for feature, info in detailed_importance.items():
            if info['importance'] > 0:
                cv = info['std'] / info['importance']  # Coefficient of variation
                stability_metrics[feature] = {
                    'coefficient_of_variation': cv,
                    'stability': 'high' if cv < 0.5 else 'medium' if cv < 1.0 else 'low'
                }
        
        return stability_metrics
    
    def _generate_rf_recommendations(self, top_features):
        """
        Generate business recommendations based on Random Forest insights.
        
        Args:
            top_features (dict): Top features with importance
            
        Returns:
            list: Business recommendations
        """
        recommendations = []
        
        # Analyze engineered features
        engineered_count = sum(1 for f in top_features.values() if f['type'] == 'engineered')
        if engineered_count > 5:
            recommendations.append("Engineered features are highly predictive - continue feature engineering efforts")
        
        # Analyze specific feature types
        for feature, info in top_features.items():
            importance = info['importance']
            if importance > 0.1:  # High importance threshold
                if 'campaign' in feature.lower():
                    recommendations.append(f"Campaign strategy ({feature}) is critical - optimize campaign parameters")
                elif 'contact' in feature.lower():
                    recommendations.append(f"Contact approach ({feature}) significantly impacts success")
                elif 'age' in feature.lower():
                    recommendations.append(f"Age segmentation ({feature}) is important for targeting")
                elif 'education' in feature.lower():
                    recommendations.append(f"Education level ({feature}) drives subscription decisions")
        
        return recommendations
    
    def __str__(self):
        """String representation with model parameters."""
        params = f"n_estimators={self.n_estimators}, max_depth={self.max_depth}"
        status = "Trained" if self.is_trained else "Not Trained"
        return f"Random Forest ({params}) - {status}"
