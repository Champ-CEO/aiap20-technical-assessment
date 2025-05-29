"""
Base Classifier Module

Provides base class for all Phase 7 model implementations with common functionality.
Handles categorical encoding, Phase 6 integration, and business metrics calculation.
"""

import pandas as pd
import numpy as np
import time
from abc import ABC, abstractmethod
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from pathlib import Path
import sys

# Phase 6 integration - simplified for now
PHASE6_INTEGRATION_AVAILABLE = False
EXPECTED_RECORD_COUNT = 41188
EXPECTED_TOTAL_FEATURES = 45
PERFORMANCE_STANDARD = 97000


class BaseClassifier(ABC):
    """
    Base class for all Phase 7 model implementations.

    Provides common functionality for:
    - Categorical encoding with LabelEncoder
    - Phase 6 integration
    - Performance monitoring
    - Business metrics calculation
    - Feature importance analysis
    """

    def __init__(self, name: str):
        """
        Initialize base classifier.

        Args:
            name (str): Name of the classifier
        """
        self.name = name
        self.model = None
        self.label_encoders = {}
        self.feature_names = None
        self.is_trained = False
        self.training_time = None
        self.performance_metrics = {}

    @abstractmethod
    def _create_model(self):
        """Create the specific model instance. Must be implemented by subclasses."""
        pass

    def _encode_categorical_features(self, X, fit=True):
        """
        Encode categorical features using LabelEncoder.

        Args:
            X (pd.DataFrame): Input features
            fit (bool): Whether to fit encoders (True for training, False for prediction)

        Returns:
            pd.DataFrame: Encoded features
        """
        X_encoded = X.copy()

        # Identify categorical columns
        categorical_columns = X_encoded.select_dtypes(include=["object"]).columns

        for col in categorical_columns:
            if fit:
                # Fit and transform for training
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                X_encoded[col] = self.label_encoders[col].fit_transform(
                    X_encoded[col].astype(str)
                )
            else:
                # Transform only for prediction
                if col in self.label_encoders:
                    # Handle unseen categories
                    unique_values = set(X_encoded[col].astype(str))
                    known_values = set(self.label_encoders[col].classes_)
                    unseen_values = unique_values - known_values

                    if unseen_values:
                        # Map unseen values to the most frequent class
                        most_frequent_class = self.label_encoders[col].classes_[0]
                        X_encoded[col] = (
                            X_encoded[col]
                            .astype(str)
                            .replace(list(unseen_values), most_frequent_class)
                        )

                    X_encoded[col] = self.label_encoders[col].transform(
                        X_encoded[col].astype(str)
                    )
                else:
                    # If encoder not found, use simple numeric encoding
                    X_encoded[col] = pd.Categorical(X_encoded[col]).codes

        return X_encoded

    def fit(self, X, y):
        """
        Train the classifier with categorical encoding.

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

        # Create and train model
        if self.model is None:
            self.model = self._create_model()

        # Train the model
        self.model.fit(X_encoded, y)

        # Record training time
        self.training_time = time.time() - start_time
        self.is_trained = True

        return self

    def predict(self, X):
        """
        Make predictions with categorical encoding.

        Args:
            X (pd.DataFrame): Features for prediction

        Returns:
            np.array: Predictions
        """
        if not self.is_trained:
            raise ValueError(f"{self.name} must be trained before making predictions")

        # Encode categorical features
        X_encoded = self._encode_categorical_features(X, fit=False)

        # Make predictions
        return self.model.predict(X_encoded)

    def predict_proba(self, X):
        """
        Predict class probabilities with categorical encoding.

        Args:
            X (pd.DataFrame): Features for prediction

        Returns:
            np.array: Class probabilities
        """
        if not self.is_trained:
            raise ValueError(f"{self.name} must be trained before making predictions")

        # Check if model supports predict_proba
        if not hasattr(self.model, "predict_proba"):
            raise ValueError(f"{self.name} does not support probability predictions")

        # Encode categorical features
        X_encoded = self._encode_categorical_features(X, fit=False)

        # Make probability predictions
        return self.model.predict_proba(X_encoded)

    def get_feature_importance(self):
        """
        Get feature importance if available.

        Returns:
            dict: Feature importance mapping
        """
        if not self.is_trained:
            raise ValueError(
                f"{self.name} must be trained before getting feature importance"
            )

        if hasattr(self.model, "feature_importances_"):
            return dict(zip(self.feature_names, self.model.feature_importances_))
        elif hasattr(self.model, "coef_"):
            # For linear models, use absolute coefficients
            return dict(zip(self.feature_names, np.abs(self.model.coef_[0])))
        else:
            return {}

    def evaluate(self, X, y):
        """
        Evaluate model performance.

        Args:
            X (pd.DataFrame): Test features
            y (pd.Series): Test target

        Returns:
            dict: Performance metrics
        """
        if not self.is_trained:
            raise ValueError(f"{self.name} must be trained before evaluation")

        # Make predictions
        y_pred = self.predict(X)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(
                y, y_pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(y, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y, y_pred, average="weighted", zero_division=0),
        }

        # Add AUC if probability predictions are available
        try:
            y_proba = self.predict_proba(X)
            if y_proba.shape[1] == 2:  # Binary classification
                metrics["auc"] = roc_auc_score(y, y_proba[:, 1])
        except:
            pass

        self.performance_metrics = metrics
        return metrics

    def __str__(self):
        """String representation of the classifier."""
        status = "Trained" if self.is_trained else "Not Trained"
        return f"{self.name} ({status})"

    def __repr__(self):
        """Detailed representation of the classifier."""
        return (
            f"{self.__class__.__name__}(name='{self.name}', trained={self.is_trained})"
        )
