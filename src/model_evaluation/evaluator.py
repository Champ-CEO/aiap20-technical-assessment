"""
Model Evaluator Module

Implements comprehensive model evaluation with performance metrics calculation.
Supports accuracy, precision, recall, F1, AUC calculation for all trained models.

Key Features:
- Load and evaluate all 5 trained models
- Calculate standard classification metrics
- Customer segment-aware evaluation
- Performance monitoring (>97K records/second)
- Integration with Phase 7 model artifacts
"""

import pandas as pd
import numpy as np
import pickle
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
TRAINED_MODELS_DIR = "trained_models"
FEATURED_DATA_PATH = "data/featured/featured-db.csv"
PERFORMANCE_STANDARD = 97000  # >97K records/second

# Phase 7 model names
EXPECTED_MODEL_NAMES = [
    "GradientBoosting",
    "NaiveBayes",
    "RandomForest",
    "LogisticRegression",
    "SVM",
]

# Customer segment rates from Phase 7
CUSTOMER_SEGMENT_RATES = {
    "Premium": 0.316,  # 31.6%
    "Standard": 0.577,  # 57.7%
    "Basic": 0.107,  # 10.7%
}


class ModelEvaluator:
    """
    Comprehensive model evaluator for Phase 8 implementation.

    Handles loading trained models, calculating performance metrics,
    and providing customer segment-aware evaluation.
    """

    def __init__(
        self, models_dir: str = TRAINED_MODELS_DIR, data_path: str = FEATURED_DATA_PATH
    ):
        """
        Initialize ModelEvaluator.

        Args:
            models_dir (str): Directory containing trained models
            data_path (str): Path to featured dataset
        """
        self.models_dir = Path(models_dir)
        self.data_path = data_path
        self.models = {}
        self.evaluation_results = {}
        self.performance_metrics = {}

        # Load test data
        self._load_test_data()

    def _load_test_data(self):
        """Load and prepare test data for evaluation."""
        try:
            # Load featured dataset
            self.data = pd.read_csv(self.data_path)
            logger.info(
                f"Loaded dataset: {len(self.data)} records, {len(self.data.columns)} features"
            )

            # Prepare features and target
            # Based on the error, models expect 'Subscription Status' to be the target
            # and 'is_high_risk' should be in the features
            target_column = "Subscription Status"

            if target_column in self.data.columns:
                self.y_test = self.data[target_column]
                self.X_test = self.data.drop(target_column, axis=1)
            elif "y" in self.data.columns:
                self.X_test = self.data.drop("y", axis=1)
                self.y_test = self.data["y"]
            else:
                # Use last column as target if neither found
                self.X_test = self.data.iloc[:, :-1]
                self.y_test = self.data.iloc[:, -1]

            logger.info(
                f"Test data prepared: {len(self.X_test)} samples, {len(self.X_test.columns)} features"
            )
            logger.info(
                f"Target column: {target_column if target_column in self.data.columns else 'last column'}"
            )
            logger.info(f"Target distribution: {self.y_test.value_counts().to_dict()}")

        except Exception as e:
            logger.error(f"Error loading test data: {str(e)}")
            raise

    def load_models(self) -> Dict[str, Any]:
        """
        Load all trained models from the models directory.

        Returns:
            Dict[str, Any]: Dictionary of loaded models
        """
        start_time = time.time()
        loaded_models = {}

        for model_name in EXPECTED_MODEL_NAMES:
            model_file = self.models_dir / f"{model_name.lower()}_model.pkl"

            try:
                with open(model_file, "rb") as f:
                    model = pickle.load(f)
                loaded_models[model_name] = model
                logger.info(f"Loaded model: {model_name}")

            except Exception as e:
                logger.error(f"Error loading {model_name}: {str(e)}")
                loaded_models[model_name] = None

        self.models = loaded_models

        # Performance tracking
        load_time = time.time() - start_time
        models_loaded = sum(1 for model in loaded_models.values() if model is not None)

        self.performance_metrics["model_loading"] = {
            "load_time": load_time,
            "models_loaded": models_loaded,
            "total_models": len(EXPECTED_MODEL_NAMES),
        }

        logger.info(
            f"Model loading completed: {models_loaded}/{len(EXPECTED_MODEL_NAMES)} models in {load_time:.2f}s"
        )
        return loaded_models

    def evaluate_model(self, model_name: str) -> Dict[str, Any]:
        """
        Evaluate a single model with comprehensive metrics.

        Args:
            model_name (str): Name of the model to evaluate

        Returns:
            Dict[str, Any]: Comprehensive evaluation metrics
        """
        if model_name not in self.models or self.models[model_name] is None:
            raise ValueError(f"Model {model_name} not loaded or unavailable")

        start_time = time.time()
        model = self.models[model_name]

        try:
            # Make predictions
            y_pred = model.predict(self.X_test)

            # Get prediction probabilities if available
            try:
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            except:
                y_pred_proba = None

            # Calculate standard metrics
            metrics = {
                "accuracy": accuracy_score(self.y_test, y_pred),
                "precision": precision_score(
                    self.y_test, y_pred, average="weighted", zero_division=0
                ),
                "recall": recall_score(
                    self.y_test, y_pred, average="weighted", zero_division=0
                ),
                "f1_score": f1_score(
                    self.y_test, y_pred, average="weighted", zero_division=0
                ),
                "confusion_matrix": confusion_matrix(self.y_test, y_pred).tolist(),
                "classification_report": classification_report(
                    self.y_test, y_pred, output_dict=True
                ),
            }

            # Add AUC if probabilities available
            if y_pred_proba is not None:
                try:
                    metrics["auc_score"] = roc_auc_score(self.y_test, y_pred_proba)
                except:
                    metrics["auc_score"] = None
            else:
                metrics["auc_score"] = None

            # Performance tracking
            eval_time = time.time() - start_time
            records_per_second = len(self.X_test) / eval_time if eval_time > 0 else 0

            metrics["performance"] = {
                "evaluation_time": eval_time,
                "records_per_second": records_per_second,
                "meets_performance_standard": records_per_second
                >= PERFORMANCE_STANDARD,
                "test_samples": len(self.X_test),
            }

            logger.info(
                f"Evaluated {model_name}: {metrics['accuracy']:.4f} accuracy, {records_per_second:,.0f} records/sec"
            )

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {str(e)}")
            raise

    def evaluate_all_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all loaded models.

        Returns:
            Dict[str, Dict[str, Any]]: Evaluation results for all models
        """
        if not self.models:
            self.load_models()

        results = {}

        for model_name in EXPECTED_MODEL_NAMES:
            if model_name in self.models and self.models[model_name] is not None:
                try:
                    results[model_name] = self.evaluate_model(model_name)
                except Exception as e:
                    logger.error(f"Failed to evaluate {model_name}: {str(e)}")
                    results[model_name] = None
            else:
                logger.warning(f"Model {model_name} not available for evaluation")
                results[model_name] = None

        self.evaluation_results = results
        return results
