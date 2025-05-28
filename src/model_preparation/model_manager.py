"""
Model Manager Module

Implements model serialization, factory pattern, and model management utilities.
Supports 45-feature schema compatibility and provides clear interfaces for AI-assisted development.

Key Features:
- Model serialization with pickle and joblib support
- Factory pattern for model selection and comparison
- Feature importance analysis for engineered features
- 45-feature schema validation and compatibility
- Performance monitoring for >97K records/second standard
"""

import pandas as pd
import numpy as np
import pickle
import joblib
import time
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Type
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Model manager for serialization and lifecycle management.
    
    Handles model saving/loading with 45-feature schema compatibility
    and provides utilities for model comparison and analysis.
    """
    
    def __init__(self, feature_schema: Optional[List[str]] = None):
        """
        Initialize model manager.
        
        Args:
            feature_schema (List[str], optional): Expected feature names for validation
        """
        self.feature_schema = feature_schema
        self.expected_feature_count = 44  # 45 total - 1 target
        self.performance_standard = 97000  # records per second
        
        # Performance tracking
        self.performance_metrics = {
            'serialization_time': 0,
            'deserialization_time': 0,
            'records_per_second': 0,
            'operations_completed': []
        }
    
    def save_model(self, model: BaseEstimator, filepath: Union[str, Path],
                   method: str = 'joblib', include_metadata: bool = True) -> Dict[str, Any]:
        """
        Save model with 45-feature schema compatibility.
        
        Args:
            model (BaseEstimator): Trained model to save
            filepath (Union[str, Path]): Path to save the model
            method (str): Serialization method ('pickle' or 'joblib')
            include_metadata (bool): Whether to include model metadata
        
        Returns:
            Dict[str, Any]: Serialization results and metadata
        """
        start_time = time.time()
        
        try:
            filepath = Path(filepath)
            
            # Validate model before saving
            self._validate_model_for_serialization(model)
            
            # Prepare metadata
            metadata = {}
            if include_metadata:
                metadata = self._extract_model_metadata(model)
            
            # Serialize model
            if method == 'pickle':
                self._save_with_pickle(model, filepath, metadata)
            elif method == 'joblib':
                self._save_with_joblib(model, filepath, metadata)
            else:
                raise ValueError(f"Unsupported serialization method: {method}")
            
            # Record performance
            serialization_time = time.time() - start_time
            self.performance_metrics['serialization_time'] = serialization_time
            self.performance_metrics['operations_completed'].append('save')
            
            result = {
                'success': True,
                'filepath': str(filepath),
                'method': method,
                'serialization_time': serialization_time,
                'metadata': metadata
            }
            
            logger.info(f"Model saved successfully: {filepath} ({method})")
            
            return result
            
        except Exception as e:
            logger.error(f"Model saving failed: {str(e)}")
            raise ValueError(f"Model saving failed: {str(e)}")
    
    def load_model(self, filepath: Union[str, Path], method: str = 'auto',
                   validate_schema: bool = True) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """
        Load model with schema validation.
        
        Args:
            filepath (Union[str, Path]): Path to the saved model
            method (str): Deserialization method ('pickle', 'joblib', or 'auto')
            validate_schema (bool): Whether to validate feature schema
        
        Returns:
            Tuple[BaseEstimator, Dict[str, Any]]: Loaded model and metadata
        """
        start_time = time.time()
        
        try:
            filepath = Path(filepath)
            
            if not filepath.exists():
                raise FileNotFoundError(f"Model file not found: {filepath}")
            
            # Auto-detect method if needed
            if method == 'auto':
                method = self._detect_serialization_method(filepath)
            
            # Deserialize model
            if method == 'pickle':
                model, metadata = self._load_with_pickle(filepath)
            elif method == 'joblib':
                model, metadata = self._load_with_joblib(filepath)
            else:
                raise ValueError(f"Unsupported deserialization method: {method}")
            
            # Validate schema if requested
            if validate_schema:
                self._validate_model_schema(model, metadata)
            
            # Record performance
            deserialization_time = time.time() - start_time
            self.performance_metrics['deserialization_time'] = deserialization_time
            self.performance_metrics['operations_completed'].append('load')
            
            logger.info(f"Model loaded successfully: {filepath} ({method})")
            
            return model, metadata
            
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise ValueError(f"Model loading failed: {str(e)}")
    
    def _validate_model_for_serialization(self, model: BaseEstimator) -> None:
        """Validate model before serialization."""
        if not hasattr(model, 'fit'):
            raise ValueError("Object is not a valid scikit-learn estimator")
        
        # Check if model is fitted
        try:
            from sklearn.utils.validation import check_is_fitted
            check_is_fitted(model)
        except:
            logger.warning("Model may not be fitted")
        
        # Validate feature compatibility
        if hasattr(model, 'n_features_in_'):
            if model.n_features_in_ != self.expected_feature_count:
                logger.warning(f"Model expects {model.n_features_in_} features, expected {self.expected_feature_count}")
    
    def _extract_model_metadata(self, model: BaseEstimator) -> Dict[str, Any]:
        """Extract metadata from model."""
        metadata = {
            'model_type': type(model).__name__,
            'model_module': type(model).__module__,
            'timestamp': pd.Timestamp.now().isoformat(),
            'feature_count': getattr(model, 'n_features_in_', None),
            'feature_names': getattr(model, 'feature_names_in_', None)
        }
        
        # Add model-specific metadata
        if hasattr(model, 'feature_importances_'):
            metadata['has_feature_importances'] = True
            metadata['feature_importances'] = model.feature_importances_.tolist()
        
        if hasattr(model, 'coef_'):
            metadata['has_coefficients'] = True
            metadata['coefficients_shape'] = model.coef_.shape
        
        # Add hyperparameters
        try:
            metadata['hyperparameters'] = model.get_params()
        except:
            metadata['hyperparameters'] = {}
        
        return metadata
    
    def _save_with_pickle(self, model: BaseEstimator, filepath: Path, metadata: Dict[str, Any]) -> None:
        """Save model using pickle."""
        save_data = {
            'model': model,
            'metadata': metadata,
            'schema_version': '1.0'
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
    
    def _save_with_joblib(self, model: BaseEstimator, filepath: Path, metadata: Dict[str, Any]) -> None:
        """Save model using joblib."""
        save_data = {
            'model': model,
            'metadata': metadata,
            'schema_version': '1.0'
        }
        
        joblib.dump(save_data, filepath)
    
    def _load_with_pickle(self, filepath: Path) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """Load model using pickle."""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        if isinstance(save_data, dict):
            return save_data['model'], save_data.get('metadata', {})
        else:
            # Legacy format - just the model
            return save_data, {}
    
    def _load_with_joblib(self, filepath: Path) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """Load model using joblib."""
        save_data = joblib.load(filepath)
        
        if isinstance(save_data, dict):
            return save_data['model'], save_data.get('metadata', {})
        else:
            # Legacy format - just the model
            return save_data, {}
    
    def _detect_serialization_method(self, filepath: Path) -> str:
        """Auto-detect serialization method from file extension."""
        suffix = filepath.suffix.lower()
        
        if suffix in ['.pkl', '.pickle']:
            return 'pickle'
        elif suffix in ['.joblib', '.jl']:
            return 'joblib'
        else:
            # Try joblib first as it's more robust
            return 'joblib'
    
    def _validate_model_schema(self, model: BaseEstimator, metadata: Dict[str, Any]) -> None:
        """Validate model schema compatibility."""
        # Check feature count
        model_feature_count = getattr(model, 'n_features_in_', None)
        if model_feature_count is not None:
            if model_feature_count != self.expected_feature_count:
                logger.warning(f"Model feature count {model_feature_count} differs from expected {self.expected_feature_count}")
        
        # Check feature names if available
        if self.feature_schema and hasattr(model, 'feature_names_in_'):
            model_features = set(model.feature_names_in_)
            expected_features = set(self.feature_schema)
            
            if model_features != expected_features:
                missing = expected_features - model_features
                extra = model_features - expected_features
                
                if missing:
                    logger.warning(f"Model missing expected features: {missing}")
                if extra:
                    logger.warning(f"Model has unexpected features: {extra}")
    
    def test_serialization_compatibility(self, model: BaseEstimator) -> Dict[str, Any]:
        """
        Test serialization compatibility for a model.
        
        Args:
            model (BaseEstimator): Model to test
        
        Returns:
            Dict[str, Any]: Compatibility test results
        """
        results = {
            'pickle_compatible': False,
            'joblib_compatible': False,
            'feature_importance_preserved': False,
            'prediction_consistency': False,
            'errors': []
        }
        
        try:
            # Create test data
            np.random.seed(42)
            X_test = np.random.randn(10, self.expected_feature_count)
            
            # Get original predictions
            original_pred = model.predict(X_test)
            
            # Test pickle serialization
            try:
                with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
                    self.save_model(model, tmp_file.name, method='pickle')
                    loaded_model, _ = self.load_model(tmp_file.name, method='pickle')
                    
                    # Test prediction consistency
                    loaded_pred = loaded_model.predict(X_test)
                    if np.array_equal(original_pred, loaded_pred):
                        results['pickle_compatible'] = True
                        results['prediction_consistency'] = True
                    
                    Path(tmp_file.name).unlink()  # Clean up
                    
            except Exception as e:
                results['errors'].append(f"Pickle test failed: {str(e)}")
            
            # Test joblib serialization
            try:
                with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_file:
                    self.save_model(model, tmp_file.name, method='joblib')
                    loaded_model, _ = self.load_model(tmp_file.name, method='joblib')
                    
                    # Test prediction consistency
                    loaded_pred = loaded_model.predict(X_test)
                    if np.array_equal(original_pred, loaded_pred):
                        results['joblib_compatible'] = True
                    
                    Path(tmp_file.name).unlink()  # Clean up
                    
            except Exception as e:
                results['errors'].append(f"Joblib test failed: {str(e)}")
            
            # Test feature importance preservation
            if hasattr(model, 'feature_importances_'):
                try:
                    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_file:
                        self.save_model(model, tmp_file.name, method='joblib')
                        loaded_model, _ = self.load_model(tmp_file.name, method='joblib')
                        
                        if hasattr(loaded_model, 'feature_importances_'):
                            if np.allclose(model.feature_importances_, loaded_model.feature_importances_):
                                results['feature_importance_preserved'] = True
                        
                        Path(tmp_file.name).unlink()  # Clean up
                        
                except Exception as e:
                    results['errors'].append(f"Feature importance test failed: {str(e)}")
            
        except Exception as e:
            results['errors'].append(f"General test failed: {str(e)}")
        
        return results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from model operations."""
        return self.performance_metrics.copy()


class ModelFactory:
    """Factory for creating and comparing different model types."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize model factory.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        
        # Available model configurations
        self.model_configs = {
            'RandomForest': {
                'class': RandomForestClassifier,
                'default_params': {'n_estimators': 100, 'random_state': random_state},
                'description': 'Random Forest with feature importance analysis'
            },
            'LogisticRegression': {
                'class': LogisticRegression,
                'default_params': {'random_state': random_state, 'max_iter': 300},
                'description': 'Logistic Regression with coefficient analysis'
            },
            'SVM': {
                'class': SVC,
                'default_params': {'random_state': random_state, 'probability': True},
                'description': 'Support Vector Machine with probability estimates'
            }
        }
    
    def create_model(self, model_type: str, **kwargs) -> BaseEstimator:
        """
        Create a model of the specified type.
        
        Args:
            model_type (str): Type of model to create
            **kwargs: Additional parameters for the model
        
        Returns:
            BaseEstimator: Configured model instance
        """
        if model_type not in self.model_configs:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(self.model_configs.keys())}")
        
        config = self.model_configs[model_type]
        params = config['default_params'].copy()
        params.update(kwargs)
        
        return config['class'](**params)
    
    def create_model_suite(self, custom_params: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, BaseEstimator]:
        """
        Create a suite of models for comparison.
        
        Args:
            custom_params (Dict, optional): Custom parameters for each model type
        
        Returns:
            Dict[str, BaseEstimator]: Dictionary of configured models
        """
        models = {}
        
        for model_type in self.model_configs.keys():
            params = {}
            if custom_params and model_type in custom_params:
                params = custom_params[model_type]
            
            models[model_type] = self.create_model(model_type, **params)
        
        return models
    
    def get_available_models(self) -> Dict[str, str]:
        """Get list of available model types with descriptions."""
        return {name: config['description'] for name, config in self.model_configs.items()}


class ModelSerializer:
    """Specialized model serializer with enhanced compatibility testing."""
    
    def __init__(self):
        """Initialize model serializer."""
        self.manager = ModelManager()
    
    def serialize_model_suite(self, models: Dict[str, BaseEstimator], 
                             output_dir: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
        """
        Serialize a suite of models with compatibility testing.
        
        Args:
            models (Dict[str, BaseEstimator]): Dictionary of models to serialize
            output_dir (Union[str, Path]): Directory to save models
        
        Returns:
            Dict[str, Dict[str, Any]]: Serialization results for each model
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for model_name, model in models.items():
            try:
                # Test compatibility first
                compatibility = self.manager.test_serialization_compatibility(model)
                
                # Save model if compatible
                if compatibility['joblib_compatible']:
                    filepath = output_dir / f"{model_name}.joblib"
                    save_result = self.manager.save_model(model, filepath, method='joblib')
                    
                    results[model_name] = {
                        'serialization_result': save_result,
                        'compatibility_test': compatibility,
                        'status': 'SUCCESS'
                    }
                else:
                    results[model_name] = {
                        'compatibility_test': compatibility,
                        'status': 'FAILED',
                        'reason': 'Serialization compatibility test failed'
                    }
                    
            except Exception as e:
                results[model_name] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
        
        return results
