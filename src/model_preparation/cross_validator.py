"""
Cross Validator Module

Implements 5-fold stratified cross-validation with customer segment awareness.
Maintains class balance within each customer segment across all folds.

Key Features:
- 5-fold stratified cross-validation
- Customer segment awareness and balance preservation
- Performance monitoring for >97K records/second standard
- Business logic validation for segment-specific subscription rates
"""

import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Iterator
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)

class CrossValidator:
    """
    Cross-validator with stratified folding and customer segment awareness.
    
    Implements 5-fold stratified cross-validation that maintains both class balance
    and customer segment distributions across all folds.
    """
    
    def __init__(self, n_splits: int = 5, preserve_segments: bool = True, 
                 shuffle: bool = True, random_state: int = 42):
        """
        Initialize cross-validator.
        
        Args:
            n_splits (int): Number of CV folds
            preserve_segments (bool): Whether to preserve segment balance
            shuffle (bool): Whether to shuffle data before splitting
            random_state (int): Random state for reproducibility
        """
        self.n_splits = n_splits
        self.preserve_segments = preserve_segments
        self.shuffle = shuffle
        self.random_state = random_state
        self.performance_standard = 97000  # records per second
        
        # Initialize stratified k-fold
        self.cv = StratifiedKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )
        
        # Performance tracking
        self.performance_metrics = {
            'setup_time': 0,
            'records_per_second': 0,
            'fold_info': []
        }
    
    def split(self, X: pd.DataFrame, y: pd.Series) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/validation indices for cross-validation.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target vector
        
        Yields:
            Tuple[np.ndarray, np.ndarray]: Train and validation indices for each fold
        """
        start_time = time.time()
        
        try:
            # Prepare stratification strategy
            if self.preserve_segments and 'customer_value_segment' in X.columns:
                stratify_key = self._create_segment_stratification_key(X, y)
                
                # Check if segment stratification is possible
                if self._can_stratify_by_segments(stratify_key):
                    yield from self._split_with_segment_awareness(X, y, stratify_key)
                else:
                    logger.warning("Segment stratification not possible, using class-only stratification")
                    yield from self._split_with_class_stratification(X, y)
            else:
                yield from self._split_with_class_stratification(X, y)
            
            # Record performance metrics
            setup_time = time.time() - start_time
            self.performance_metrics['setup_time'] = setup_time
            self.performance_metrics['records_per_second'] = len(X) / setup_time if setup_time > 0 else float('inf')
            
            logger.info(f"Cross-validation setup completed: {self.n_splits} folds")
            logger.info(f"Setup performance: {self.performance_metrics['records_per_second']:,.0f} records/sec")
            
        except Exception as e:
            logger.error(f"Cross-validation setup failed: {str(e)}")
            raise ValueError(f"Cross-validation setup failed: {str(e)}")
    
    def _create_segment_stratification_key(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Create stratification key combining target and customer segments."""
        return y.astype(str) + '_' + X['customer_value_segment'].astype(str)
    
    def _can_stratify_by_segments(self, stratify_key: pd.Series) -> bool:
        """Check if segment stratification is possible."""
        stratify_counts = stratify_key.value_counts()
        min_count = stratify_counts.min()
        
        # Need at least n_splits samples per group for stratification
        return min_count >= self.n_splits
    
    def _split_with_segment_awareness(self, X: pd.DataFrame, y: pd.Series, 
                                    stratify_key: pd.Series) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Split with segment awareness using combined stratification key."""
        try:
            cv_segment = StratifiedKFold(
                n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state
            )
            
            fold_idx = 0
            for train_idx, val_idx in cv_segment.split(X, stratify_key):
                # Validate fold quality
                self._validate_fold_quality(X, y, train_idx, val_idx, fold_idx)
                
                yield train_idx, val_idx
                fold_idx += 1
                
        except ValueError as e:
            logger.warning(f"Segment-aware stratification failed: {str(e)}, falling back to class-only")
            yield from self._split_with_class_stratification(X, y)
    
    def _split_with_class_stratification(self, X: pd.DataFrame, y: pd.Series) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Split with class-only stratification."""
        fold_idx = 0
        for train_idx, val_idx in self.cv.split(X, y):
            # Validate fold quality
            self._validate_fold_quality(X, y, train_idx, val_idx, fold_idx)
            
            yield train_idx, val_idx
            fold_idx += 1
    
    def _validate_fold_quality(self, X: pd.DataFrame, y: pd.Series, 
                              train_idx: np.ndarray, val_idx: np.ndarray, fold_idx: int) -> None:
        """Validate quality of a single fold."""
        # Calculate subscription rates
        train_rate = y.iloc[train_idx].mean()
        val_rate = y.iloc[val_idx].mean()
        rate_diff = abs(train_rate - val_rate)
        
        fold_info = {
            'fold': fold_idx + 1,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'train_subscription_rate': train_rate,
            'val_subscription_rate': val_rate,
            'rate_difference': rate_diff
        }
        
        # Validate segment balance if applicable
        if 'customer_value_segment' in X.columns:
            segment_balance = self._validate_segment_balance(X, train_idx, val_idx)
            fold_info['segment_balance'] = segment_balance
        
        self.performance_metrics['fold_info'].append(fold_info)
        
        # Log warnings for poor balance
        if rate_diff > 0.05:  # 5% tolerance
            logger.warning(f"Fold {fold_idx + 1}: Large subscription rate difference {rate_diff:.3f}")
    
    def _validate_segment_balance(self, X: pd.DataFrame, train_idx: np.ndarray, 
                                val_idx: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Validate customer segment balance within a fold."""
        segment_balance = {}
        
        train_segments = X.iloc[train_idx]['customer_value_segment'].value_counts(normalize=True).to_dict()
        val_segments = X.iloc[val_idx]['customer_value_segment'].value_counts(normalize=True).to_dict()
        
        for segment in ['Premium', 'Standard', 'Basic']:
            train_rate = train_segments.get(segment, 0)
            val_rate = val_segments.get(segment, 0)
            
            segment_balance[segment] = {
                'train_rate': train_rate,
                'val_rate': val_rate,
                'difference': abs(train_rate - val_rate)
            }
        
        return segment_balance
    
    def cross_validate(self, estimator: BaseEstimator, X: pd.DataFrame, y: pd.Series,
                      scoring: str = 'roc_auc') -> Dict[str, Any]:
        """
        Perform cross-validation with the given estimator.
        
        Args:
            estimator (BaseEstimator): Scikit-learn estimator
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target vector
            scoring (str): Scoring metric
        
        Returns:
            Dict[str, Any]: Cross-validation results
        """
        try:
            # Perform cross-validation
            cv_scores = cross_val_score(estimator, X, y, cv=self.cv, scoring=scoring)
            
            results = {
                'scores': cv_scores.tolist(),
                'mean_score': cv_scores.mean(),
                'std_score': cv_scores.std(),
                'scoring_metric': scoring,
                'n_splits': self.n_splits
            }
            
            logger.info(f"Cross-validation completed: {scoring} = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Cross-validation failed: {str(e)}")
            raise ValueError(f"Cross-validation failed: {str(e)}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from cross-validation setup."""
        return self.performance_metrics.copy()
    
    def validate_performance_standard(self) -> bool:
        """Check if CV setup performance meets the >97K records/second standard."""
        return self.performance_metrics['records_per_second'] >= self.performance_standard
    
    def get_fold_summary(self) -> Dict[str, Any]:
        """Get summary of fold quality metrics."""
        if not self.performance_metrics['fold_info']:
            return {}
        
        fold_info = self.performance_metrics['fold_info']
        
        summary = {
            'n_folds': len(fold_info),
            'avg_train_size': np.mean([f['train_size'] for f in fold_info]),
            'avg_val_size': np.mean([f['val_size'] for f in fold_info]),
            'max_rate_difference': max([f['rate_difference'] for f in fold_info]),
            'avg_rate_difference': np.mean([f['rate_difference'] for f in fold_info])
        }
        
        # Add segment balance summary if available
        if 'segment_balance' in fold_info[0]:
            segment_diffs = []
            for fold in fold_info:
                for segment, balance in fold['segment_balance'].items():
                    segment_diffs.append(balance['difference'])
            
            summary['max_segment_difference'] = max(segment_diffs) if segment_diffs else 0
            summary['avg_segment_difference'] = np.mean(segment_diffs) if segment_diffs else 0
        
        return summary


class SegmentAwareCrossValidator(CrossValidator):
    """Specialized cross-validator with enhanced segment awareness."""
    
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        """
        Initialize segment-aware cross-validator.
        
        Args:
            n_splits (int): Number of CV folds
            shuffle (bool): Whether to shuffle data before splitting
            random_state (int): Random state for reproducibility
        """
        super().__init__(
            n_splits=n_splits, preserve_segments=True, 
            shuffle=shuffle, random_state=random_state
        )
    
    def validate_segment_consistency(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Validate segment consistency across all folds.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target vector
        
        Returns:
            Dict[str, Any]: Segment consistency validation results
        """
        if 'customer_value_segment' not in X.columns:
            return {'error': 'customer_value_segment column not found'}
        
        segment_consistency = {
            'segments_validated': ['Premium', 'Standard', 'Basic'],
            'fold_consistency': [],
            'overall_quality': 'PASSED'
        }
        
        fold_idx = 0
        for train_idx, val_idx in self.split(X, y):
            fold_segments = self._validate_segment_balance(X, train_idx, val_idx)
            
            fold_quality = 'PASSED'
            for segment, balance in fold_segments.items():
                if balance['difference'] > 0.1:  # 10% tolerance
                    fold_quality = 'WARNING'
                    segment_consistency['overall_quality'] = 'WARNING'
            
            segment_consistency['fold_consistency'].append({
                'fold': fold_idx + 1,
                'quality': fold_quality,
                'segment_balance': fold_segments
            })
            
            fold_idx += 1
        
        return segment_consistency
