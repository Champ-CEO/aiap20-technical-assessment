"""
Data Splitter Module

Implements stratified data splitting with customer segment awareness.
Preserves both class distribution (11.3% subscription rate) and customer segment 
distributions (Premium: 31.6%, Standard: 57.7%, Basic: 10.7%).

Key Features:
- Stratified splitting preserving class and segment distributions
- Train/validation/test splits with configurable proportions
- Performance monitoring for >97K records/second standard
- Business logic validation for customer segments
"""

import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class DataSplitter:
    """
    Data splitter with stratified splitting and customer segment awareness.
    
    Implements stratified data splitting that preserves both target class distribution
    and customer segment distributions across train/validation/test splits.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize data splitter.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.performance_standard = 97000  # records per second
        
        # Performance tracking
        self.performance_metrics = {
            'splitting_time': 0,
            'records_per_second': 0,
            'split_info': {}
        }
    
    def split_data(self, df: pd.DataFrame, target_column: str = 'Subscription Status',
                   test_size: float = 0.2, validation_size: float = 0.2,
                   preserve_segments: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Split data with stratification preserving class and segment distributions.
        
        Args:
            df (pd.DataFrame): Input dataset
            target_column (str): Name of target column
            test_size (float): Proportion for test set
            validation_size (float): Proportion for validation set
            preserve_segments (bool): Whether to preserve customer segment distributions
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary with train/validation/test splits
        
        Raises:
            ValueError: If splitting parameters are invalid
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            self._validate_split_inputs(df, target_column, test_size, validation_size)
            
            # Prepare stratification strategy
            if preserve_segments and 'customer_value_segment' in df.columns:
                splits = self._split_with_segment_preservation(
                    df, target_column, test_size, validation_size
                )
            else:
                splits = self._split_with_class_stratification(
                    df, target_column, test_size, validation_size
                )
            
            # Record performance metrics
            splitting_time = time.time() - start_time
            self.performance_metrics['splitting_time'] = splitting_time
            self.performance_metrics['records_per_second'] = len(df) / splitting_time if splitting_time > 0 else float('inf')
            
            # Validate split quality
            self._validate_split_quality(splits, df, target_column)
            
            # Record split information
            self._record_split_info(splits, df)
            
            logger.info(f"Data splitting completed: {len(splits)} splits created")
            logger.info(f"Splitting performance: {self.performance_metrics['records_per_second']:,.0f} records/sec")
            
            return splits
            
        except Exception as e:
            logger.error(f"Data splitting failed: {str(e)}")
            raise ValueError(f"Data splitting failed: {str(e)}")
    
    def _validate_split_inputs(self, df: pd.DataFrame, target_column: str,
                              test_size: float, validation_size: float) -> None:
        """Validate splitting inputs."""
        if len(df) == 0:
            raise ValueError("Dataset is empty")
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        if test_size + validation_size >= 1.0:
            raise ValueError("Combined test_size and validation_size must be < 1.0")
        
        if test_size <= 0 or validation_size < 0:
            raise ValueError("test_size must be > 0 and validation_size must be >= 0")
    
    def _split_with_segment_preservation(self, df: pd.DataFrame, target_column: str,
                                       test_size: float, validation_size: float) -> Dict[str, pd.DataFrame]:
        """Split data preserving both class and segment distributions."""
        try:
            # Create combined stratification key
            stratify_key = (
                df[target_column].astype(str) + '_' + 
                df['customer_value_segment'].astype(str)
            )
            
            # Check if stratification is possible
            stratify_counts = stratify_key.value_counts()
            min_count = stratify_counts.min()
            
            if min_count < 2:
                logger.warning("Some stratification groups have < 2 samples, falling back to class-only stratification")
                return self._split_with_class_stratification(df, target_column, test_size, validation_size)
            
            # Perform stratified splits
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # First split: separate test set
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, stratify=stratify_key, random_state=self.random_state
            )
            
            # Second split: separate train and validation
            if validation_size > 0:
                # Adjust validation size for remaining data
                val_size_adjusted = validation_size / (1 - test_size)
                
                # Create stratification key for remaining data
                temp_df = pd.concat([X_temp, y_temp], axis=1)
                temp_stratify_key = (
                    temp_df[target_column].astype(str) + '_' + 
                    temp_df['customer_value_segment'].astype(str)
                )
                
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=val_size_adjusted, 
                    stratify=temp_stratify_key, random_state=self.random_state
                )
                
                # Combine features and target
                splits = {
                    'train': pd.concat([X_train, y_train], axis=1),
                    'validation': pd.concat([X_val, y_val], axis=1),
                    'test': pd.concat([X_test, y_test], axis=1)
                }
            else:
                # Only train/test split
                splits = {
                    'train': pd.concat([X_temp, y_temp], axis=1),
                    'test': pd.concat([X_test, y_test], axis=1)
                }
            
            return splits
            
        except ValueError as e:
            logger.warning(f"Segment-aware stratification failed: {str(e)}, falling back to class-only")
            return self._split_with_class_stratification(df, target_column, test_size, validation_size)
    
    def _split_with_class_stratification(self, df: pd.DataFrame, target_column: str,
                                       test_size: float, validation_size: float) -> Dict[str, pd.DataFrame]:
        """Split data with class-only stratification."""
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )
        
        # Second split: separate train and validation
        if validation_size > 0:
            val_size_adjusted = validation_size / (1 - test_size)
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, 
                stratify=y_temp, random_state=self.random_state
            )
            
            splits = {
                'train': pd.concat([X_train, y_train], axis=1),
                'validation': pd.concat([X_val, y_val], axis=1),
                'test': pd.concat([X_test, y_test], axis=1)
            }
        else:
            splits = {
                'train': pd.concat([X_temp, y_temp], axis=1),
                'test': pd.concat([X_test, y_test], axis=1)
            }
        
        return splits
    
    def _validate_split_quality(self, splits: Dict[str, pd.DataFrame], 
                               original_df: pd.DataFrame, target_column: str) -> None:
        """Validate quality of data splits."""
        original_rate = original_df[target_column].mean()
        rate_tolerance = 0.02  # 2% tolerance
        
        for split_name, split_df in splits.items():
            split_rate = split_df[target_column].mean()
            rate_diff = abs(split_rate - original_rate)
            
            if rate_diff > rate_tolerance:
                logger.warning(f"{split_name} split rate {split_rate:.3f} differs from original {original_rate:.3f}")
        
        # Validate segment preservation if applicable
        if 'customer_value_segment' in original_df.columns:
            self._validate_segment_preservation(splits, original_df)
    
    def _validate_segment_preservation(self, splits: Dict[str, pd.DataFrame], 
                                     original_df: pd.DataFrame) -> None:
        """Validate customer segment preservation across splits."""
        original_segment_dist = original_df['customer_value_segment'].value_counts(normalize=True).to_dict()
        segment_tolerance = 0.05  # 5% tolerance
        
        for split_name, split_df in splits.items():
            if 'customer_value_segment' in split_df.columns:
                split_segment_dist = split_df['customer_value_segment'].value_counts(normalize=True).to_dict()
                
                for segment in ['Premium', 'Standard', 'Basic']:
                    if segment in original_segment_dist and segment in split_segment_dist:
                        original_rate = original_segment_dist[segment]
                        split_rate = split_segment_dist[segment]
                        rate_diff = abs(split_rate - original_rate)
                        
                        if rate_diff > segment_tolerance:
                            logger.warning(f"{split_name} {segment} segment rate {split_rate:.3f} differs from original {original_rate:.3f}")
    
    def _record_split_info(self, splits: Dict[str, pd.DataFrame], original_df: pd.DataFrame) -> None:
        """Record information about the splits."""
        total_records = len(original_df)
        
        split_info = {
            'total_records': total_records,
            'splits': {}
        }
        
        for split_name, split_df in splits.items():
            split_info['splits'][split_name] = {
                'records': len(split_df),
                'proportion': len(split_df) / total_records,
                'subscription_rate': split_df['Subscription Status'].mean() if 'Subscription Status' in split_df.columns else None
            }
        
        self.performance_metrics['split_info'] = split_info
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from last splitting operation."""
        return self.performance_metrics.copy()
    
    def validate_performance_standard(self) -> bool:
        """Check if splitting performance meets the >97K records/second standard."""
        return self.performance_metrics['records_per_second'] >= self.performance_standard


class StratifiedSplitter(DataSplitter):
    """Specialized stratified splitter with enhanced segment awareness."""
    
    def __init__(self, preserve_segments: bool = True, random_state: int = 42):
        """
        Initialize stratified splitter.
        
        Args:
            preserve_segments (bool): Whether to always preserve customer segments
            random_state (int): Random state for reproducibility
        """
        super().__init__(random_state=random_state)
        self.preserve_segments = preserve_segments
    
    def split_data(self, df: pd.DataFrame, target_column: str = 'Subscription Status',
                   test_size: float = 0.2, validation_size: float = 0.2) -> Dict[str, pd.DataFrame]:
        """Split data with enhanced stratification."""
        return super().split_data(
            df, target_column=target_column, test_size=test_size,
            validation_size=validation_size, preserve_segments=self.preserve_segments
        )
