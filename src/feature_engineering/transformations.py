"""
Feature Transformation Module

This module provides feature transformation utilities for scaling, encoding,
and dimensionality reduction with clear business purpose for each transformation.

Transformations:
1. Scaling: Standardization for model performance
2. Encoding: One-hot encoding for categorical variables
3. Dimensionality: PCA if needed for computational efficiency
4. Memory Optimization: Efficient data types for large datasets

All transformations maintain data integrity and include performance monitoring.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
import warnings

logger = logging.getLogger(__name__)


class FeatureTransformer:
    """
    Feature transformation utilities with business purpose documentation.
    
    This class provides production-ready feature transformations while maintaining
    clear business rationale and performance optimization.
    """
    
    def __init__(self):
        """Initialize feature transformer with transformation configurations."""
        self.transformation_config = {
            'scaling_method': 'standard',  # standard, minmax, robust
            'encoding_method': 'onehot',   # onehot, label, target
            'pca_variance_threshold': 0.95,  # Retain 95% of variance
            'memory_optimization': True,
            'categorical_threshold': 10  # Max categories for one-hot encoding
        }
        
        # Store fitted transformers for consistency
        self.fitted_transformers = {
            'scaler': None,
            'encoders': {},
            'pca': None
        }
        
        # Track transformation statistics
        self.transformation_stats = {
            'features_scaled': 0,
            'features_encoded': 0,
            'features_reduced': 0,
            'memory_optimized': False,
            'original_memory_mb': 0,
            'optimized_memory_mb': 0
        }
    
    def optimize_memory_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize memory usage by converting to efficient data types.
        
        Business Purpose: Handle large datasets efficiently for production deployment
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with optimized data types
        """
        logger.info("Optimizing memory usage...")
        
        df_optimized = df.copy()
        
        # Track original memory usage
        original_memory = df_optimized.memory_usage(deep=True).sum() / 1024**2
        self.transformation_stats['original_memory_mb'] = original_memory
        
        # Optimize integer columns
        for col in df_optimized.select_dtypes(include=['int64']).columns:
            col_min = df_optimized[col].min()
            col_max = df_optimized[col].max()
            
            if col_min >= 0:  # Unsigned integers
                if col_max < 255:
                    df_optimized[col] = df_optimized[col].astype('uint8')
                elif col_max < 65535:
                    df_optimized[col] = df_optimized[col].astype('uint16')
                elif col_max < 4294967295:
                    df_optimized[col] = df_optimized[col].astype('uint32')
            else:  # Signed integers
                if col_min > -128 and col_max < 127:
                    df_optimized[col] = df_optimized[col].astype('int8')
                elif col_min > -32768 and col_max < 32767:
                    df_optimized[col] = df_optimized[col].astype('int16')
                elif col_min > -2147483648 and col_max < 2147483647:
                    df_optimized[col] = df_optimized[col].astype('int32')
        
        # Optimize float columns
        for col in df_optimized.select_dtypes(include=['float64']).columns:
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
        
        # Optimize categorical columns
        for col in df_optimized.select_dtypes(include=['object']).columns:
            num_unique = df_optimized[col].nunique()
            if num_unique < len(df_optimized) * 0.5:  # Less than 50% unique values
                df_optimized[col] = df_optimized[col].astype('category')
        
        # Track optimized memory usage
        optimized_memory = df_optimized.memory_usage(deep=True).sum() / 1024**2
        self.transformation_stats['optimized_memory_mb'] = optimized_memory
        self.transformation_stats['memory_optimized'] = True
        
        memory_reduction = ((original_memory - optimized_memory) / original_memory) * 100
        logger.info(f"Memory optimization: {original_memory:.1f}MB → {optimized_memory:.1f}MB ({memory_reduction:.1f}% reduction)")
        
        return df_optimized
    
    def scale_numerical_features(self, df: pd.DataFrame, 
                                numerical_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Scale numerical features for model performance.
        
        Business Purpose: Ensure all numerical features contribute equally to model training
        
        Args:
            df: Input DataFrame
            numerical_columns: List of columns to scale (auto-detect if None)
            
        Returns:
            DataFrame with scaled numerical features
        """
        logger.info("Scaling numerical features...")
        
        df_scaled = df.copy()
        
        # Auto-detect numerical columns if not provided
        if numerical_columns is None:
            numerical_columns = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude binary flags and categorical encoded features
            numerical_columns = [col for col in numerical_columns 
                               if not col.endswith('_flag') and 
                                  not col.endswith('_bin') and
                                  df_scaled[col].nunique() > 10]
        
        if not numerical_columns:
            logger.info("No numerical columns found for scaling")
            return df_scaled
        
        # Initialize scaler if not fitted
        if self.fitted_transformers['scaler'] is None:
            self.fitted_transformers['scaler'] = StandardScaler()
            
            # Fit and transform
            df_scaled[numerical_columns] = self.fitted_transformers['scaler'].fit_transform(
                df_scaled[numerical_columns]
            )
            logger.info(f"Fitted and transformed {len(numerical_columns)} numerical features")
        else:
            # Transform using fitted scaler
            df_scaled[numerical_columns] = self.fitted_transformers['scaler'].transform(
                df_scaled[numerical_columns]
            )
            logger.info(f"Transformed {len(numerical_columns)} numerical features using fitted scaler")
        
        self.transformation_stats['features_scaled'] = len(numerical_columns)
        
        # Log scaling statistics
        for col in numerical_columns:
            mean_val = df_scaled[col].mean()
            std_val = df_scaled[col].std()
            logger.info(f"  • {col}: mean={mean_val:.3f}, std={std_val:.3f}")
        
        return df_scaled
    
    def encode_categorical_features(self, df: pd.DataFrame,
                                  categorical_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Encode categorical features for model compatibility.
        
        Business Purpose: Convert categorical business segments into model-ready format
        
        Args:
            df: Input DataFrame
            categorical_columns: List of columns to encode (auto-detect if None)
            
        Returns:
            DataFrame with encoded categorical features
        """
        logger.info("Encoding categorical features...")
        
        df_encoded = df.copy()
        
        # Auto-detect categorical columns if not provided
        if categorical_columns is None:
            categorical_columns = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
            # Exclude already processed features
            categorical_columns = [col for col in categorical_columns 
                                 if not col.endswith('_segment') or 
                                    col in ['customer_value_segment', 'campaign_intensity']]
        
        if not categorical_columns:
            logger.info("No categorical columns found for encoding")
            return df_encoded
        
        encoded_features = []
        
        for col in categorical_columns:
            unique_values = df_encoded[col].nunique()
            
            if unique_values <= self.transformation_config['categorical_threshold']:
                # One-hot encoding for low cardinality
                if col not in self.fitted_transformers['encoders']:
                    # Create dummy variables
                    dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                    self.fitted_transformers['encoders'][col] = {
                        'type': 'onehot',
                        'columns': dummies.columns.tolist()
                    }
                else:
                    # Use fitted encoder columns
                    encoder_info = self.fitted_transformers['encoders'][col]
                    dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                    
                    # Ensure consistent columns
                    for encoded_col in encoder_info['columns']:
                        if encoded_col not in dummies.columns:
                            dummies[encoded_col] = 0
                    
                    dummies = dummies[encoder_info['columns']]
                
                # Add dummy variables to dataframe
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                encoded_features.extend(dummies.columns.tolist())
                
                logger.info(f"  • {col}: One-hot encoded into {len(dummies.columns)} features")
                
            else:
                # Label encoding for high cardinality
                if col not in self.fitted_transformers['encoders']:
                    encoder = LabelEncoder()
                    df_encoded[f"{col}_encoded"] = encoder.fit_transform(df_encoded[col].astype(str))
                    self.fitted_transformers['encoders'][col] = {
                        'type': 'label',
                        'encoder': encoder
                    }
                else:
                    encoder = self.fitted_transformers['encoders'][col]['encoder']
                    # Handle unseen categories
                    try:
                        df_encoded[f"{col}_encoded"] = encoder.transform(df_encoded[col].astype(str))
                    except ValueError:
                        # Handle unseen categories by assigning a default value
                        df_encoded[f"{col}_encoded"] = df_encoded[col].astype(str).apply(
                            lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                        )
                
                encoded_features.append(f"{col}_encoded")
                logger.info(f"  • {col}: Label encoded into {col}_encoded")
        
        self.transformation_stats['features_encoded'] = len(encoded_features)
        
        return df_encoded
    
    def apply_dimensionality_reduction(self, df: pd.DataFrame,
                                     feature_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Apply PCA for dimensionality reduction if needed.
        
        Business Purpose: Reduce computational complexity while preserving information
        
        Args:
            df: Input DataFrame
            feature_columns: Columns to apply PCA to (auto-detect if None)
            
        Returns:
            DataFrame with PCA components if reduction was beneficial
        """
        logger.info("Evaluating dimensionality reduction...")
        
        df_reduced = df.copy()
        
        # Auto-detect numerical feature columns
        if feature_columns is None:
            feature_columns = df_reduced.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude target and ID columns
            feature_columns = [col for col in feature_columns 
                             if 'subscription' not in col.lower() and 
                                'id' not in col.lower()]
        
        if len(feature_columns) < 10:
            logger.info(f"Only {len(feature_columns)} features, skipping PCA")
            return df_reduced
        
        # Apply PCA if beneficial
        if self.fitted_transformers['pca'] is None:
            # Fit PCA
            pca = PCA(n_components=self.transformation_config['pca_variance_threshold'])
            
            # Ensure no missing values
            feature_data = df_reduced[feature_columns].fillna(0)
            
            pca_components = pca.fit_transform(feature_data)
            
            # Check if PCA provides significant reduction
            n_components = pca.n_components_
            original_features = len(feature_columns)
            
            if n_components < original_features * 0.8:  # At least 20% reduction
                self.fitted_transformers['pca'] = pca
                
                # Add PCA components
                pca_columns = [f'pca_component_{i+1}' for i in range(n_components)]
                pca_df = pd.DataFrame(pca_components, columns=pca_columns, index=df_reduced.index)
                df_reduced = pd.concat([df_reduced, pca_df], axis=1)
                
                self.transformation_stats['features_reduced'] = n_components
                
                logger.info(f"PCA applied: {original_features} → {n_components} components "
                           f"({pca.explained_variance_ratio_.sum():.1%} variance retained)")
            else:
                logger.info("PCA not beneficial, keeping original features")
        
        return df_reduced
    
    def get_transformation_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive transformation report.
        
        Returns:
            Dict containing transformation statistics and fitted transformers info
        """
        return {
            'transformation_stats': self.transformation_stats.copy(),
            'transformation_config': self.transformation_config.copy(),
            'fitted_transformers_info': {
                'scaler_fitted': self.fitted_transformers['scaler'] is not None,
                'encoders_fitted': len(self.fitted_transformers['encoders']),
                'pca_fitted': self.fitted_transformers['pca'] is not None
            }
        }
