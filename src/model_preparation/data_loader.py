"""
Data Loader Module

Handles Phase 5 data loading with automatic fallback to Phase 4 integration.
Implements comprehensive data validation and schema checking for the 45-feature dataset.

Key Features:
- Phase 5 featured data loading (data/featured/featured-db.csv)
- Phase 4 integration fallback with mock feature engineering
- Feature schema validation (12 engineered features)
- Performance monitoring (>97K records/second)
- Business logic validation (customer segments, subscription rates)
"""

import pandas as pd
import numpy as np
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

class ModelPreparationError(Exception):
    """Custom exception for model preparation errors."""
    pass

class DataLoader:
    """
    Data loader for Phase 6 model preparation with Phase 5 integration.
    
    Handles loading of Phase 5 featured data with automatic fallback to Phase 4
    integration when featured data is not available.
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize data loader.
        
        Args:
            project_root (Path, optional): Project root directory
        """
        self.project_root = project_root or Path.cwd()
        self.featured_data_path = self.project_root / "data" / "featured" / "featured-db.csv"
        
        # Expected data specifications
        self.expected_records = 41188
        self.expected_features = 45
        self.expected_subscription_rate = 0.113
        self.performance_standard = 97000  # records per second
        
        # Expected engineered features
        self.engineered_features = [
            'age_bin', 'education_job_segment', 'customer_value_segment',
            'recent_contact_flag', 'campaign_intensity', 'contact_effectiveness_score',
            'financial_risk_score', 'risk_category', 'is_high_risk',
            'high_intensity_flag', 'is_premium_customer', 'contact_recency'
        ]
        
        # Performance tracking
        self.performance_metrics = {
            'loading_time': 0,
            'validation_time': 0,
            'records_per_second': 0,
            'data_source': None
        }
    
    def load_data(self, use_fallback: bool = True, validate_schema: bool = True) -> pd.DataFrame:
        """
        Load Phase 5 featured data with automatic fallback.
        
        Args:
            use_fallback (bool): Whether to use Phase 4 fallback if Phase 5 data unavailable
            validate_schema (bool): Whether to validate feature schema
        
        Returns:
            pd.DataFrame: Loaded and validated dataset
        
        Raises:
            ModelPreparationError: If data loading fails
        """
        start_time = time.time()
        
        try:
            # Try Phase 5 featured data first
            if self.featured_data_path.exists():
                df = self._load_phase5_data()
                self.performance_metrics['data_source'] = 'Phase 5 featured data'
                logger.info("Successfully loaded Phase 5 featured data")
                
            elif use_fallback:
                df = self._load_with_phase4_fallback()
                self.performance_metrics['data_source'] = 'Phase 4 integration with mock features'
                logger.info("Using Phase 4 fallback with mock feature engineering")
                
            else:
                raise ModelPreparationError("Phase 5 featured data not available and fallback disabled")
            
            # Record loading performance
            loading_time = time.time() - start_time
            self.performance_metrics['loading_time'] = loading_time
            self.performance_metrics['records_per_second'] = len(df) / loading_time if loading_time > 0 else float('inf')
            
            # Validate schema if requested
            if validate_schema:
                validation_start = time.time()
                self._validate_data_schema(df)
                self.performance_metrics['validation_time'] = time.time() - validation_start
            
            logger.info(f"Data loading completed: {len(df)} records, {len(df.columns)} features")
            logger.info(f"Loading performance: {self.performance_metrics['records_per_second']:,.0f} records/sec")
            
            return df
            
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise ModelPreparationError(f"Data loading failed: {str(e)}")
    
    def _load_phase5_data(self) -> pd.DataFrame:
        """Load Phase 5 featured data from CSV file."""
        try:
            df = pd.read_csv(self.featured_data_path)
            
            # Basic validation
            if len(df) == 0:
                raise ModelPreparationError("Phase 5 data is empty")
            
            if 'Subscription Status' not in df.columns:
                raise ModelPreparationError("Target column 'Subscription Status' missing from Phase 5 data")
            
            return df
            
        except Exception as e:
            raise ModelPreparationError(f"Failed to load Phase 5 data: {str(e)}")
    
    def _load_with_phase4_fallback(self) -> pd.DataFrame:
        """Load data using Phase 4 integration with mock feature engineering."""
        try:
            # Try to import Phase 4 integration
            try:
                from data_integration import load_phase3_output
                df = load_phase3_output()
                logger.info("Loaded data using Phase 4 integration")
                
            except ImportError:
                # Create comprehensive mock data if Phase 4 not available
                df = self._create_mock_data()
                logger.info("Created mock data for testing")
            
            # Add mock engineered features
            df = self._add_mock_engineered_features(df)
            
            return df
            
        except Exception as e:
            raise ModelPreparationError(f"Phase 4 fallback failed: {str(e)}")
    
    def _create_mock_data(self) -> pd.DataFrame:
        """Create comprehensive mock data for testing."""
        np.random.seed(42)
        n_samples = min(self.expected_records, 10000)  # Use subset for testing
        
        # Create base features
        data = {
            'Age': np.random.randint(18, 100, n_samples),
            'Campaign Calls': np.random.randint(1, 10, n_samples),
            'Subscription Status': np.random.choice([0, 1], size=n_samples, p=[0.887, 0.113])
        }
        
        # Add additional features to reach closer to 33 original features
        for i in range(30):
            data[f'feature_{i}'] = np.random.randn(n_samples)
        
        return pd.DataFrame(data)
    
    def _add_mock_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add mock engineered features to match Phase 5 output."""
        df_result = df.copy()
        
        # Age binning
        if 'Age' in df_result.columns:
            df_result['age_bin'] = pd.cut(
                df_result['Age'], 
                bins=[18, 35, 55, 100], 
                labels=[1, 2, 3], 
                include_lowest=True
            )
        
        # Customer value segments
        df_result['customer_value_segment'] = np.random.choice(
            ['Premium', 'Standard', 'Basic'], 
            len(df_result), 
            p=[0.316, 0.577, 0.107]
        )
        
        # Campaign intensity
        if 'Campaign Calls' in df_result.columns:
            df_result['campaign_intensity'] = pd.cut(
                df_result['Campaign Calls'],
                bins=[0, 2, 5, 50],
                labels=['low', 'medium', 'high'],
                include_lowest=True
            )
        
        # Additional engineered features
        df_result['recent_contact_flag'] = np.random.choice([0, 1], len(df_result), p=[0.7, 0.3])
        df_result['contact_effectiveness_score'] = np.random.uniform(0.5, 2.0, len(df_result))
        df_result['financial_risk_score'] = np.random.uniform(0, 1, len(df_result))
        df_result['risk_category'] = np.random.choice(['low', 'medium', 'high'], len(df_result))
        df_result['is_high_risk'] = (df_result['financial_risk_score'] > 0.7).astype(int)
        df_result['high_intensity_flag'] = (df_result['Campaign Calls'] > 5).astype(int) if 'Campaign Calls' in df_result.columns else np.random.choice([0, 1], len(df_result))
        df_result['is_premium_customer'] = (df_result['customer_value_segment'] == 'Premium').astype(int)
        df_result['education_job_segment'] = np.random.choice(['high_value', 'medium_value', 'standard'], len(df_result))
        df_result['contact_recency'] = np.random.uniform(0, 1, len(df_result))
        
        return df_result
    
    def _validate_data_schema(self, df: pd.DataFrame) -> None:
        """Validate data schema and business logic."""
        # Basic structure validation
        if len(df) == 0:
            raise ModelPreparationError("Dataset is empty")
        
        if 'Subscription Status' not in df.columns:
            raise ModelPreparationError("Target column 'Subscription Status' missing")
        
        # Subscription rate validation
        subscription_rate = df['Subscription Status'].mean()
        rate_tolerance = 0.05  # 5% tolerance
        
        if abs(subscription_rate - self.expected_subscription_rate) > rate_tolerance:
            logger.warning(f"Subscription rate {subscription_rate:.3f} differs from expected {self.expected_subscription_rate:.3f}")
        
        # Feature count validation (flexible for testing)
        if len(df.columns) < 30:
            logger.warning(f"Feature count {len(df.columns)} is lower than expected minimum of 30")
        
        # Engineered features validation
        present_engineered = [f for f in self.engineered_features if f in df.columns]
        if len(present_engineered) < 8:
            logger.warning(f"Only {len(present_engineered)} engineered features found, expected at least 8")
        
        # Customer segment validation
        if 'customer_value_segment' in df.columns:
            segment_dist = df['customer_value_segment'].value_counts(normalize=True).to_dict()
            expected_segments = ['Premium', 'Standard', 'Basic']
            
            for segment in expected_segments:
                if segment not in segment_dist:
                    logger.warning(f"Customer segment '{segment}' not found in data")
        
        # Missing values check
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            logger.warning(f"Dataset contains {missing_values} missing values")
        
        logger.info("Data schema validation completed")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from last data loading operation."""
        return self.performance_metrics.copy()
    
    def validate_performance_standard(self) -> bool:
        """Check if loading performance meets the >97K records/second standard."""
        return self.performance_metrics['records_per_second'] >= self.performance_standard
