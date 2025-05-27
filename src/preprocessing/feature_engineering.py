"""
Feature Engineering Module for Banking Marketing Dataset

This module provides feature engineering utilities based on Phase 2 EDA findings
and Phase 3 requirements for preparing features for Phase 4.

Key feature engineering operations:
1. Age group categorization
2. Campaign intensity features
3. Interaction feature preparation
4. Binary flag creation
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering utilities for banking marketing dataset.
    
    Based on Phase 2 EDA insights and Phase 3 requirements:
    - Create meaningful age segments for marketing
    - Engineer campaign intensity features
    - Prepare interaction features
    - Generate binary indicators
    """
    
    def __init__(self):
        """Initialize feature engineering configurations."""
        self.feature_config = {
            'age_bins': [18, 25, 35, 45, 55, 65, 100],
            'age_labels': ['Young Adult', 'Adult', 'Middle Age', 'Pre-Senior', 'Senior', 'Elder'],
            'campaign_bins': [0, 1, 3, 5, 50],
            'campaign_labels': ['No Contact', 'Low', 'Medium', 'High']
        }
        
        self.engineering_stats = {
            'age_groups_created': 0,
            'campaign_intensity_features': 0,
            'binary_flags_created': 0,
            'interaction_features': 0
        }
    
    def create_age_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create meaningful age group categories for marketing analysis.
        
        Business Logic: Age segments based on banking customer lifecycle
        - Young Adult (18-25): Entry-level banking
        - Adult (25-35): Career building
        - Middle Age (35-45): Peak earning
        - Pre-Senior (45-55): Wealth accumulation
        - Senior (55-65): Pre-retirement
        - Elder (65+): Retirement
        
        Args:
            df: DataFrame with numeric Age column
            
        Returns:
            DataFrame with Age_Group feature
        """
        logger.info("Creating age group categories...")
        
        if 'Age' not in df.columns:
            logger.warning("Age column not found")
            return df
        
        df_engineered = df.copy()
        
        # Create age groups using pandas cut
        df_engineered['Age_Group'] = pd.cut(
            df_engineered['Age'],
            bins=self.feature_config['age_bins'],
            labels=self.feature_config['age_labels'],
            include_lowest=True
        )
        
        # Convert to string for consistency
        df_engineered['Age_Group'] = df_engineered['Age_Group'].astype(str)
        
        # Count age group distribution
        age_group_dist = df_engineered['Age_Group'].value_counts()
        self.engineering_stats['age_groups_created'] = len(df_engineered)
        
        logger.info(f"Age groups created for {len(df_engineered)} records:")
        for group, count in age_group_dist.items():
            percentage = (count / len(df_engineered)) * 100
            logger.info(f"  • {group}: {count} ({percentage:.1f}%)")
        
        return df_engineered
    
    def create_campaign_intensity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create campaign intensity features based on contact patterns.
        
        Business Value: Optimize contact frequency for subscription likelihood
        - No Contact (0): Never contacted
        - Low (1-2): Minimal contact
        - Medium (3-4): Moderate contact
        - High (5+): Intensive contact
        
        Args:
            df: DataFrame with Campaign Calls column
            
        Returns:
            DataFrame with campaign intensity features
        """
        logger.info("Creating campaign intensity features...")
        
        if 'Campaign Calls' not in df.columns:
            logger.warning("Campaign Calls column not found")
            return df
        
        df_engineered = df.copy()
        
        # Create campaign intensity categories
        df_engineered['Campaign_Intensity'] = pd.cut(
            df_engineered['Campaign Calls'],
            bins=self.feature_config['campaign_bins'],
            labels=self.feature_config['campaign_labels'],
            include_lowest=True
        )
        
        # Convert to string for consistency
        df_engineered['Campaign_Intensity'] = df_engineered['Campaign_Intensity'].astype(str)
        
        # Create binary flags for high-intensity campaigns
        df_engineered['High_Intensity_Campaign'] = (df_engineered['Campaign Calls'] >= 5).astype(int)
        
        # Create contact recency indicators (if No_Previous_Contact exists)
        if 'No_Previous_Contact' in df_engineered.columns:
            df_engineered['Recent_Contact'] = (1 - df_engineered['No_Previous_Contact']).astype(int)
            self.engineering_stats['binary_flags_created'] += 1
        
        # Count intensity distribution
        intensity_dist = df_engineered['Campaign_Intensity'].value_counts()
        self.engineering_stats['campaign_intensity_features'] = len(df_engineered)
        self.engineering_stats['binary_flags_created'] += 1
        
        logger.info(f"Campaign intensity features created:")
        for intensity, count in intensity_dist.items():
            percentage = (count / len(df_engineered)) * 100
            logger.info(f"  • {intensity}: {count} ({percentage:.1f}%)")
        
        return df_engineered
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features for improved model performance.
        
        Business Logic:
        - Education-Occupation combinations for customer segmentation
        - Loan status interactions for financial profile
        - Contact method-age interactions for channel preferences
        
        Args:
            df: DataFrame with relevant columns
            
        Returns:
            DataFrame with interaction features
        """
        logger.info("Creating interaction features...")
        
        df_engineered = df.copy()
        interaction_count = 0
        
        # Education-Occupation interaction
        if 'Education Level' in df_engineered.columns and 'Occupation' in df_engineered.columns:
            df_engineered['Education_Occupation'] = (
                df_engineered['Education Level'].astype(str) + '_' + 
                df_engineered['Occupation'].astype(str)
            )
            interaction_count += 1
            logger.info("Created Education-Occupation interaction feature")
        
        # Loan status interaction
        if 'Housing Loan' in df_engineered.columns and 'Personal Loan' in df_engineered.columns:
            # Create combined loan status
            housing_loan = df_engineered['Housing Loan'].fillna('unknown')
            personal_loan = df_engineered['Personal Loan'].fillna('unknown')
            
            df_engineered['Loan_Profile'] = housing_loan.astype(str) + '_' + personal_loan.astype(str)
            
            # Create binary flags for loan combinations
            df_engineered['Has_Any_Loan'] = (
                ((housing_loan == 'yes') | (personal_loan == 'yes'))
            ).astype(int)
            
            df_engineered['Has_Both_Loans'] = (
                ((housing_loan == 'yes') & (personal_loan == 'yes'))
            ).astype(int)
            
            interaction_count += 3
            logger.info("Created loan status interaction features")
        
        # Contact method-age group interaction
        if 'Contact Method' in df_engineered.columns and 'Age_Group' in df_engineered.columns:
            df_engineered['Contact_Age_Profile'] = (
                df_engineered['Contact Method'].astype(str) + '_' + 
                df_engineered['Age_Group'].astype(str)
            )
            interaction_count += 1
            logger.info("Created Contact Method-Age Group interaction feature")
        
        self.engineering_stats['interaction_features'] = interaction_count
        logger.info(f"Created {interaction_count} interaction features")
        
        return df_engineered
    
    def create_binary_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary indicator features for categorical variables.
        
        Business Value: Enable better model interpretation and performance
        
        Args:
            df: DataFrame with categorical columns
            
        Returns:
            DataFrame with binary indicator features
        """
        logger.info("Creating binary indicator features...")
        
        df_engineered = df.copy()
        binary_features_created = 0
        
        # Create binary indicators for key categorical features
        categorical_features = {
            'Credit Default': ['yes', 'no'],
            'Housing Loan': ['yes', 'no'],
            'Personal Loan': ['yes', 'no'],
            'Contact Method': ['cellular', 'telephone']
        }
        
        for feature, values in categorical_features.items():
            if feature in df_engineered.columns:
                for value in values:
                    binary_col_name = f"{feature.replace(' ', '_')}_{value}"
                    df_engineered[binary_col_name] = (
                        df_engineered[feature] == value
                    ).astype(int)
                    binary_features_created += 1
        
        # Create marital status indicators
        if 'Marital Status' in df_engineered.columns:
            marital_values = df_engineered['Marital Status'].unique()
            for status in marital_values:
                if status not in ['unknown', 'Information Not Available']:
                    binary_col_name = f"Marital_{status}"
                    df_engineered[binary_col_name] = (
                        df_engineered['Marital Status'] == status
                    ).astype(int)
                    binary_features_created += 1
        
        self.engineering_stats['binary_flags_created'] += binary_features_created
        logger.info(f"Created {binary_features_created} binary indicator features")
        
        return df_engineered
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main feature engineering pipeline.
        
        Orchestrates all feature engineering operations:
        1. Age group categorization
        2. Campaign intensity features
        3. Interaction features
        4. Binary indicators
        
        Args:
            df: Cleaned DataFrame from data cleaning pipeline
            
        Returns:
            DataFrame with engineered features ready for Phase 4
        """
        logger.info("=" * 80)
        logger.info("STARTING FEATURE ENGINEERING PIPELINE")
        logger.info("=" * 80)
        
        initial_columns = len(df.columns)
        logger.info(f"Initial feature count: {initial_columns}")
        
        # Step 1: Create age groups
        df_engineered = self.create_age_groups(df)
        logger.info("✅ Step 1: Age group categorization completed")
        
        # Step 2: Create campaign intensity features
        df_engineered = self.create_campaign_intensity_features(df_engineered)
        logger.info("✅ Step 2: Campaign intensity features completed")
        
        # Step 3: Create interaction features
        df_engineered = self.create_interaction_features(df_engineered)
        logger.info("✅ Step 3: Interaction features completed")
        
        # Step 4: Create binary indicators
        df_engineered = self.create_binary_indicators(df_engineered)
        logger.info("✅ Step 4: Binary indicator features completed")
        
        final_columns = len(df_engineered.columns)
        features_added = final_columns - initial_columns
        
        logger.info(f"Final feature count: {final_columns}")
        logger.info(f"Features added: {features_added}")
        
        # Generate feature engineering summary
        self.generate_feature_summary(initial_columns, final_columns)
        
        logger.info("=" * 80)
        logger.info("FEATURE ENGINEERING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        return df_engineered
    
    def generate_feature_summary(self, initial_features: int, final_features: int) -> Dict[str, Any]:
        """
        Generate feature engineering summary report.
        
        Args:
            initial_features: Initial feature count
            final_features: Final feature count
            
        Returns:
            Dict containing feature engineering summary
        """
        summary = {
            'feature_transformation': {
                'initial_features': initial_features,
                'final_features': final_features,
                'features_added': final_features - initial_features
            },
            'engineering_operations': self.engineering_stats.copy()
        }
        
        logger.info("🔧 FEATURE ENGINEERING SUMMARY:")
        logger.info(f"   • Age groups created: {self.engineering_stats['age_groups_created']} records")
        logger.info(f"   • Campaign intensity features: {self.engineering_stats['campaign_intensity_features']} records")
        logger.info(f"   • Interaction features: {self.engineering_stats['interaction_features']} features")
        logger.info(f"   • Binary flags created: {self.engineering_stats['binary_flags_created']} features")
        logger.info(f"   • Total features added: {final_features - initial_features}")
        
        return summary
