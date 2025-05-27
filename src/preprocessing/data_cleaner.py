"""
Data Cleaning Module for Banking Marketing Dataset

This module provides comprehensive data cleaning functionality based on
Phase 2 EDA findings and Phase 3 requirements from TASKS.md.

Key cleaning operations:
1. Age conversion from text to numeric
2. Missing values handling (28,935 total)
3. Special values cleaning (12,008 total)
4. Contact method standardization
5. Target variable binary encoding
6. Data validation and quality assurance
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BankingDataCleaner:
    """
    Comprehensive data cleaner for banking marketing dataset.

    Based on Phase 2 EDA findings:
    - 28,935 missing values requiring attention
    - 12,008 special values requiring cleaning
    - Age stored as text format
    - Inconsistent contact method values
    - Target variable needs binary encoding
    """

    def __init__(self):
        """Initialize the data cleaner with cleaning configurations."""
        self.cleaning_config = {
            "age_min": 18,
            "age_max": 100,
            "campaign_calls_max": 50,
            "target_encoding": {"yes": 1, "no": 0},
            "contact_method_mapping": {
                "Cell": "cellular",
                "cellular": "cellular",
                "Telephone": "telephone",
                "telephone": "telephone",
            },
        }

        self.cleaning_stats = {
            "age_conversions": 0,
            "missing_values_handled": 0,
            "special_values_cleaned": 0,
            "contact_methods_standardized": 0,
            "target_variables_encoded": 0,
            "outliers_capped": 0,
        }

        self.data_quality_metrics = {
            "initial_missing_count": 0,
            "final_missing_count": 0,
            "initial_special_values": 0,
            "final_special_values": 0,
        }

    def clean_age_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert age from text format to numeric.

        EDA Finding: Age stored as text ('57 years', '55 years', etc.)
        Business Rule: Valid age range 18-100 years

        Args:
            df: DataFrame with Age column in text format

        Returns:
            DataFrame with numeric Age column
        """
        logger.info("Starting age column cleaning...")

        if "Age" not in df.columns:
            logger.warning("Age column not found in DataFrame")
            return df

        df_cleaned = df.copy()
        original_age_dtype = df_cleaned["Age"].dtype

        # Extract numeric values from text format
        if df_cleaned["Age"].dtype == "object":
            # Use regex to extract numbers from text like '57 years'
            df_cleaned["Age"] = df_cleaned["Age"].astype(str).str.extract(r"(\d+)")[0]
            df_cleaned["Age"] = pd.to_numeric(df_cleaned["Age"], errors="coerce")
            self.cleaning_stats["age_conversions"] = len(df_cleaned)
            logger.info(
                f"Converted {self.cleaning_stats['age_conversions']} age values from text to numeric"
            )

        # Handle outliers and impossible values
        # EDA found ages like 150 years - these are data quality issues
        outliers_mask = (df_cleaned["Age"] > self.cleaning_config["age_max"]) | (
            df_cleaned["Age"] < self.cleaning_config["age_min"]
        )
        outliers_count = outliers_mask.sum()

        if outliers_count > 0:
            logger.warning(
                f"Found {outliers_count} age outliers outside {self.cleaning_config['age_min']}-{self.cleaning_config['age_max']} range"
            )

            # Cap extreme values
            df_cleaned.loc[
                df_cleaned["Age"] > self.cleaning_config["age_max"], "Age"
            ] = self.cleaning_config["age_max"]
            df_cleaned.loc[
                df_cleaned["Age"] < self.cleaning_config["age_min"], "Age"
            ] = self.cleaning_config["age_min"]
            self.cleaning_stats["outliers_capped"] = outliers_count

        logger.info(
            f"Age column cleaning completed. Data type: {original_age_dtype} -> {df_cleaned['Age'].dtype}"
        )
        return df_cleaned

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values based on EDA findings and business logic.

        EDA Findings:
        - Housing Loan: 24,789 missing (60.2%)
        - Personal Loan: 4,146 missing (10.1%)
        - Total: 28,935 missing values

        Strategy: Create 'Information Not Available' category for business analysis

        Args:
            df: DataFrame with missing values

        Returns:
            DataFrame with missing values handled
        """
        logger.info("Starting missing values handling...")

        df_cleaned = df.copy()
        initial_missing = df_cleaned.isnull().sum().sum()
        self.data_quality_metrics["initial_missing_count"] = initial_missing

        # Handle categorical missing values
        categorical_columns = [
            "Housing Loan",
            "Personal Loan",
            "Occupation",
            "Marital Status",
            "Education Level",
            "Credit Default",
            "Contact Method",
        ]

        for col in categorical_columns:
            if col in df_cleaned.columns:
                missing_count = df_cleaned[col].isnull().sum()
                if missing_count > 0:
                    # Replace missing with 'Information Not Available' for business analysis
                    df_cleaned[col] = df_cleaned[col].fillna(
                        "Information Not Available"
                    )
                    logger.info(
                        f"Filled {missing_count} missing values in '{col}' with 'Information Not Available'"
                    )
                    self.cleaning_stats["missing_values_handled"] += missing_count

        # Handle numerical missing values
        numerical_columns = ["Age", "Campaign Calls", "Previous Contact Days"]

        for col in numerical_columns:
            if col in df_cleaned.columns:
                missing_count = df_cleaned[col].isnull().sum()
                if missing_count > 0:
                    if col == "Age":
                        # Use median for age
                        median_age = df_cleaned[col].median()
                        df_cleaned[col] = df_cleaned[col].fillna(median_age)
                        logger.info(
                            f"Filled {missing_count} missing ages with median: {median_age}"
                        )
                    elif col == "Campaign Calls":
                        # Use 1 (minimum realistic campaign calls)
                        df_cleaned[col] = df_cleaned[col].fillna(1)
                        logger.info(
                            f"Filled {missing_count} missing campaign calls with 1"
                        )
                    elif col == "Previous Contact Days":
                        # Use 999 (no previous contact indicator)
                        df_cleaned[col] = df_cleaned[col].fillna(999)
                        logger.info(
                            f"Filled {missing_count} missing previous contact days with 999"
                        )

                    self.cleaning_stats["missing_values_handled"] += missing_count

        final_missing = df_cleaned.isnull().sum().sum()
        self.data_quality_metrics["final_missing_count"] = final_missing

        logger.info(
            f"Missing values handling completed. {initial_missing} -> {final_missing} missing values"
        )
        return df_cleaned

    def clean_special_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean special values based on EDA findings.

        EDA Findings:
        - Credit Default: 8,597 unknown values (20.9%)
        - Education Level: 1,731 unknown values (4.2%)
        - Personal Loan: 877 unknown values (2.1%)
        - Previous Contact Days: 39,673 rows with '999' (96.3%)

        Strategy: Retain 'unknown' as distinct business category

        Args:
            df: DataFrame with special values

        Returns:
            DataFrame with special values cleaned
        """
        logger.info("Starting special values cleaning...")

        df_cleaned = df.copy()

        # Count initial special values
        initial_special = 0
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == "object":
                initial_special += (df_cleaned[col] == "unknown").sum()

        if "Previous Contact Days" in df_cleaned.columns:
            initial_special += (df_cleaned["Previous Contact Days"] == 999).sum()

        self.data_quality_metrics["initial_special_values"] = initial_special

        # Handle 'unknown' values - keep as distinct category for business analysis
        categorical_columns = [
            "Occupation",
            "Marital Status",
            "Education Level",
            "Credit Default",
            "Housing Loan",
            "Personal Loan",
            "Contact Method",
        ]

        for col in categorical_columns:
            if col in df_cleaned.columns:
                unknown_count = (df_cleaned[col] == "unknown").sum()
                if unknown_count > 0:
                    # Keep 'unknown' as is - it represents real business information state
                    logger.info(
                        f"Retained {unknown_count} 'unknown' values in '{col}' as business category"
                    )
                    self.cleaning_stats["special_values_cleaned"] += unknown_count

        # Handle Previous Contact Days special value (999)
        if "Previous Contact Days" in df_cleaned.columns:
            special_999_count = (df_cleaned["Previous Contact Days"] == 999).sum()
            if special_999_count > 0:
                # Create binary flag for 'No Previous Contact'
                df_cleaned["No_Previous_Contact"] = (
                    df_cleaned["Previous Contact Days"] == 999
                ).astype(int)
                logger.info(
                    f"Created 'No_Previous_Contact' flag for {special_999_count} records with 999 days"
                )
                self.cleaning_stats["special_values_cleaned"] += special_999_count

        # Count final special values
        final_special = 0
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == "object":
                final_special += (df_cleaned[col] == "unknown").sum()

        self.data_quality_metrics["final_special_values"] = final_special

        logger.info(
            f"Special values cleaning completed. Processed {initial_special} special values"
        )
        return df_cleaned

    def standardize_contact_methods(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize contact method values.

        EDA Finding: Inconsistent values ('Cell' vs 'cellular', 'Telephone' vs 'telephone')
        Distribution: Cell: 31.8%, cellular: 31.7%, Telephone: 18.4%, telephone: 18.1%

        Args:
            df: DataFrame with Contact Method column

        Returns:
            DataFrame with standardized contact methods
        """
        logger.info("Starting contact method standardization...")

        if "Contact Method" not in df.columns:
            logger.warning("Contact Method column not found")
            return df

        df_cleaned = df.copy()

        # Apply standardization mapping
        original_values = df_cleaned["Contact Method"].value_counts()
        df_cleaned["Contact Method"] = df_cleaned["Contact Method"].map(
            self.cleaning_config["contact_method_mapping"]
        )

        # Count standardizations
        standardized_count = len(df_cleaned)
        self.cleaning_stats["contact_methods_standardized"] = standardized_count

        new_values = df_cleaned["Contact Method"].value_counts()

        logger.info(f"Contact method standardization completed:")
        logger.info(f"Original distribution: {original_values.to_dict()}")
        logger.info(f"Standardized distribution: {new_values.to_dict()}")

        return df_cleaned

    def encode_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode target variable from text to binary format.

        EDA Finding: Target variable in text format ('yes', 'no')
        Requirement: Binary encoding (1=yes, 0=no) for model compatibility

        Args:
            df: DataFrame with Subscription Status column

        Returns:
            DataFrame with binary encoded target variable
        """
        logger.info("Starting target variable encoding...")

        if "Subscription Status" not in df.columns:
            logger.warning("Subscription Status column not found")
            return df

        df_cleaned = df.copy()

        # Apply binary encoding
        original_distribution = df_cleaned["Subscription Status"].value_counts()
        df_cleaned["Subscription Status"] = df_cleaned["Subscription Status"].map(
            self.cleaning_config["target_encoding"]
        )

        # Count encodings
        encoded_count = len(df_cleaned)
        self.cleaning_stats["target_variables_encoded"] = encoded_count

        new_distribution = df_cleaned["Subscription Status"].value_counts()

        logger.info(f"Target variable encoding completed:")
        logger.info(f"Original distribution: {original_distribution.to_dict()}")
        logger.info(f"Encoded distribution: {new_distribution.to_dict()}")

        return df_cleaned

    def validate_campaign_calls(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean campaign calls column.

        EDA Finding: Negative values detected (range: -41 to 56 calls)
        Business Rule: Cap at realistic limits (0-50 calls)

        Args:
            df: DataFrame with Campaign Calls column

        Returns:
            DataFrame with validated campaign calls
        """
        logger.info("Starting campaign calls validation...")

        if "Campaign Calls" not in df.columns:
            logger.warning("Campaign Calls column not found")
            return df

        df_cleaned = df.copy()

        # Handle negative values
        negative_mask = df_cleaned["Campaign Calls"] < 0
        negative_count = negative_mask.sum()

        if negative_count > 0:
            logger.warning(
                f"Found {negative_count} negative campaign call values - setting to 1"
            )
            df_cleaned.loc[negative_mask, "Campaign Calls"] = 1
            self.cleaning_stats["outliers_capped"] += negative_count

        # Cap extreme high values
        high_mask = (
            df_cleaned["Campaign Calls"] > self.cleaning_config["campaign_calls_max"]
        )
        high_count = high_mask.sum()

        if high_count > 0:
            logger.warning(
                f"Found {high_count} campaign calls above {self.cleaning_config['campaign_calls_max']} - capping"
            )
            df_cleaned.loc[high_mask, "Campaign Calls"] = self.cleaning_config[
                "campaign_calls_max"
            ]
            self.cleaning_stats["outliers_capped"] += high_count

        logger.info(
            f"Campaign calls validation completed. Capped {negative_count + high_count} outliers"
        )
        return df_cleaned

    def clean_banking_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main data cleaning pipeline implementing all Phase 3 requirements.

        This function orchestrates all cleaning operations based on EDA findings:
        1. Age conversion from text to numeric
        2. Missing values handling (28,935 total)
        3. Special values cleaning (12,008 total)
        4. Contact method standardization
        5. Target variable binary encoding
        6. Campaign calls validation

        Args:
            df: Raw DataFrame from initial_dataset.csv

        Returns:
            Cleaned DataFrame ready for Phase 4
        """
        logger.info("=" * 80)
        logger.info("STARTING COMPREHENSIVE DATA CLEANING PIPELINE")
        logger.info("=" * 80)

        # Initialize cleaning stats
        initial_shape = df.shape
        logger.info(f"Initial dataset shape: {initial_shape}")

        # Step 1: Clean age column
        df_cleaned = self.clean_age_column(df)
        logger.info("✅ Step 1: Age column cleaning completed")

        # Step 2: Handle missing values
        df_cleaned = self.handle_missing_values(df_cleaned)
        logger.info("✅ Step 2: Missing values handling completed")

        # Step 3: Clean special values
        df_cleaned = self.clean_special_values(df_cleaned)
        logger.info("✅ Step 3: Special values cleaning completed")

        # Step 4: Standardize contact methods
        df_cleaned = self.standardize_contact_methods(df_cleaned)
        logger.info("✅ Step 4: Contact method standardization completed")

        # Step 5: Encode target variable
        df_cleaned = self.encode_target_variable(df_cleaned)
        logger.info("✅ Step 5: Target variable encoding completed")

        # Step 6: Validate campaign calls
        df_cleaned = self.validate_campaign_calls(df_cleaned)
        logger.info("✅ Step 6: Campaign calls validation completed")

        # Final validation
        final_shape = df_cleaned.shape
        logger.info(f"Final dataset shape: {final_shape}")

        # Generate cleaning summary
        self.generate_cleaning_summary(initial_shape, final_shape)

        logger.info("=" * 80)
        logger.info("DATA CLEANING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

        return df_cleaned

    def generate_cleaning_summary(
        self, initial_shape: Tuple[int, int], final_shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive cleaning summary report.

        Args:
            initial_shape: Initial dataset shape
            final_shape: Final dataset shape

        Returns:
            Dict containing cleaning summary
        """
        summary = {
            "dataset_transformation": {
                "initial_shape": initial_shape,
                "final_shape": final_shape,
                "rows_changed": final_shape[0] - initial_shape[0],
                "columns_added": final_shape[1] - initial_shape[1],
            },
            "cleaning_operations": self.cleaning_stats.copy(),
            "data_quality_improvement": self.data_quality_metrics.copy(),
        }

        # Calculate improvement metrics
        missing_improvement = (
            self.data_quality_metrics["initial_missing_count"]
            - self.data_quality_metrics["final_missing_count"]
        )

        summary["data_quality_improvement"][
            "missing_values_resolved"
        ] = missing_improvement
        summary["data_quality_improvement"]["missing_improvement_rate"] = (
            missing_improvement
            / max(self.data_quality_metrics["initial_missing_count"], 1)
            * 100
        )

        logger.info("📊 CLEANING SUMMARY:")
        logger.info(f"   • Age conversions: {self.cleaning_stats['age_conversions']}")
        logger.info(
            f"   • Missing values handled: {self.cleaning_stats['missing_values_handled']}"
        )
        logger.info(
            f"   • Special values cleaned: {self.cleaning_stats['special_values_cleaned']}"
        )
        logger.info(
            f"   • Contact methods standardized: {self.cleaning_stats['contact_methods_standardized']}"
        )
        logger.info(
            f"   • Target variables encoded: {self.cleaning_stats['target_variables_encoded']}"
        )
        logger.info(f"   • Outliers capped: {self.cleaning_stats['outliers_capped']}")
        logger.info(f"   • Missing values resolved: {missing_improvement}")

        return summary
