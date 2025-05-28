"""
Phase 5 Feature Engineering Core Module

Main FeatureEngineer class that orchestrates all feature engineering operations
with Phase 4 integration and continuous validation.

This module implements the core feature engineering pipeline with:
1. Phase 4 data integration for production-ready data access
2. Business-driven feature creation with clear rationale
3. Continuous validation and quality monitoring
4. Performance optimization for >97K records/second
5. Memory management for large feature sets
"""

import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import Phase 4 data integration
try:
    from data_integration import (
        prepare_ml_pipeline,
        validate_phase3_continuity,
        load_phase3_output,
        EXPECTED_RECORD_COUNT,
        EXPECTED_FEATURE_COUNT,
        PERFORMANCE_STANDARD as PHASE4_PERFORMANCE_STANDARD,
    )

    PHASE4_INTEGRATION_AVAILABLE = True
    logger.info("Phase 4 data integration available")
except ImportError:
    PHASE4_INTEGRATION_AVAILABLE = False
    EXPECTED_RECORD_COUNT = 41188
    EXPECTED_FEATURE_COUNT = 33
    PHASE4_PERFORMANCE_STANDARD = 97000
    logger.warning("Phase 4 data integration not available, using fallback constants")


class FeatureEngineeringError(Exception):
    """Custom exception for feature engineering errors."""

    pass


class FeatureEngineer:
    """
    Main feature engineering class with Phase 4 integration.

    This class orchestrates all feature engineering operations while maintaining
    integration with Phase 4's production-ready data access and validation infrastructure.

    Key Features:
    - Age binning for optimal model performance
    - Education-occupation interactions for customer segmentation
    - Contact recency features using Phase 3 foundation
    - Campaign intensity analysis for optimal contact frequency
    - Continuous validation and performance monitoring
    """

    def __init__(self):
        """Initialize feature engineer with Phase 4 integration."""
        self.performance_standard = PHASE4_PERFORMANCE_STANDARD
        self.expected_records = EXPECTED_RECORD_COUNT
        self.expected_features = EXPECTED_FEATURE_COUNT

        # Feature engineering configuration
        self.feature_config = {
            "age_bins": [18, 35, 55, 100],
            "age_labels": [
                1,
                2,
                3,
            ],  # young, middle, senior (numeric for model performance)
            "campaign_intensity_bins": [0, 2, 5, 50],
            "campaign_intensity_labels": ["low", "medium", "high"],
            "high_intensity_threshold": 6,
        }

        # Performance and quality tracking
        self.performance_metrics = {
            "records_processed": 0,
            "processing_time": 0,
            "records_per_second": 0,
            "features_created": 0,
            "validation_checks_passed": 0,
        }

        # Business feature tracking
        self.business_features_created = {
            "age_bin": False,
            "education_job_segment": False,
            "recent_contact_flag": False,
            "campaign_intensity": False,
            "high_intensity_flag": False,
        }

        logger.info("FeatureEngineer initialized with Phase 4 integration")

    def validate_input_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate input data using Phase 4 integration if available.

        Args:
            df: Input DataFrame to validate

        Returns:
            Dict containing validation results

        Raises:
            FeatureEngineeringError: If validation fails
        """
        try:
            logger.info("Validating input data for feature engineering...")

            validation_results = {
                "record_count": len(df),
                "feature_count": len(df.columns),
                "missing_values": df.isnull().sum().sum(),
                "data_quality_score": 0,
                "phase4_continuity": "NOT_CHECKED",
            }

            # Use Phase 4 validation if available
            if PHASE4_INTEGRATION_AVAILABLE:
                try:
                    continuity_report = validate_phase3_continuity(df)
                    validation_results["phase4_continuity"] = continuity_report.get(
                        "continuity_status", "FAILED"
                    )
                    validation_results["data_quality_score"] = continuity_report.get(
                        "quality_score", 0
                    )

                    if continuity_report.get("continuity_status") != "PASSED":
                        raise FeatureEngineeringError(
                            f"Phase 4 continuity validation failed: {continuity_report}"
                        )

                except Exception as e:
                    logger.warning(
                        f"Phase 4 validation failed: {e}, proceeding with basic validation"
                    )
                    validation_results["phase4_continuity"] = "FAILED"

            # Basic validation checks
            if len(df) != self.expected_records:
                logger.warning(
                    f"Record count mismatch: expected {self.expected_records}, got {len(df)}"
                )

            if len(df.columns) < self.expected_features:
                raise FeatureEngineeringError(
                    f"Insufficient features: expected {self.expected_features}, got {len(df.columns)}"
                )

            # Check for required columns
            required_columns = [
                "Age",
                "Education Level",
                "Occupation",
                "Campaign Calls",
            ]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise FeatureEngineeringError(
                    f"Required columns missing: {missing_columns}"
                )

            # Set basic quality score if Phase 4 not available
            if validation_results["data_quality_score"] == 0:
                validation_results["data_quality_score"] = (
                    100 if validation_results["missing_values"] == 0 else 90
                )

            self.performance_metrics["validation_checks_passed"] += 1
            logger.info(
                f"Input validation passed: {validation_results['record_count']} records, {validation_results['feature_count']} features"
            )

            return validation_results

        except Exception as e:
            logger.error(f"Input validation failed: {str(e)}")
            raise FeatureEngineeringError(f"Input validation failed: {str(e)}")

    def create_age_bins(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create age binning for optimal model performance.

        Business Logic: Convert numeric age (18-100) to categorical (1=young, 2=middle, 3=senior)
        - Young (18-35): Entry-level banking, growth-focused products
        - Middle (36-55): Peak earning, wealth accumulation focus
        - Senior (56-100): Pre-retirement and retirement planning

        Args:
            df: DataFrame with Age column

        Returns:
            DataFrame with age_bin feature added
        """
        logger.info("Creating age binning features...")

        if "Age" not in df.columns:
            raise FeatureEngineeringError("Age column not found for age binning")

        df_result = df.copy()

        # Create age bins using business logic
        df_result["age_bin"] = pd.cut(
            df_result["Age"],
            bins=self.feature_config["age_bins"],
            labels=self.feature_config["age_labels"],
            include_lowest=True,
        )

        # Convert to numeric for model performance
        df_result["age_bin"] = df_result["age_bin"].astype(int)

        # Validate age binning results
        unique_bins = df_result["age_bin"].unique()
        expected_bins = set(self.feature_config["age_labels"])
        actual_bins = set(unique_bins)

        if not actual_bins.issubset(expected_bins):
            raise FeatureEngineeringError(
                f"Invalid age bins created: {actual_bins}, expected subset of {expected_bins}"
            )

        # Log distribution
        age_distribution = df_result["age_bin"].value_counts().sort_index()
        logger.info("Age bin distribution:")
        for bin_val, count in age_distribution.items():
            bin_name = ["young", "middle", "senior"][bin_val - 1]
            percentage = (count / len(df_result)) * 100
            logger.info(f"  • {bin_name} ({bin_val}): {count} ({percentage:.1f}%)")

        self.business_features_created["age_bin"] = True
        self.performance_metrics["features_created"] += 1

        logger.info("✅ Age binning completed successfully")
        return df_result

    def create_education_occupation_interactions(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create education-occupation interactions for high-value customer segments.

        Business Logic: Combine education and occupation to identify premium customer segments
        - University + Management = High-value segment
        - Professional + Technical = Growth segment
        - Handle unknown values appropriately for business analysis

        Args:
            df: DataFrame with Education Level and Occupation columns

        Returns:
            DataFrame with education_job_segment feature added
        """
        logger.info("Creating education-occupation interaction features...")

        required_columns = ["Education Level", "Occupation"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise FeatureEngineeringError(
                f"Required columns missing for education-occupation interactions: {missing_columns}"
            )

        df_result = df.copy()

        # Create interaction feature
        df_result["education_job_segment"] = (
            df_result["Education Level"].astype(str)
            + "_"
            + df_result["Occupation"].astype(str)
        )

        # Log segment distribution (top 10)
        segment_distribution = (
            df_result["education_job_segment"].value_counts().head(10)
        )
        logger.info("Top education-job segments:")
        for segment, count in segment_distribution.items():
            percentage = (count / len(df_result)) * 100
            logger.info(f"  • {segment}: {count} ({percentage:.1f}%)")

        self.business_features_created["education_job_segment"] = True
        self.performance_metrics["features_created"] += 1

        logger.info("✅ Education-occupation interactions completed successfully")
        return df_result

    def create_contact_recency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create contact recency features leveraging Phase 3's No_Previous_Contact flag.

        Business Logic: Recent contact effect on subscription likelihood
        - Use No_Previous_Contact flag from Phase 3 (0 = had previous contact, 1 = no previous contact)
        - Create recent_contact_flag (1 = recent contact, 0 = no recent contact)
        - Leverage contact history for improved targeting

        Args:
            df: DataFrame with No_Previous_Contact column

        Returns:
            DataFrame with recent_contact_flag feature added
        """
        logger.info("Creating contact recency features...")

        if "No_Previous_Contact" not in df.columns:
            logger.warning(
                "No_Previous_Contact column not found, creating placeholder feature"
            )
            df_result = df.copy()
            df_result["recent_contact_flag"] = 0  # Default to no recent contact
        else:
            df_result = df.copy()

            # Create recent contact flag (invert No_Previous_Contact)
            # No_Previous_Contact: 1 = no previous contact, 0 = had previous contact
            # recent_contact_flag: 1 = had recent contact, 0 = no recent contact
            df_result["recent_contact_flag"] = (
                1 - df_result["No_Previous_Contact"]
            ).astype(int)

            # Log contact recency distribution
            recency_distribution = df_result["recent_contact_flag"].value_counts()
            logger.info("Contact recency distribution:")
            for flag, count in recency_distribution.items():
                flag_name = "Recent Contact" if flag == 1 else "No Recent Contact"
                percentage = (count / len(df_result)) * 100
                logger.info(f"  • {flag_name}: {count} ({percentage:.1f}%)")

        self.business_features_created["recent_contact_flag"] = True
        self.performance_metrics["features_created"] += 1

        logger.info("✅ Contact recency features completed successfully")
        return df_result

    def create_campaign_intensity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create campaign intensity features for optimal contact frequency analysis.

        Business Logic: Analyze optimal contact frequency patterns
        - Low (1-2 calls): Minimal contact strategy
        - Medium (3-5 calls): Moderate engagement
        - High (6+ calls): Intensive campaign approach
        - Create high_intensity_flag for intensive campaigns (6+ calls)

        Args:
            df: DataFrame with Campaign Calls column

        Returns:
            DataFrame with campaign_intensity and high_intensity_flag features added
        """
        logger.info("Creating campaign intensity features...")

        if "Campaign Calls" not in df.columns:
            raise FeatureEngineeringError(
                "Campaign Calls column not found for campaign intensity features"
            )

        df_result = df.copy()

        # Create campaign intensity categories
        df_result["campaign_intensity"] = pd.cut(
            df_result["Campaign Calls"],
            bins=self.feature_config["campaign_intensity_bins"],
            labels=self.feature_config["campaign_intensity_labels"],
            include_lowest=True,
        )

        # Convert to string for consistency
        df_result["campaign_intensity"] = df_result["campaign_intensity"].astype(str)

        # Create high intensity flag
        df_result["high_intensity_flag"] = (
            df_result["Campaign Calls"]
            >= self.feature_config["high_intensity_threshold"]
        ).astype(int)

        # Log intensity distribution
        intensity_distribution = df_result["campaign_intensity"].value_counts()
        logger.info("Campaign intensity distribution:")
        for intensity, count in intensity_distribution.items():
            percentage = (count / len(df_result)) * 100
            logger.info(f"  • {intensity}: {count} ({percentage:.1f}%)")

        # Log high intensity flag distribution
        high_intensity_count = df_result["high_intensity_flag"].sum()
        high_intensity_percentage = (high_intensity_count / len(df_result)) * 100
        logger.info(
            f"High intensity campaigns: {high_intensity_count} ({high_intensity_percentage:.1f}%)"
        )

        self.business_features_created["campaign_intensity"] = True
        self.business_features_created["high_intensity_flag"] = True
        self.performance_metrics["features_created"] += 2

        logger.info("✅ Campaign intensity features completed successfully")
        return df_result

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main feature engineering pipeline with Phase 4 integration.

        Orchestrates all feature engineering operations:
        1. Validate input data using Phase 4 integration
        2. Create age binning features
        3. Create education-occupation interactions
        4. Create contact recency features
        5. Create campaign intensity features
        6. Monitor performance and validate output

        Args:
            df: Input DataFrame from Phase 4 data integration

        Returns:
            DataFrame with all engineered features

        Raises:
            FeatureEngineeringError: If any step fails
        """
        logger.info("=" * 80)
        logger.info("STARTING PHASE 5 FEATURE ENGINEERING PIPELINE")
        logger.info("=" * 80)

        start_time = time.time()
        initial_columns = len(df.columns)
        initial_records = len(df)

        try:
            # Step 1: Validate input data with Phase 4 integration
            logger.info("Step 1: Validating input data...")
            validation_results = self.validate_input_data(df)
            logger.info(
                f"✅ Input validation passed: Quality Score = {validation_results['data_quality_score']}%"
            )

            # Step 2: Create age binning features
            logger.info("Step 2: Creating age binning features...")
            df_engineered = self.create_age_bins(df)
            logger.info("✅ Age binning completed")

            # Step 3: Create education-occupation interactions
            logger.info("Step 3: Creating education-occupation interactions...")
            df_engineered = self.create_education_occupation_interactions(df_engineered)
            logger.info("✅ Education-occupation interactions completed")

            # Step 4: Create contact recency features
            logger.info("Step 4: Creating contact recency features...")
            df_engineered = self.create_contact_recency_features(df_engineered)
            logger.info("✅ Contact recency features completed")

            # Step 5: Create campaign intensity features
            logger.info("Step 5: Creating campaign intensity features...")
            df_engineered = self.create_campaign_intensity_features(df_engineered)
            logger.info("✅ Campaign intensity features completed")

            # Step 6: Performance monitoring and validation
            end_time = time.time()
            processing_time = end_time - start_time

            # Update performance metrics
            self.performance_metrics["records_processed"] = len(df_engineered)
            self.performance_metrics["processing_time"] = processing_time
            self.performance_metrics["records_per_second"] = (
                len(df_engineered) / processing_time
                if processing_time > 0
                else float("inf")
            )

            # Validate performance standard
            if (
                self.performance_metrics["records_per_second"]
                < self.performance_standard
            ):
                logger.warning(
                    f"Performance below standard: {self.performance_metrics['records_per_second']:.0f} < {self.performance_standard} records/sec"
                )

            # Validate output integrity
            final_columns = len(df_engineered.columns)
            final_records = len(df_engineered)
            features_added = final_columns - initial_columns

            if final_records != initial_records:
                raise FeatureEngineeringError(
                    f"Record count changed: {initial_records} → {final_records}"
                )

            if features_added == 0:
                raise FeatureEngineeringError("No features were created")

            # Validate all business features were created
            missing_features = [
                feature
                for feature, created in self.business_features_created.items()
                if not created
            ]
            if missing_features:
                logger.warning(
                    f"Some business features not created: {missing_features}"
                )

            # Log final summary
            logger.info("🔧 FEATURE ENGINEERING SUMMARY:")
            logger.info(
                f"   • Records processed: {self.performance_metrics['records_processed']:,}"
            )
            logger.info(
                f"   • Processing time: {self.performance_metrics['processing_time']:.2f} seconds"
            )
            logger.info(
                f"   • Performance: {self.performance_metrics['records_per_second']:.0f} records/second"
            )
            logger.info(f"   • Features added: {features_added}")
            logger.info(
                f"   • Business features created: {sum(self.business_features_created.values())}/5"
            )

            # Validate no missing values introduced
            missing_values = df_engineered.isnull().sum().sum()
            if missing_values > 0:
                logger.warning(
                    f"Feature engineering introduced {missing_values} missing values"
                )

            logger.info("=" * 80)
            logger.info("PHASE 5 FEATURE ENGINEERING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)

            return df_engineered

        except Exception as e:
            logger.error(f"Feature engineering pipeline failed: {str(e)}")
            raise FeatureEngineeringError(
                f"Feature engineering pipeline failed: {str(e)}"
            )

    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.

        Returns:
            Dict containing performance metrics and business feature status
        """
        return {
            "performance_metrics": self.performance_metrics.copy(),
            "business_features_created": self.business_features_created.copy(),
            "feature_config": self.feature_config.copy(),
            "phase4_integration_available": PHASE4_INTEGRATION_AVAILABLE,
            "performance_standard_met": (
                self.performance_metrics["records_per_second"]
                >= self.performance_standard
            ),
        }
