"""
Model Visualizer Module

Implements visualization generation for model evaluation results.
Creates performance charts, feature importance plots, and comparison visualizations.

Key Features:
- Performance comparison charts
- Feature importance plots
- Confusion matrix visualizations
- ROC curves and precision-recall curves
- Business metrics visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use("ggplot")
sns.set_palette("husl")

# Constants
VISUALIZATIONS_DIR = "visualizations"
FIGURE_SIZE = (12, 8)
DPI = 300


class ModelVisualizer:
    """
    Model visualizer for Phase 8 implementation.

    Generates comprehensive visualizations for model evaluation results
    including performance comparisons and business metrics.
    """

    def __init__(self, output_dir: str = VISUALIZATIONS_DIR):
        """
        Initialize ModelVisualizer.

        Args:
            output_dir (str): Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.generated_charts = {}

    def generate_performance_comparison(
        self, evaluation_results: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Generate performance comparison chart.

        Args:
            evaluation_results (Dict): Model evaluation results

        Returns:
            str: Path to saved chart
        """
        start_time = time.time()

        # Extract performance metrics
        models = []
        accuracy_scores = []
        f1_scores = []
        auc_scores = []
        speed_scores = []

        for model_name, results in evaluation_results.items():
            if results is not None:
                models.append(model_name)
                accuracy_scores.append(results.get("accuracy", 0))
                f1_scores.append(results.get("f1_score", 0))
                auc_scores.append(results.get("auc_score", 0) or 0)

                performance_metrics = results.get("performance", {})
                speed_scores.append(performance_metrics.get("records_per_second", 0))

        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold")

        # Accuracy comparison
        bars1 = ax1.bar(models, accuracy_scores, color="skyblue", alpha=0.8)
        ax1.set_title("Model Accuracy Comparison")
        ax1.set_ylabel("Accuracy")
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis="x", rotation=45)

        # Add value labels on bars
        for bar, score in zip(bars1, accuracy_scores):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom",
            )

        # F1 Score comparison
        bars2 = ax2.bar(models, f1_scores, color="lightgreen", alpha=0.8)
        ax2.set_title("Model F1 Score Comparison")
        ax2.set_ylabel("F1 Score")
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis="x", rotation=45)

        for bar, score in zip(bars2, f1_scores):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom",
            )

        # AUC Score comparison
        bars3 = ax3.bar(models, auc_scores, color="lightcoral", alpha=0.8)
        ax3.set_title("Model AUC Score Comparison")
        ax3.set_ylabel("AUC Score")
        ax3.set_ylim(0, 1)
        ax3.tick_params(axis="x", rotation=45)

        for bar, score in zip(bars3, auc_scores):
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom",
            )

        # Speed comparison (log scale for better visualization)
        bars4 = ax4.bar(models, speed_scores, color="gold", alpha=0.8)
        ax4.set_title("Model Speed Comparison (Records/Second)")
        ax4.set_ylabel("Records per Second")
        ax4.set_yscale("log")
        ax4.tick_params(axis="x", rotation=45)

        for bar, score in zip(bars4, speed_scores):
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                height * 1.1,
                f"{score:,.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()

        # Save chart
        chart_path = self.output_dir / "performance_comparison.png"
        plt.savefig(chart_path, dpi=DPI, bbox_inches="tight")
        plt.close()

        generation_time = time.time() - start_time
        logger.info(
            f"Performance comparison chart generated in {generation_time:.2f}s: {chart_path}"
        )

        return str(chart_path)

    def generate_feature_importance(
        self, feature_importance_data: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Generate feature importance plots.

        Args:
            feature_importance_data (Dict): Feature importance data for all models

        Returns:
            str: Path to saved chart
        """
        start_time = time.time()

        # Count available models
        available_models = [
            model for model, data in feature_importance_data.items() if data is not None
        ]

        if not available_models:
            logger.warning("No feature importance data available")
            return ""

        # Create subplots based on number of models
        n_models = len(available_models)
        n_cols = min(2, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6 * n_rows))
        if n_models == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if n_models == 1 else axes
        else:
            axes = axes.flatten()

        fig.suptitle("Feature Importance by Model", fontsize=16, fontweight="bold")

        for idx, model_name in enumerate(available_models):
            data = feature_importance_data[model_name]

            if "top_5_features" in data:
                features, importances = zip(*data["top_5_features"])

                # Create horizontal bar plot
                y_pos = np.arange(len(features))
                axes[idx].barh(y_pos, importances, alpha=0.8)
                axes[idx].set_yticks(y_pos)
                axes[idx].set_yticklabels(features)
                axes[idx].set_xlabel("Importance")
                axes[idx].set_title(f"{model_name} - Top 5 Features")
                axes[idx].invert_yaxis()  # Top feature at top

                # Add value labels
                for i, importance in enumerate(importances):
                    axes[idx].text(
                        importance + max(importances) * 0.01,
                        i,
                        f"{importance:.3f}",
                        va="center",
                    )

        # Hide unused subplots
        for idx in range(len(available_models), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()

        # Save chart
        chart_path = self.output_dir / "feature_importance.png"
        plt.savefig(chart_path, dpi=DPI, bbox_inches="tight")
        plt.close()

        generation_time = time.time() - start_time
        logger.info(
            f"Feature importance chart generated in {generation_time:.2f}s: {chart_path}"
        )

        return str(chart_path)

    def generate_confusion_matrices(
        self, evaluation_results: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Generate confusion matrix visualizations.

        Args:
            evaluation_results (Dict): Model evaluation results

        Returns:
            str: Path to saved chart
        """
        start_time = time.time()

        # Extract confusion matrices
        available_models = []
        confusion_matrices = []

        for model_name, results in evaluation_results.items():
            if results is not None and "confusion_matrix" in results:
                available_models.append(model_name)
                confusion_matrices.append(np.array(results["confusion_matrix"]))

        if not available_models:
            logger.warning("No confusion matrix data available")
            return ""

        # Create subplots
        n_models = len(available_models)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_models == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_models > 1 else [axes]
        else:
            axes = axes.flatten()

        fig.suptitle("Confusion Matrices by Model", fontsize=16, fontweight="bold")

        for idx, (model_name, cm) in enumerate(
            zip(available_models, confusion_matrices)
        ):
            # Create heatmap
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["No", "Yes"],
                yticklabels=["No", "Yes"],
                ax=axes[idx],
            )
            axes[idx].set_title(f"{model_name}")
            axes[idx].set_xlabel("Predicted")
            axes[idx].set_ylabel("Actual")

        # Hide unused subplots
        for idx in range(len(available_models), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()

        # Save chart
        chart_path = self.output_dir / "confusion_matrices.png"
        plt.savefig(chart_path, dpi=DPI, bbox_inches="tight")
        plt.close()

        generation_time = time.time() - start_time
        logger.info(
            f"Confusion matrices generated in {generation_time:.2f}s: {chart_path}"
        )

        return str(chart_path)

    def generate_business_metrics_chart(
        self, business_results: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Generate business metrics visualization.

        Args:
            business_results (Dict): Business analysis results

        Returns:
            str: Path to saved chart
        """
        start_time = time.time()

        # Extract ROI data
        models = []
        roi_values = []

        for model_name, results in business_results.items():
            if results is not None and "marketing_roi" in results:
                roi_data = results["marketing_roi"]
                models.append(model_name)
                roi_values.append(roi_data.get("overall_roi", 0))

        if not models:
            logger.warning("No business metrics data available")
            return ""

        # Create chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Business Metrics Analysis", fontsize=16, fontweight="bold")

        # ROI comparison
        bars = ax1.bar(models, roi_values, color="green", alpha=0.7)
        ax1.set_title("Marketing ROI by Model")
        ax1.set_ylabel("ROI (%)")
        ax1.tick_params(axis="x", rotation=45)
        ax1.axhline(y=0, color="red", linestyle="--", alpha=0.5)

        # Add value labels
        for bar, roi in zip(bars, roi_values):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(roi_values) * 0.01,
                f"{roi:.1%}",
                ha="center",
                va="bottom",
            )

        # Customer segment analysis (if available)
        if models and business_results[models[0]] is not None:
            first_model_results = business_results[models[0]]
            if (
                "marketing_roi" in first_model_results
                and "segment_roi" in first_model_results["marketing_roi"]
            ):
                segment_data = first_model_results["marketing_roi"]["segment_roi"]

                segments = list(segment_data.keys())
                segment_rois = [segment_data[seg]["roi"] for seg in segments]

                ax2.bar(
                    segments,
                    segment_rois,
                    color=["gold", "silver", "bronze"],
                    alpha=0.7,
                )
                ax2.set_title(f"ROI by Customer Segment ({models[0]})")
                ax2.set_ylabel("ROI (%)")

                # Add value labels
                for i, (segment, roi) in enumerate(zip(segments, segment_rois)):
                    ax2.text(
                        i,
                        roi + max(segment_rois) * 0.01,
                        f"{roi:.1%}",
                        ha="center",
                        va="bottom",
                    )

        plt.tight_layout()

        # Save chart
        chart_path = self.output_dir / "business_metrics.png"
        plt.savefig(chart_path, dpi=DPI, bbox_inches="tight")
        plt.close()

        generation_time = time.time() - start_time
        logger.info(
            f"Business metrics chart generated in {generation_time:.2f}s: {chart_path}"
        )

        return str(chart_path)

    def generate_chart(self, chart_type: str, data: Dict[str, Any]) -> Optional[str]:
        """
        Generate a specific chart type.

        Args:
            chart_type (str): Type of chart to generate
            data (Dict): Data for chart generation

        Returns:
            Optional[str]: Path to generated chart
        """
        chart_generators = {
            "performance_comparison": self.generate_performance_comparison,
            "feature_importance": self.generate_feature_importance,
            "confusion_matrix": self.generate_confusion_matrices,
            "business_metrics": self.generate_business_metrics_chart,
        }

        if chart_type in chart_generators:
            try:
                return chart_generators[chart_type](data)
            except Exception as e:
                logger.error(f"Error generating {chart_type} chart: {str(e)}")
                return None
        else:
            logger.warning(f"Unknown chart type: {chart_type}")
            return None

    def generate_all_charts(self, evaluation_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate all available charts.

        Args:
            evaluation_results (Dict): Complete evaluation results

        Returns:
            Dict[str, str]: Mapping of chart types to file paths
        """
        generated_charts = {}

        # Performance comparison
        if "detailed_evaluation" in evaluation_results:
            chart_path = self.generate_performance_comparison(
                evaluation_results["detailed_evaluation"]
            )
            if chart_path:
                generated_charts["performance_comparison"] = chart_path

        # Feature importance
        if "feature_importance" in evaluation_results:
            chart_path = self.generate_feature_importance(
                evaluation_results["feature_importance"]
            )
            if chart_path:
                generated_charts["feature_importance"] = chart_path

        # Confusion matrices
        if "detailed_evaluation" in evaluation_results:
            chart_path = self.generate_confusion_matrices(
                evaluation_results["detailed_evaluation"]
            )
            if chart_path:
                generated_charts["confusion_matrices"] = chart_path

        # Business metrics
        if "business_analysis" in evaluation_results:
            chart_path = self.generate_business_metrics_chart(
                evaluation_results["business_analysis"]
            )
            if chart_path:
                generated_charts["business_metrics"] = chart_path

        self.generated_charts = generated_charts
        logger.info(f"Generated {len(generated_charts)} charts")

        return generated_charts

    def generate_feature_importance_chart(
        self, feature_importance_data: Dict[str, Any]
    ) -> Path:
        """
        Generate feature importance chart for a single model.

        Args:
            feature_importance_data (Dict): Feature importance data

        Returns:
            Path: Path to saved chart
        """
        start_time = time.time()

        # Extract data
        feature_names = feature_importance_data.get("feature_names", [])
        importance_scores = feature_importance_data.get("importance_scores", [])
        model_name = feature_importance_data.get("model_name", "Unknown")

        if not feature_names or not importance_scores:
            logger.warning("No feature importance data available")
            return self.output_dir / "feature_importance_empty.png"

        # Create chart
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create horizontal bar plot
        y_pos = np.arange(len(feature_names))
        bars = ax.barh(y_pos, importance_scores, alpha=0.8, color="steelblue")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.set_xlabel("Importance Score")
        ax.set_title(f"{model_name} - Feature Importance")
        ax.invert_yaxis()  # Top feature at top

        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, importance_scores)):
            width = bar.get_width()
            ax.text(
                width + max(importance_scores) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}",
                ha="left",
                va="center",
            )

        plt.tight_layout()

        # Save chart
        chart_path = self.output_dir / f"feature_importance_{model_name.lower()}.png"
        plt.savefig(chart_path, dpi=DPI, bbox_inches="tight")
        plt.close()

        generation_time = time.time() - start_time
        logger.info(
            f"Feature importance chart generated in {generation_time:.2f}s: {chart_path}"
        )

        return chart_path
