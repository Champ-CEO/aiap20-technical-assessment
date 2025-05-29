"""
Business Metrics Calculator Module

Implements business-relevant evaluation with customer segment awareness and ROI calculation.
Provides marketing ROI, precision/recall trade-offs, and threshold optimization.

Key Features:
- Marketing ROI calculation by customer segment (Premium: 31.6%, Standard: 57.7%, Basic: 10.7%)
- Precision/recall trade-offs using Phase 7 top features
- Threshold optimization for business outcomes
- Campaign intensity and customer value analysis
- Expected lift and cost per acquisition metrics
"""

import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.metrics import precision_recall_curve, roc_curve

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Customer segment rates from Phase 7
CUSTOMER_SEGMENT_RATES = {
    "Premium": 0.316,  # 31.6%
    "Standard": 0.577,  # 57.7%
    "Basic": 0.107,    # 10.7%
}

# Phase 7 top features for business analysis
PHASE7_TOP_FEATURES = [
    "Client ID",
    "Previous Contact Days", 
    "contact_effectiveness_score"
]

# Business value parameters
BUSINESS_VALUES = {
    "Premium": {
        "conversion_value": 5000,    # Revenue per Premium conversion
        "contact_cost": 50,          # Cost per Premium contact
        "campaign_intensity": 1.5    # Higher intensity for Premium
    },
    "Standard": {
        "conversion_value": 2000,    # Revenue per Standard conversion
        "contact_cost": 25,          # Cost per Standard contact
        "campaign_intensity": 1.0    # Standard intensity
    },
    "Basic": {
        "conversion_value": 500,     # Revenue per Basic conversion
        "contact_cost": 10,          # Cost per Basic contact
        "campaign_intensity": 0.5    # Lower intensity for Basic
    }
}

# Performance standard
PERFORMANCE_STANDARD = 97000  # >97K records/second


class BusinessMetricsCalculator:
    """
    Business metrics calculator for Phase 8 implementation.
    
    Handles ROI calculation, customer segment analysis, and business value optimization
    based on model predictions and customer segments.
    """
    
    def __init__(self):
        """Initialize BusinessMetricsCalculator."""
        self.business_results = {}
        self.performance_metrics = {}
        
    def calculate_marketing_roi(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              y_pred_proba: Optional[np.ndarray] = None,
                              customer_segments: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculate marketing ROI by customer segment.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_pred_proba (np.ndarray, optional): Prediction probabilities
            customer_segments (np.ndarray, optional): Customer segment labels
            
        Returns:
            Dict[str, Any]: Marketing ROI analysis by segment
        """
        start_time = time.time()
        
        # If no segments provided, create mock segments based on segment rates
        if customer_segments is None:
            customer_segments = self._generate_mock_segments(len(y_true))
        
        roi_results = {}
        
        # Calculate ROI for each customer segment
        for segment in CUSTOMER_SEGMENT_RATES.keys():
            segment_mask = customer_segments == segment
            
            if np.sum(segment_mask) == 0:
                continue
                
            segment_y_true = y_true[segment_mask]
            segment_y_pred = y_pred[segment_mask]
            
            # Calculate confusion matrix components
            tp = np.sum((segment_y_true == 1) & (segment_y_pred == 1))
            fp = np.sum((segment_y_true == 0) & (segment_y_pred == 1))
            tn = np.sum((segment_y_true == 0) & (segment_y_pred == 0))
            fn = np.sum((segment_y_true == 1) & (segment_y_pred == 0))
            
            # Get business values for this segment
            values = BUSINESS_VALUES[segment]
            
            # Calculate ROI metrics
            total_contacts = tp + fp
            total_conversions = tp
            missed_conversions = fn
            
            # Revenue and cost calculation
            total_revenue = total_conversions * values["conversion_value"]
            total_cost = total_contacts * values["contact_cost"]
            
            # ROI calculation
            roi = (total_revenue - total_cost) / max(total_cost, 1) if total_cost > 0 else 0
            
            # Additional business metrics
            conversion_rate = total_conversions / max(total_contacts, 1)
            cost_per_acquisition = total_cost / max(total_conversions, 1)
            
            roi_results[segment] = {
                'total_customers': int(np.sum(segment_mask)),
                'total_contacts': int(total_contacts),
                'total_conversions': int(total_conversions),
                'missed_conversions': int(missed_conversions),
                'conversion_rate': conversion_rate,
                'total_revenue': total_revenue,
                'total_cost': total_cost,
                'roi': roi,
                'cost_per_acquisition': cost_per_acquisition,
                'campaign_intensity': values["campaign_intensity"],
                'segment_rate': CUSTOMER_SEGMENT_RATES[segment]
            }
        
        # Calculate overall ROI
        overall_revenue = sum(result['total_revenue'] for result in roi_results.values())
        overall_cost = sum(result['total_cost'] for result in roi_results.values())
        overall_roi = (overall_revenue - overall_cost) / max(overall_cost, 1) if overall_cost > 0 else 0
        
        # Performance tracking
        calc_time = time.time() - start_time
        records_per_second = len(y_true) / calc_time if calc_time > 0 else 0
        
        results = {
            'segment_roi': roi_results,
            'overall_roi': overall_roi,
            'overall_revenue': overall_revenue,
            'overall_cost': overall_cost,
            'performance': {
                'calculation_time': calc_time,
                'records_per_second': records_per_second,
                'meets_performance_standard': records_per_second >= PERFORMANCE_STANDARD
            }
        }
        
        logger.info(f"Marketing ROI calculated: {overall_roi:.2%} overall ROI, {records_per_second:,.0f} records/sec")
        return results
    
    def analyze_precision_recall_tradeoffs(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                         customer_segments: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Analyze precision/recall trade-offs for business optimization.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred_proba (np.ndarray): Prediction probabilities
            customer_segments (np.ndarray, optional): Customer segment labels
            
        Returns:
            Dict[str, Any]: Precision/recall trade-off analysis
        """
        start_time = time.time()
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        # Find optimal threshold based on F1 score
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        # Calculate business impact at different thresholds
        threshold_analysis = {}
        test_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, optimal_threshold]
        
        for threshold in test_thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            
            # Calculate ROI at this threshold
            roi_result = self.calculate_marketing_roi(y_true, y_pred_thresh, y_pred_proba, customer_segments)
            
            threshold_analysis[f"threshold_{threshold:.2f}"] = {
                'threshold': threshold,
                'overall_roi': roi_result['overall_roi'],
                'overall_revenue': roi_result['overall_revenue'],
                'overall_cost': roi_result['overall_cost'],
                'precision': precision[min(optimal_idx, len(precision)-1)],
                'recall': recall[min(optimal_idx, len(recall)-1)]
            }
        
        # Performance tracking
        analysis_time = time.time() - start_time
        records_per_second = len(y_true) / analysis_time if analysis_time > 0 else 0
        
        results = {
            'optimal_threshold': optimal_threshold,
            'optimal_f1_score': f1_scores[optimal_idx],
            'threshold_analysis': threshold_analysis,
            'precision_recall_curve': {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': thresholds.tolist()
            },
            'performance': {
                'analysis_time': analysis_time,
                'records_per_second': records_per_second,
                'meets_performance_standard': records_per_second >= PERFORMANCE_STANDARD
            }
        }
        
        logger.info(f"Precision/recall analysis completed: optimal threshold {optimal_threshold:.3f}, {records_per_second:,.0f} records/sec")
        return results
    
    def calculate_expected_lift(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                              customer_segments: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculate expected lift from targeted campaigns.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred_proba (np.ndarray): Prediction probabilities
            customer_segments (np.ndarray, optional): Customer segment labels
            
        Returns:
            Dict[str, Any]: Expected lift analysis
        """
        start_time = time.time()
        
        # If no segments provided, create mock segments
        if customer_segments is None:
            customer_segments = self._generate_mock_segments(len(y_true))
        
        # Calculate baseline conversion rate (no targeting)
        baseline_conversion_rate = np.mean(y_true)
        
        # Calculate lift for different targeting strategies
        lift_results = {}
        
        # Strategy 1: Target top 10% by probability
        top_10_pct_threshold = np.percentile(y_pred_proba, 90)
        top_10_pct_mask = y_pred_proba >= top_10_pct_threshold
        top_10_pct_conversion = np.mean(y_true[top_10_pct_mask]) if np.sum(top_10_pct_mask) > 0 else 0
        
        lift_results['top_10_percent'] = {
            'threshold': top_10_pct_threshold,
            'targeted_customers': int(np.sum(top_10_pct_mask)),
            'conversion_rate': top_10_pct_conversion,
            'lift': top_10_pct_conversion / max(baseline_conversion_rate, 1e-8),
            'expected_conversions': int(np.sum(y_true[top_10_pct_mask]))
        }
        
        # Strategy 2: Target top 25% by probability
        top_25_pct_threshold = np.percentile(y_pred_proba, 75)
        top_25_pct_mask = y_pred_proba >= top_25_pct_threshold
        top_25_pct_conversion = np.mean(y_true[top_25_pct_mask]) if np.sum(top_25_pct_mask) > 0 else 0
        
        lift_results['top_25_percent'] = {
            'threshold': top_25_pct_threshold,
            'targeted_customers': int(np.sum(top_25_pct_mask)),
            'conversion_rate': top_25_pct_conversion,
            'lift': top_25_pct_conversion / max(baseline_conversion_rate, 1e-8),
            'expected_conversions': int(np.sum(y_true[top_25_pct_mask]))
        }
        
        # Segment-specific lift analysis
        segment_lift = {}
        for segment in CUSTOMER_SEGMENT_RATES.keys():
            segment_mask = customer_segments == segment
            if np.sum(segment_mask) == 0:
                continue
                
            segment_baseline = np.mean(y_true[segment_mask])
            segment_top_10_mask = segment_mask & top_10_pct_mask
            segment_targeted_conversion = np.mean(y_true[segment_top_10_mask]) if np.sum(segment_top_10_mask) > 0 else 0
            
            segment_lift[segment] = {
                'baseline_conversion': segment_baseline,
                'targeted_conversion': segment_targeted_conversion,
                'lift': segment_targeted_conversion / max(segment_baseline, 1e-8),
                'targeted_customers': int(np.sum(segment_top_10_mask))
            }
        
        # Performance tracking
        lift_time = time.time() - start_time
        records_per_second = len(y_true) / lift_time if lift_time > 0 else 0
        
        results = {
            'baseline_conversion_rate': baseline_conversion_rate,
            'targeting_strategies': lift_results,
            'segment_lift': segment_lift,
            'performance': {
                'calculation_time': lift_time,
                'records_per_second': records_per_second,
                'meets_performance_standard': records_per_second >= PERFORMANCE_STANDARD
            }
        }
        
        logger.info(f"Expected lift calculated: {lift_results['top_10_percent']['lift']:.2f}x lift for top 10%, {records_per_second:,.0f} records/sec")
        return results
    
    def _generate_mock_segments(self, n_samples: int) -> np.ndarray:
        """Generate mock customer segments based on segment rates."""
        segments = []
        for segment, rate in CUSTOMER_SEGMENT_RATES.items():
            n_segment = int(n_samples * rate)
            segments.extend([segment] * n_segment)
        
        # Fill remaining samples with Standard segment
        while len(segments) < n_samples:
            segments.append("Standard")
        
        # Shuffle and return
        segments = np.array(segments[:n_samples])
        np.random.shuffle(segments)
        return segments
