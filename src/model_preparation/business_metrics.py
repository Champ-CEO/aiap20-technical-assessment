"""
Business Metrics Module

Implements business-relevant metrics calculation with customer segment awareness.
Provides ROI calculation by campaign intensity and precision/recall by customer segments.

Key Features:
- Standard classification metrics (precision, recall, F1, AUC)
- Business ROI calculation with segment-specific values
- Customer segment-aware metrics analysis
- Campaign intensity-based ROI computation
- Performance monitoring for >97K records/second standard
"""

import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    classification_report, roc_auc_score, confusion_matrix
)

logger = logging.getLogger(__name__)

class BusinessMetrics:
    """
    Business metrics calculator with customer segment awareness.
    
    Calculates standard classification metrics and business-specific metrics
    including ROI by customer segment and campaign intensity.
    """
    
    def __init__(self):
        """Initialize business metrics calculator."""
        self.performance_standard = 97000  # records per second
        
        # Business value configurations
        self.segment_values = {
            'Premium': {'conversion_value': 200, 'contact_cost': 25},
            'Standard': {'conversion_value': 120, 'contact_cost': 15},
            'Basic': {'conversion_value': 80, 'contact_cost': 10}
        }
        
        self.campaign_intensity_costs = {
            'high': {'contact_cost': 50, 'conversion_value': 150},
            'medium': {'contact_cost': 30, 'conversion_value': 120},
            'low': {'contact_cost': 15, 'conversion_value': 100}
        }
        
        # Performance tracking
        self.performance_metrics = {
            'calculation_time': 0,
            'records_per_second': 0,
            'metrics_calculated': []
        }
    
    def calculate_comprehensive_metrics(self, y_true: Union[np.ndarray, pd.Series], 
                                      y_pred: Union[np.ndarray, pd.Series],
                                      y_pred_proba: Optional[Union[np.ndarray, pd.Series]] = None,
                                      segments: Optional[Union[np.ndarray, pd.Series]] = None,
                                      campaign_intensity: Optional[Union[np.ndarray, pd.Series]] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive business metrics with segment awareness.
        
        Args:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
            y_pred_proba (array-like, optional): Prediction probabilities
            segments (array-like, optional): Customer segments
            campaign_intensity (array-like, optional): Campaign intensity levels
        
        Returns:
            Dict[str, Any]: Comprehensive metrics including business ROI
        """
        start_time = time.time()
        
        try:
            # Convert to numpy arrays for consistency
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            
            # Calculate standard classification metrics
            standard_metrics = self._calculate_standard_metrics(y_true, y_pred, y_pred_proba)
            
            # Calculate segment-aware metrics
            segment_metrics = {}
            if segments is not None:
                segment_metrics = self._calculate_segment_metrics(y_true, y_pred, segments)
            
            # Calculate campaign intensity ROI
            campaign_roi = {}
            if campaign_intensity is not None:
                campaign_roi = self._calculate_campaign_roi(y_true, y_pred, campaign_intensity)
            
            # Calculate overall business ROI
            overall_roi = self._calculate_overall_roi(y_true, y_pred, segments)
            
            # Combine all metrics
            comprehensive_metrics = {
                'standard_metrics': standard_metrics,
                'segment_metrics': segment_metrics,
                'campaign_roi': campaign_roi,
                'overall_roi': overall_roi,
                'calculation_timestamp': pd.Timestamp.now(),
                'sample_size': len(y_true)
            }
            
            # Record performance
            calculation_time = time.time() - start_time
            self.performance_metrics['calculation_time'] = calculation_time
            self.performance_metrics['records_per_second'] = len(y_true) / calculation_time if calculation_time > 0 else float('inf')
            self.performance_metrics['metrics_calculated'].append('comprehensive')
            
            logger.info(f"Comprehensive metrics calculated for {len(y_true)} samples")
            logger.info(f"Calculation performance: {self.performance_metrics['records_per_second']:,.0f} records/sec")
            
            return comprehensive_metrics
            
        except Exception as e:
            logger.error(f"Metrics calculation failed: {str(e)}")
            raise ValueError(f"Metrics calculation failed: {str(e)}")
    
    def _calculate_standard_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate standard classification metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        # Add AUC if probabilities provided
        if y_pred_proba is not None:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
            except ValueError:
                metrics['auc'] = 0.0
        
        # Add confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics.update({
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        })
        
        return metrics
    
    def _calculate_segment_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 segments: Union[np.ndarray, pd.Series]) -> Dict[str, Dict[str, float]]:
        """Calculate metrics by customer segment."""
        segments = np.array(segments)
        segment_metrics = {}
        
        for segment in ['Premium', 'Standard', 'Basic']:
            segment_mask = segments == segment
            if segment_mask.sum() > 0:
                segment_y_true = y_true[segment_mask]
                segment_y_pred = y_pred[segment_mask]
                
                if len(segment_y_true) > 0:
                    segment_metrics[segment] = {
                        'precision': precision_score(segment_y_true, segment_y_pred, zero_division=0),
                        'recall': recall_score(segment_y_true, segment_y_pred, zero_division=0),
                        'f1_score': f1_score(segment_y_true, segment_y_pred, zero_division=0),
                        'accuracy': accuracy_score(segment_y_true, segment_y_pred),
                        'sample_size': len(segment_y_true),
                        'positive_rate': segment_y_true.mean()
                    }
                    
                    # Calculate segment-specific ROI
                    segment_roi = self._calculate_segment_roi(segment_y_true, segment_y_pred, segment)
                    segment_metrics[segment].update(segment_roi)
        
        return segment_metrics
    
    def _calculate_segment_roi(self, y_true: np.ndarray, y_pred: np.ndarray, segment: str) -> Dict[str, float]:
        """Calculate ROI for a specific customer segment."""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        # Get segment-specific values
        segment_config = self.segment_values.get(segment, self.segment_values['Standard'])
        conversion_value = segment_config['conversion_value']
        contact_cost = segment_config['contact_cost']
        
        # Calculate ROI components
        total_contacts = tp + fp
        total_revenue = tp * conversion_value
        total_cost = total_contacts * contact_cost
        
        roi = (total_revenue - total_cost) / max(total_cost, 1) if total_cost > 0 else 0
        
        return {
            'roi': roi,
            'total_contacts': int(total_contacts),
            'conversions': int(tp),
            'conversion_rate': tp / max(total_contacts, 1),
            'revenue': total_revenue,
            'cost': total_cost,
            'profit': total_revenue - total_cost
        }
    
    def _calculate_campaign_roi(self, y_true: np.ndarray, y_pred: np.ndarray,
                               campaign_intensity: Union[np.ndarray, pd.Series]) -> Dict[str, Dict[str, float]]:
        """Calculate ROI by campaign intensity."""
        campaign_intensity = np.array(campaign_intensity)
        campaign_roi = {}
        
        for intensity in ['low', 'medium', 'high']:
            intensity_mask = campaign_intensity == intensity
            if intensity_mask.sum() > 0:
                intensity_y_true = y_true[intensity_mask]
                intensity_y_pred = y_pred[intensity_mask]
                
                if len(intensity_y_true) > 0:
                    tp = np.sum((intensity_y_true == 1) & (intensity_y_pred == 1))
                    fp = np.sum((intensity_y_true == 0) & (intensity_y_pred == 1))
                    
                    # Get intensity-specific costs
                    intensity_config = self.campaign_intensity_costs.get(intensity, self.campaign_intensity_costs['medium'])
                    contact_cost = intensity_config['contact_cost']
                    conversion_value = intensity_config['conversion_value']
                    
                    total_contacts = tp + fp
                    total_revenue = tp * conversion_value
                    total_cost = total_contacts * contact_cost
                    
                    roi = (total_revenue - total_cost) / max(total_cost, 1) if total_cost > 0 else 0
                    
                    campaign_roi[intensity] = {
                        'roi': roi,
                        'total_contacts': int(total_contacts),
                        'conversions': int(tp),
                        'conversion_rate': tp / max(total_contacts, 1),
                        'revenue': total_revenue,
                        'cost': total_cost,
                        'profit': total_revenue - total_cost,
                        'sample_size': len(intensity_y_true)
                    }
        
        return campaign_roi
    
    def _calculate_overall_roi(self, y_true: np.ndarray, y_pred: np.ndarray,
                              segments: Optional[Union[np.ndarray, pd.Series]] = None) -> Dict[str, float]:
        """Calculate overall business ROI."""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        # Use weighted average values if segments available
        if segments is not None:
            segments = np.array(segments)
            total_revenue = 0
            total_cost = 0
            
            for segment in ['Premium', 'Standard', 'Basic']:
                segment_mask = segments == segment
                if segment_mask.sum() > 0:
                    segment_tp = np.sum((y_true[segment_mask] == 1) & (y_pred[segment_mask] == 1))
                    segment_fp = np.sum((y_true[segment_mask] == 0) & (y_pred[segment_mask] == 1))
                    
                    segment_config = self.segment_values[segment]
                    total_revenue += segment_tp * segment_config['conversion_value']
                    total_cost += (segment_tp + segment_fp) * segment_config['contact_cost']
        else:
            # Use standard values
            conversion_value = 120  # Average value
            contact_cost = 20       # Average cost
            
            total_revenue = tp * conversion_value
            total_cost = (tp + fp) * contact_cost
        
        roi = (total_revenue - total_cost) / max(total_cost, 1) if total_cost > 0 else 0
        
        return {
            'roi': roi,
            'total_contacts': int(tp + fp),
            'conversions': int(tp),
            'missed_conversions': int(fn),
            'conversion_rate': tp / max(tp + fp, 1),
            'revenue': total_revenue,
            'cost': total_cost,
            'profit': total_revenue - total_cost
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from metrics calculation."""
        return self.performance_metrics.copy()
    
    def validate_performance_standard(self) -> bool:
        """Check if metrics calculation performance meets the >97K records/second standard."""
        return self.performance_metrics['records_per_second'] >= self.performance_standard


class SegmentMetrics(BusinessMetrics):
    """Specialized metrics calculator focused on customer segments."""
    
    def calculate_segment_comparison(self, y_true: Union[np.ndarray, pd.Series],
                                   y_pred: Union[np.ndarray, pd.Series],
                                   segments: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
        """
        Calculate comparative metrics across customer segments.
        
        Args:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
            segments (array-like): Customer segments
        
        Returns:
            Dict[str, Any]: Comparative segment analysis
        """
        segment_metrics = self._calculate_segment_metrics(
            np.array(y_true), np.array(y_pred), segments
        )
        
        # Calculate segment comparisons
        comparison = {
            'segment_metrics': segment_metrics,
            'best_performing_segment': None,
            'segment_rankings': {},
            'performance_gaps': {}
        }
        
        if segment_metrics:
            # Rank segments by ROI
            roi_rankings = sorted(
                segment_metrics.items(),
                key=lambda x: x[1].get('roi', 0),
                reverse=True
            )
            
            comparison['best_performing_segment'] = roi_rankings[0][0] if roi_rankings else None
            comparison['segment_rankings'] = {
                'by_roi': [segment for segment, _ in roi_rankings],
                'by_precision': sorted(
                    segment_metrics.items(),
                    key=lambda x: x[1].get('precision', 0),
                    reverse=True
                )
            }
            
            # Calculate performance gaps
            if len(roi_rankings) > 1:
                best_roi = roi_rankings[0][1]['roi']
                worst_roi = roi_rankings[-1][1]['roi']
                comparison['performance_gaps']['roi_gap'] = best_roi - worst_roi
        
        return comparison


class ROICalculator:
    """Specialized ROI calculator for business value analysis."""
    
    def __init__(self):
        """Initialize ROI calculator."""
        self.default_values = {
            'conversion_value': 120,
            'contact_cost': 20,
            'opportunity_cost': 50
        }
    
    def calculate_roi_scenarios(self, y_true: np.ndarray, y_pred: np.ndarray,
                               scenarios: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate ROI under different business scenarios.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            scenarios (Dict): Different business value scenarios
        
        Returns:
            Dict[str, Dict[str, float]]: ROI for each scenario
        """
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        roi_scenarios = {}
        
        for scenario_name, values in scenarios.items():
            conversion_value = values.get('conversion_value', self.default_values['conversion_value'])
            contact_cost = values.get('contact_cost', self.default_values['contact_cost'])
            opportunity_cost = values.get('opportunity_cost', self.default_values['opportunity_cost'])
            
            revenue = tp * conversion_value
            cost = (tp + fp) * contact_cost + fn * opportunity_cost
            
            roi = (revenue - cost) / max(cost, 1) if cost > 0 else 0
            
            roi_scenarios[scenario_name] = {
                'roi': roi,
                'revenue': revenue,
                'cost': cost,
                'profit': revenue - cost
            }
        
        return roi_scenarios
