"""
Report Generator Module

Implements comprehensive evaluation report generation.
Creates markdown and JSON reports with evaluation results and business insights.

Key Features:
- Markdown report generation
- JSON report export
- Executive summary creation
- Business recommendations
- Performance analysis
"""

import pandas as pd
import numpy as np
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
OUTPUT_DIR = "specs/output"
REPORT_TEMPLATE = "phase8-evaluation-report.md"


class ReportGenerator:
    """
    Report generator for Phase 8 implementation.
    
    Generates comprehensive evaluation reports including executive summaries,
    detailed analysis, and business recommendations.
    """
    
    def __init__(self, output_dir: str = OUTPUT_DIR):
        """
        Initialize ReportGenerator.
        
        Args:
            output_dir (str): Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.generated_reports = {}
        
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any]) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            evaluation_results (Dict): Complete evaluation results
            
        Returns:
            str: Path to generated report
        """
        start_time = time.time()
        
        # Generate markdown report
        markdown_content = self._generate_markdown_report(evaluation_results)
        
        # Save markdown report
        report_path = self.output_dir / REPORT_TEMPLATE
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        # Generate JSON summary
        json_summary = self._generate_json_summary(evaluation_results)
        json_path = self.output_dir / "evaluation_summary.json"
        with open(json_path, 'w') as f:
            json.dump(json_summary, f, indent=2, default=str)
        
        generation_time = time.time() - start_time
        logger.info(f"Evaluation report generated in {generation_time:.2f}s: {report_path}")
        
        return str(report_path)
    
    def _generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """Generate markdown report content."""
        
        # Extract key information
        evaluation_summary = results.get('evaluation_summary', {})
        model_comparison = results.get('model_comparison', {})
        business_analysis = results.get('business_analysis', {})
        production_recommendations = results.get('production_recommendations', {})
        
        # Start building markdown content
        content = []
        
        # Header
        content.append("# Phase 8 Model Evaluation Report")
        content.append("")
        content.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append("**Phase:** 8 - Model Evaluation")
        content.append("**Step:** 2 - Core Functionality Implementation")
        content.append("**Status:** ✅ COMPLETED")
        content.append("")
        
        # Executive Summary
        content.append("## Executive Summary")
        content.append("")
        
        total_models = evaluation_summary.get('total_models_evaluated', 0)
        content.append(f"Phase 8 Step 2 has been successfully completed with comprehensive evaluation of **{total_models} models**.")
        content.append("The evaluation includes performance metrics, business analysis, and production deployment recommendations.")
        content.append("")
        
        # Top Performers
        if 'top_performers' in evaluation_summary:
            content.append("### Top Performers")
            content.append("")
            top_performers = evaluation_summary['top_performers']
            
            for criterion, performer in top_performers.items():
                model_name = performer.get('model', 'Unknown')
                score = performer.get('score', 0)
                
                if criterion == 'overall':
                    content.append(f"- **Best Overall:** {model_name} (Score: {score:.4f})")
                elif criterion == 'accuracy':
                    content.append(f"- **Highest Accuracy:** {model_name} ({score:.1%})")
                elif criterion == 'speed':
                    content.append(f"- **Fastest Processing:** {model_name} ({score:,.0f} records/sec)")
            content.append("")
        
        # Model Performance Summary
        content.append("## Model Performance Summary")
        content.append("")
        
        if 'model_summary' in evaluation_summary:
            content.append("| Model | Accuracy | F1 Score | AUC Score | Speed (rec/sec) |")
            content.append("|-------|----------|----------|-----------|-----------------|")
            
            model_summary = evaluation_summary['model_summary']
            for model_name, metrics in model_summary.items():
                accuracy = metrics.get('accuracy', 0)
                f1_score = metrics.get('f1_score', 0)
                auc_score = metrics.get('auc_score', 0)
                speed = metrics.get('records_per_second', 0)
                
                content.append(f"| {model_name} | {accuracy:.3f} | {f1_score:.3f} | {auc_score:.3f} | {speed:,.0f} |")
            content.append("")
        
        # Business Analysis
        content.append("## Business Analysis")
        content.append("")
        
        if business_analysis:
            content.append("### Marketing ROI Analysis")
            content.append("")
            
            # Find best ROI model
            best_roi_model = None
            best_roi_value = -float('inf')
            
            for model_name, results in business_analysis.items():
                if results and 'marketing_roi' in results:
                    roi = results['marketing_roi'].get('overall_roi', 0)
                    if roi > best_roi_value:
                        best_roi_value = roi
                        best_roi_model = model_name
            
            if best_roi_model:
                content.append(f"**Best ROI Model:** {best_roi_model} ({best_roi_value:.1%} ROI)")
                content.append("")
                
                # Customer segment analysis
                roi_data = business_analysis[best_roi_model]['marketing_roi']
                if 'segment_roi' in roi_data:
                    content.append("#### Customer Segment ROI")
                    content.append("")
                    content.append("| Segment | ROI | Conversion Rate | Total Contacts |")
                    content.append("|---------|-----|-----------------|----------------|")
                    
                    segment_roi = roi_data['segment_roi']
                    for segment, data in segment_roi.items():
                        roi = data.get('roi', 0)
                        conv_rate = data.get('conversion_rate', 0)
                        contacts = data.get('total_contacts', 0)
                        content.append(f"| {segment} | {roi:.1%} | {conv_rate:.1%} | {contacts:,} |")
                    content.append("")
        
        # Model Comparison
        content.append("## Model Comparison Analysis")
        content.append("")
        
        if 'comparison_summary' in model_comparison:
            comparison_summary = model_comparison['comparison_summary']
            
            # Performance statistics
            if 'performance_statistics' in comparison_summary:
                stats = comparison_summary['performance_statistics']
                content.append("### Performance Statistics")
                content.append("")
                
                if 'accuracy' in stats:
                    acc_stats = stats['accuracy']
                    content.append(f"- **Accuracy Range:** {acc_stats['min']:.3f} - {acc_stats['max']:.3f}")
                    content.append(f"- **Average Accuracy:** {acc_stats['mean']:.3f} (±{acc_stats['std']:.3f})")
                
                if 'speed' in stats:
                    speed_stats = stats['speed']
                    content.append(f"- **Speed Range:** {speed_stats['min']:,.0f} - {speed_stats['max']:,.0f} records/sec")
                    content.append(f"- **Average Speed:** {speed_stats['mean']:,.0f} records/sec")
                content.append("")
        
        # Production Recommendations
        content.append("## Production Deployment Recommendations")
        content.append("")
        
        if production_recommendations:
            primary_model = production_recommendations.get('primary_model')
            secondary_model = production_recommendations.get('secondary_model')
            tertiary_model = production_recommendations.get('tertiary_model')
            
            content.append("### Recommended Deployment Strategy")
            content.append("")
            content.append(f"1. **Primary Model:** {primary_model or 'Not determined'}")
            content.append(f"2. **Secondary Model:** {secondary_model or 'Not determined'}")
            content.append(f"3. **Tertiary Model:** {tertiary_model or 'Not determined'}")
            content.append("")
            
            if 'rationale' in production_recommendations:
                content.append("### Rationale")
                content.append("")
                rationale = production_recommendations['rationale']
                for tier, reason in rationale.items():
                    content.append(f"- **{tier.title()}:** {reason}")
                content.append("")
        
        # Performance Monitoring
        if 'pipeline_performance' in results:
            content.append("## Performance Monitoring")
            content.append("")
            
            perf = results['pipeline_performance']
            total_time = perf.get('total_time', 0)
            records_per_second = perf.get('records_per_second', 0)
            meets_standard = perf.get('meets_performance_standard', False)
            
            content.append(f"- **Total Evaluation Time:** {total_time:.2f} seconds")
            content.append(f"- **Processing Speed:** {records_per_second:,.0f} records/second")
            content.append(f"- **Performance Standard:** {'✅ MET' if meets_standard else '❌ NOT MET'} (>97K records/sec)")
            content.append("")
        
        # Technical Details
        content.append("## Technical Implementation Details")
        content.append("")
        content.append("### Evaluation Components")
        content.append("")
        content.append("- **ModelEvaluator:** Performance metrics calculation")
        content.append("- **ModelComparator:** Model ranking and comparison")
        content.append("- **BusinessMetricsCalculator:** ROI and business value analysis")
        content.append("- **ProductionDeploymentValidator:** 3-tier deployment strategy")
        content.append("- **ModelVisualizer:** Performance charts and visualizations")
        content.append("- **ReportGenerator:** Comprehensive report generation")
        content.append("")
        
        content.append("### Data Sources")
        content.append("")
        content.append("- **Trained Models:** `trained_models/` directory (5 models)")
        content.append("- **Test Data:** `data/featured/featured-db.csv` (45 features)")
        content.append("- **Customer Segments:** Premium (31.6%), Standard (57.7%), Basic (10.7%)")
        content.append("")
        
        # Next Steps
        content.append("## Next Steps")
        content.append("")
        content.append("### Phase 8 Step 3: Comprehensive Testing")
        content.append("")
        content.append("1. **End-to-end pipeline validation**")
        content.append("2. **Performance optimization (>97K records/second)**")
        content.append("3. **Complete test execution of all existing tests**")
        content.append("4. **Business metrics validation with customer segment awareness**")
        content.append("5. **Comprehensive documentation at `specs/Phase8-report.md`**")
        content.append("")
        
        content.append("### Phase 9 Preparation")
        content.append("")
        content.append("1. **Model Selection and Optimization planning**")
        content.append("2. **Production deployment strategy finalization**")
        content.append("3. **Performance monitoring system setup**")
        content.append("4. **Business integration planning**")
        content.append("")
        
        # Footer
        content.append("---")
        content.append("")
        content.append("**Report Generated:** Phase 8 Model Evaluation Pipeline")
        content.append(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append("**Status:** ✅ STEP 2 COMPLETED - Ready for Step 3")
        
        return "\n".join(content)
    
    def _generate_json_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate JSON summary of evaluation results."""
        
        summary = {
            'report_metadata': {
                'generation_timestamp': datetime.now().isoformat(),
                'phase': 'Phase 8 - Model Evaluation',
                'step': 'Step 2 - Core Functionality Implementation',
                'status': 'COMPLETED'
            },
            'evaluation_summary': results.get('evaluation_summary', {}),
            'top_performers': {},
            'business_insights': {},
            'production_recommendations': results.get('production_recommendations', {}),
            'performance_metrics': results.get('pipeline_performance', {})
        }
        
        # Extract top performers
        if 'evaluation_summary' in results and 'top_performers' in results['evaluation_summary']:
            summary['top_performers'] = results['evaluation_summary']['top_performers']
        
        # Extract business insights
        business_analysis = results.get('business_analysis', {})
        if business_analysis:
            # Find best ROI
            best_roi = 0
            best_roi_model = None
            
            for model_name, model_results in business_analysis.items():
                if model_results and 'marketing_roi' in model_results:
                    roi = model_results['marketing_roi'].get('overall_roi', 0)
                    if roi > best_roi:
                        best_roi = roi
                        best_roi_model = model_name
            
            summary['business_insights'] = {
                'best_roi_model': best_roi_model,
                'best_roi_value': best_roi,
                'customer_segments_analyzed': ['Premium', 'Standard', 'Basic']
            }
        
        return summary
    
    def generate_executive_summary(self, results: Dict[str, Any]) -> str:
        """
        Generate executive summary report.
        
        Args:
            results (Dict): Evaluation results
            
        Returns:
            str: Path to executive summary
        """
        summary_content = []
        
        # Header
        summary_content.append("# Phase 8 Model Evaluation - Executive Summary")
        summary_content.append("")
        summary_content.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}")
        summary_content.append("")
        
        # Key Findings
        evaluation_summary = results.get('evaluation_summary', {})
        
        if 'top_performers' in evaluation_summary:
            top_performers = evaluation_summary['top_performers']
            
            summary_content.append("## Key Findings")
            summary_content.append("")
            
            if 'overall' in top_performers:
                best_model = top_performers['overall']['model']
                best_score = top_performers['overall']['score']
                summary_content.append(f"- **Best Overall Model:** {best_model} (Score: {best_score:.4f})")
            
            if 'accuracy' in top_performers:
                acc_model = top_performers['accuracy']['model']
                acc_score = top_performers['accuracy']['score']
                summary_content.append(f"- **Highest Accuracy:** {acc_model} ({acc_score:.1%})")
            
            summary_content.append("")
        
        # Business Impact
        business_analysis = results.get('business_analysis', {})
        if business_analysis:
            summary_content.append("## Business Impact")
            summary_content.append("")
            
            # Find best ROI
            best_roi_model = None
            best_roi_value = -float('inf')
            
            for model_name, model_results in business_analysis.items():
                if model_results and 'marketing_roi' in model_results:
                    roi = model_results['marketing_roi'].get('overall_roi', 0)
                    if roi > best_roi_value:
                        best_roi_value = roi
                        best_roi_model = model_name
            
            if best_roi_model:
                summary_content.append(f"- **Best ROI Model:** {best_roi_model} ({best_roi_value:.1%} ROI)")
                summary_content.append("- **Customer Segments:** Premium, Standard, Basic analyzed")
            
            summary_content.append("")
        
        # Recommendations
        production_recommendations = results.get('production_recommendations', {})
        if production_recommendations:
            summary_content.append("## Recommendations")
            summary_content.append("")
            
            primary_model = production_recommendations.get('primary_model')
            if primary_model:
                summary_content.append(f"- **Deploy Primary Model:** {primary_model}")
            
            summary_content.append("- **Proceed to Phase 8 Step 3:** Comprehensive testing and validation")
            summary_content.append("")
        
        # Save executive summary
        summary_path = self.output_dir / "executive_summary.md"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(summary_content))
        
        logger.info(f"Executive summary generated: {summary_path}")
        return str(summary_path)
