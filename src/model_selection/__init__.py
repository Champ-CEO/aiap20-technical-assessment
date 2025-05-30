"""
Phase 9 Model Selection Module

Provides model selection capabilities based on Phase 8 evaluation results.
Implements business-driven model selection with performance and ROI optimization.

Key Components:
- ModelSelector: Core model selection logic with Phase 8 integration
- Selection criteria: accuracy, speed, business_value
- 3-tier deployment strategy: Primary, Secondary, Tertiary models

Integration:
- Phase 8 evaluation results from data/results/evaluation_summary.json
- Customer segment awareness (Premium: 31.6%, Standard: 57.7%, Basic: 10.7%)
- Performance standard: >97K records/second
"""

from .model_selector import ModelSelector

__all__ = ['ModelSelector']
