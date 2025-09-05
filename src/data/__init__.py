# src/data/__init__.py
"""
Data Processing Module
=====================

Handles data collection, integration, processing, and quality assessment.
"""

from .hr_data_pipeline import HRDataProcessor
from .hr_data_utils import (
    DataQualityChecker,
    FeatureAnalyzer,
    DataExporter,
    ValidationHelper,
    ConfigManager,
    quick_quality_check,
    quick_feature_summary,
    validate_pipeline_output,
    run_full_analysis
)

__all__ = [
    'HRDataProcessor',
    'DataQualityChecker',
    'FeatureAnalyzer', 
    'DataExporter',
    'ValidationHelper',
    'ConfigManager',
    'quick_quality_check',
    'quick_feature_summary',
    'validate_pipeline_output',
    'run_full_analysis'
]