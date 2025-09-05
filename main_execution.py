#!/usr/bin/env python3
"""
Main execution script for HR Attrition Analysis Project
Orchestrates the complete pipeline from data collection to analysis
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Tuple, Dict, Optional
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT / 'src'))

try:
    from src.config.config_system import (
        HRAttritionConfig, 
        load_and_validate_config, 
        save_default_config
    )
    from src.data.hr_data_pipeline import HRDataProcessor
    from src.data.hr_data_utils import (
        DataQualityChecker, 
        ValidationHelper,
        quick_quality_check,
        validate_pipeline_output
    )
    from analysis.data_analysis import AdvancedEDA
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Please ensure all modules are properly installed and the src directory structure is correct")
    sys.exit(1)


class HRAttritionPipeline:
    """Main pipeline orchestrator for HR Attrition Analysis"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the pipeline
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or "config/hr_attrition_config.yaml"
        self.config = None
        self.processor = None
        self.df_processed = None
        self.eda_results = None
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary project directories"""
        directories = [
            "data/raw", "data/processed", "data/external",
            "reports/data_quality_reports", "reports/eda_reports",
            "reports/model_performance", "logs", "config"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")
    
    def load_configuration(self) -> HRAttritionConfig:
        """Load and validate configuration"""
        logger.info("Loading configuration...")
        
        try:
            if not Path(self.config_path).exists():
                logger.warning(f"Configuration file {self.config_path} not found. Creating default configuration.")
                save_default_config(self.config_path)
            
            self.config = load_and_validate_config(self.config_path)
            logger.info(f"Configuration loaded successfully from {self.config_path}")
            return self.config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def run_data_pipeline(self) -> pd.DataFrame:
        """Execute the data collection and processing pipeline"""
        logger.info("Starting data collection and processing pipeline...")
        
        try:
            # Initialize processor
            data_path = self.config.data.data_path
            if not Path(data_path).exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            self.processor = HRDataProcessor(data_path)
            
            # Run complete pipeline
            include_unemployment = True  # Could be made configurable
            self.df_processed = self.processor.run_complete_pipeline(
                include_unemployment=include_unemployment
            )
            
            logger.info(f"Data pipeline completed. Final dataset shape: {self.df_processed.shape}")
            return self.df_processed
            
        except Exception as e:
            logger.error(f"Data pipeline failed: {e}")
            raise
    
    def run_quality_assessment(self) -> Dict:
        """Run comprehensive data quality assessment"""
        logger.info("Running data quality assessment...")
        
        if self.df_processed is None:
            raise ValueError("No processed data available. Run data pipeline first.")
        
        try:
            # Quick quality overview
            logger.info("Quick quality check:")
            quick_quality_check(self.df_processed)
            
            # Detailed quality assessment
            quality_checker = DataQualityChecker()
            quality_report = quality_checker.check_data_quality(
                self.df_processed, 
                target_col=self.config.data.target_column
            )
            
            # Validation
            validator = ValidationHelper()
            validation_results = validator.validate_employee_data(self.df_processed)
            
            # Combined quality results
            quality_results = {
                'quality_report': quality_report,
                'validation_results': validation_results,
                'is_valid': validate_pipeline_output(self.df_processed)
            }
            
            # Log results
            if quality_results['is_valid']:
                logger.info("✓ Data quality validation passed")
            else:
                logger.warning("⚠ Data quality validation failed - check logs for details")
            
            return quality_results
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            raise
    
    def run_eda_analysis(self) -> Dict:
        """Run comprehensive exploratory data analysis"""
        logger.info("Running exploratory data analysis...")
        
        if self.df_processed is None:
            raise ValueError("No processed data available. Run data pipeline first.")
        
        try:
            # Initialize EDA
            eda = AdvancedEDA(df=self.df_processed)
            
            # Run complete EDA analysis
            eda.run_full_analysis()
            
            # Store EDA results (could be expanded to capture more details)
            self.eda_results = {
                'dataset_shape': self.df_processed.shape,
                'numeric_columns': eda.numeric_cols,
                'categorical_columns': eda.categorical_cols,
                'datetime_columns': eda.datetime_cols,
                'analysis_completed': True
            }
            
            logger.info("✓ EDA analysis completed successfully")
            return self.eda_results
            
        except Exception as e:
            logger.error(f"EDA analysis failed: {e}")
            raise
    
    def generate_reports(self):
        """Generate comprehensive reports"""
        logger.info("Generating reports...")
        
        try:
            from src.data.hr_data_utils import DataExporter
            
            exporter = DataExporter()
            
            # Export data summary
            exporter.export_data_summary(
                self.df_processed, 
                output_dir='reports/data_quality_reports'
            )
            
            # Additional reporting could be added here
            logger.info("✓ Reports generated successfully")
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            raise
    
    def run_complete_pipeline(self) -> Tuple[pd.DataFrame, Dict, Dict]:
        """
        Run the complete HR Attrition analysis pipeline
        
        Returns:
            Tuple of (processed_dataframe, quality_results, eda_results)
        """
        logger.info("="*60)
        logger.info("STARTING HR ATTRITION ANALYSIS PIPELINE")
        logger.info("="*60)
        
        try:
            # 1. Load configuration
            self.load_configuration()
            
            # 2. Run data pipeline
            processed_df = self.run_data_pipeline()
            
            # 3. Quality assessment
            quality_results = self.run_quality_assessment()
            
            # 4. EDA analysis
            eda_results = self.run_eda_analysis()
            
            # 5. Generate reports
            self.generate_reports()
            
            logger.info("="*60)
            logger.info("HR ATTRITION ANALYSIS PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*60)
            logger.info(f"Final dataset shape: {processed_df.shape}")
            logger.info(f"Data quality status: {'✓ Passed' if quality_results['is_valid'] else '⚠ Issues detected'}")
            logger.info(f"EDA analysis: {'✓ Completed' if eda_results['analysis_completed'] else '✗ Failed'}")
            
            return processed_df, quality_results, eda_results
            
        except Exception as e:
            logger.error("="*60)
            logger.error("PIPELINE EXECUTION FAILED")
            logger.error("="*60)
            logger.error(f"Error: {e}")
            raise


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='HR Attrition Analysis Pipeline')
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/hr_attrition_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--skip-eda', 
        action='store_true',
        help='Skip EDA analysis'
    )
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='reports',
        help='Output directory for reports'
    )
    
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_arguments()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize pipeline
        pipeline = HRAttritionPipeline(config_path=args.config)
        
        # Run complete pipeline
        processed_df, quality_results, eda_results = pipeline.run_complete_pipeline()
        
        # Optional: Skip EDA if requested
        if args.skip_eda:
            logger.info("EDA analysis skipped as requested")
        
        # Print final summary
        print("\n" + "="*60)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*60)
        print(f"Dataset shape: {processed_df.shape}")
        print(f"Quality status: {'PASSED' if quality_results['is_valid'] else 'ISSUES DETECTED'}")
        print(f"EDA completed: {'YES' if eda_results['analysis_completed'] else 'NO'}")
        print(f"Reports location: {args.output_dir}")
        print("="*60)
        
        return processed_df, quality_results, eda_results
        
    except KeyboardInterrupt:
        logger.info("Pipeline execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Execute main pipeline
    df, quality, eda = main()
