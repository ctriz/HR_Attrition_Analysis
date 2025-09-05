# =============================================================================
# HR Data Processing Utilities
# Supporting functions for the HR Attrition Analysis pipeline
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import json

class DataQualityChecker:
    """Data quality assessment utilities."""
    
    @staticmethod
    def check_data_quality(df: pd.DataFrame, target_col: str = 'Attrition') -> Dict[str, Any]:
        """Comprehensive data quality assessment."""
        quality_report = {
            'basic_info': {
                'shape': df.shape,
                'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
                'total_cells': df.shape[0] * df.shape[1]
            },
            'missing_data': DataQualityChecker._analyze_missing_data(df),
            'duplicates': {
                'duplicate_rows': df.duplicated().sum(),
                'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100
            },
            'data_types': DataQualityChecker._analyze_data_types(df),
            'target_distribution': df[target_col].value_counts().to_dict() if target_col in df.columns else None
        }
        
        return quality_report
    
    @staticmethod
    def _analyze_missing_data(df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns."""
        missing_data = df.isnull().sum()
        missing_percentage = (missing_data / len(df)) * 100
        
        return {
            'total_missing': missing_data.sum(),
            'columns_with_missing': (missing_data > 0).sum(),
            'missing_by_column': missing_data[missing_data > 0].to_dict(),
            'missing_percentage': missing_percentage[missing_percentage > 0].to_dict()
        }
    
    @staticmethod
    def _analyze_data_types(df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data types distribution."""
        dtype_counts = df.dtypes.value_counts().to_dict()
        
        return {
            'dtype_distribution': {str(k): v for k, v in dtype_counts.items()},
            'numerical_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist()
        }

class FeatureAnalyzer:
    """Feature analysis and summary utilities."""
    
    @staticmethod
    def analyze_categorical_features(df: pd.DataFrame, max_categories: int = 20) -> Dict[str, Dict]:
        """Analyze categorical features."""
        categorical_cols = df.select_dtypes(include=['object']).columns
        analysis = {}
        
        for col in categorical_cols:
            unique_values = df[col].nunique()
            if unique_values <= max_categories:
                analysis[col] = {
                    'unique_count': unique_values,
                    'value_counts': df[col].value_counts().to_dict(),
                    'most_frequent': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                    'least_frequent': df[col].value_counts().index[-1]
                }
        
        return analysis
    
    @staticmethod
    def analyze_numerical_features(df: pd.DataFrame) -> Dict[str, Dict]:
        """Analyze numerical features."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        analysis = {}
        
        for col in numerical_cols:
            analysis[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'q25': df[col].quantile(0.25),
                'q75': df[col].quantile(0.75),
                'skewness': df[col].skew(),
                'kurtosis': df[col].kurtosis(),
                'zeros_count': (df[col] == 0).sum(),
                'outliers_iqr': FeatureAnalyzer._count_outliers_iqr(df[col])
            }
        
        return analysis
    
    @staticmethod
    def _count_outliers_iqr(series: pd.Series) -> int:
        """Count outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return ((series < lower_bound) | (series > upper_bound)).sum()

class DataExporter:
    """Data export utilities."""
    
    @staticmethod
    def export_data_summary(df: pd.DataFrame, output_dir: str = 'reports') -> None:
        """Export comprehensive data summary to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Basic info
        with open(output_path / 'data_info.txt', 'w') as f:
            f.write(f"Dataset Shape: {df.shape}\n")
            f.write(f"Columns: {list(df.columns)}\n\n")
            f.write("Data Types:\n")
            f.write(str(df.dtypes))
        
        # Statistical summary
        df.describe().to_csv(output_path / 'numerical_summary.csv')
        df.describe(include='O').to_csv(output_path / 'categorical_summary.csv')
        
        # Missing values
        missing_df = df.isnull().sum().to_frame('Missing_Count')
        missing_df['Missing_Percentage'] = (missing_df['Missing_Count'] / len(df)) * 100
        missing_df.to_csv(output_path / 'missing_values.csv')
        
        print(f"Data summary exported to {output_path}")
    
    @staticmethod
    def save_quality_report(quality_report: Dict, output_path: str = 'reports/quality_report.json') -> None:
        """Save data quality report as JSON."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        clean_report = convert_numpy_types(quality_report)
        
        with open(output_file, 'w') as f:
            json.dump(clean_report, f, indent=2)
        
        print(f"Quality report saved to {output_file}")

class ValidationHelper:
    """Data validation utilities."""
    
    @staticmethod
    def validate_employee_data(df: pd.DataFrame) -> Dict[str, List[str]]:
        """Validate HR employee data for common issues."""
        issues = {
            'warnings': [],
            'errors': [],
            'suggestions': []
        }
        
        # Check for essential columns
        essential_columns = ['Age', 'Department', 'JobRole', 'MonthlyIncome']
        missing_essential = [col for col in essential_columns if col not in df.columns]
        if missing_essential:
            issues['errors'].append(f"Missing essential columns: {missing_essential}")
        
        # Age validation
        if 'Age' in df.columns:
            invalid_ages = ((df['Age'] < 16) | (df['Age'] > 70)).sum()
            if invalid_ages > 0:
                issues['warnings'].append(f"Found {invalid_ages} records with unusual ages")
        
        # Income validation
        if 'MonthlyIncome' in df.columns:
            zero_income = (df['MonthlyIncome'] <= 0).sum()
            if zero_income > 0:
                issues['warnings'].append(f"Found {zero_income} records with zero/negative income")
        
        # Years validation
        year_columns = [col for col in df.columns if 'Years' in col]
        for col in year_columns:
            if col in df.columns:
                negative_years = (df[col] < 0).sum()
                if negative_years > 0:
                    issues['warnings'].append(f"Found {negative_years} negative values in {col}")
        
        # Consistency checks
        if 'YearsAtCompany' in df.columns and 'TotalWorkingYears' in df.columns:
            inconsistent = (df['YearsAtCompany'] > df['TotalWorkingYears']).sum()
            if inconsistent > 0:
                issues['warnings'].append(f"Found {inconsistent} records where YearsAtCompany > TotalWorkingYears")
        
        # Suggestions
        if len(df) < 1000:
            issues['suggestions'].append("Consider collecting more data for better model performance")
        
        if df.isnull().sum().sum() > 0:
            issues['suggestions'].append("Consider handling missing values before analysis")
        
        return issues

class ConfigManager:
    """Configuration management for the pipeline."""
    
    DEFAULT_CONFIG = {
        'data_paths': {
            'raw_data': 'data/HRAttrition.csv',
            'processed_data': 'data/HRAttrition_Processed.csv',
            'output_dir': 'data/processed',
            'reports_dir': 'reports'
        },
        'survey_config': {
            'num_questions': 5,
            'sentiment_phrases_file': None,  # If None, uses default phrases
            'engagement_score_range': [1, 5]
        },
        'bls_config': {
            'series_id': 'LNS14000000',
            'start_year': '1980',
            'end_year': '2020',
            'api_key': None  # Add your BLS API key here
        },
        'preprocessing': {
            'columns_to_drop': ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'],
            'categorical_threshold': 30,  # Max unique values for categorical
            'outlier_method': 'iqr'  # 'iqr' or 'zscore'
        },
        'validation': {
            'age_range': [16, 70],
            'income_min': 0,
            'check_consistency': True
        }
    }
    
    @classmethod
    def load_config(cls, config_path: Optional[str] = None) -> Dict:
        """Load configuration from file or return default."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            
            # Merge with default config
            config = cls.DEFAULT_CONFIG.copy()
            config.update(user_config)
            return config
        
        return cls.DEFAULT_CONFIG.copy()
    
    @classmethod
    def save_config(cls, config: Dict, config_path: str = 'config/pipeline_config.json') -> None:
        """Save configuration to file."""
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Configuration saved to {config_file}")

# =============================================================================
# Integrated Processing Function
# =============================================================================

def run_full_analysis(config_path: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Run complete analysis with quality checks and validation.
    
    Returns:
        Tuple of (processed_dataframe, analysis_results)
    """
    from src.data.hr_data_pipeline import HRDataProcessor  # Import the main processor
    
    # Load configuration
    config = ConfigManager.load_config(config_path)
    
    # Initialize processor with config
    processor = HRDataProcessor(config['data_paths']['raw_data'])
    
    # Run pipeline
    processed_df = processor.run_complete_pipeline(
        include_unemployment=True
    )
    
    # Quality assessment
    quality_checker = DataQualityChecker()
    quality_report = quality_checker.check_data_quality(processed_df)
    
    # Feature analysis
    feature_analyzer = FeatureAnalyzer()
    categorical_analysis = feature_analyzer.analyze_categorical_features(processed_df)
    numerical_analysis = feature_analyzer.analyze_numerical_features(processed_df)
    
    # Validation
    validator = ValidationHelper()
    validation_results = validator.validate_employee_data(processed_df)
    
    # Export results
    exporter = DataExporter()
    exporter.export_data_summary(processed_df, config['data_paths']['reports_dir'])
    exporter.save_quality_report(quality_report)
    
    # Compile analysis results
    analysis_results = {
        'quality_report': quality_report,
        'categorical_analysis': categorical_analysis,
        'numerical_analysis': numerical_analysis,
        'validation_results': validation_results,
        'config_used': config
    }
    
    return processed_df, analysis_results

# =============================================================================
# Quick Access Functions
# =============================================================================

def quick_quality_check(df: pd.DataFrame) -> None:
    """Quick data quality overview."""
    checker = DataQualityChecker()
    quality_report = checker.check_data_quality(df)
    
    print("=== DATA QUALITY OVERVIEW ===")
    print(f"Shape: {quality_report['basic_info']['shape']}")
    print(f"Missing values: {quality_report['missing_data']['total_missing']}")
    print(f"Duplicates: {quality_report['duplicates']['duplicate_rows']}")
    
    if quality_report['missing_data']['columns_with_missing'] > 0:
        print(f"Columns with missing data: {quality_report['missing_data']['columns_with_missing']}")
    
    print(f"Data types: {quality_report['data_types']['dtype_distribution']}")

def quick_feature_summary(df: pd.DataFrame) -> None:
    """Quick feature analysis summary."""
    analyzer = FeatureAnalyzer()
    
    categorical_analysis = analyzer.analyze_categorical_features(df)
    numerical_analysis = analyzer.analyze_numerical_features(df)
    
    print("=== FEATURE SUMMARY ===")
    print(f"Categorical features: {len(categorical_analysis)}")
    print(f"Numerical features: {len(numerical_analysis)}")
    
    # Show high cardinality categorical features
    high_cardinality = {k: v['unique_count'] for k, v in categorical_analysis.items() 
                       if v['unique_count'] > 10}
    if high_cardinality:
        print(f"High cardinality categoricals: {high_cardinality}")
    
    # Show features with outliers
    outlier_features = {k: v['outliers_iqr'] for k, v in numerical_analysis.items() 
                       if v['outliers_iqr'] > 0}
    if outlier_features:
        print(f"Features with outliers (IQR method): {outlier_features}")

def validate_pipeline_output(df: pd.DataFrame) -> bool:
    """Validate that the pipeline output meets expectations."""
    validator = ValidationHelper()
    validation_results = validator.validate_employee_data(df)
    
    print("=== VALIDATION RESULTS ===")
    
    if validation_results['errors']:
        print("ERRORS:")
        for error in validation_results['errors']:
            print(f"  - {error}")
        return False
    
    if validation_results['warnings']:
        print("WARNINGS:")
        for warning in validation_results['warnings']:
            print(f"  - {warning}")
    
    if validation_results['suggestions']:
        print("SUGGESTIONS:")
        for suggestion in validation_results['suggestions']:
            print(f"  - {suggestion}")
    
    print("Validation completed - no critical errors found!")
    return True


# =============================================================================
# Example Usage and Testing
# =============================================================================

def run_pipeline_example():
    """Example of running the complete pipeline with utilities."""
    
    print("Running HR Attrition Data Pipeline Example")
    print("=" * 50)
    
    try:
        # Method 1: Using the integrated analysis function
        processed_df, analysis_results = run_full_analysis()
        
        # Quick quality check
        quick_quality_check(processed_df)
        print()
        
        # Quick feature summary
        quick_feature_summary(processed_df)
        print()
        
        # Validate output
        is_valid = validate_pipeline_output(processed_df)
        
        if is_valid:
            print("\n✓ Pipeline completed successfully!")
            print(f"Final dataset shape: {processed_df.shape}")
            print(f"Ready for EDA and modeling phases")
        else:
            print("\n✗ Pipeline completed with errors - please review")
        
        return processed_df, analysis_results
        
    except Exception as e:
        print(f"Pipeline failed with error: {str(e)}")
        print("Please check your data paths and dependencies")
        return None, None

if __name__ == "__main__":
    # Run the example
    df, results = run_pipeline_example()
    
    if df is not None:
        print("\nSample of processed data:")
        print(df.head())
        
        print(f"\nColumns in final dataset:")
        print(df.columns.tolist())