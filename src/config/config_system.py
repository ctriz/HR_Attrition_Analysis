"""
Configuration System for HR Attrition Prediction with Bias Mitigation
Provides centralized configuration management and hyperparameter tuning
"""

import yaml
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
import numpy as np

@dataclass
class ModelConfig:
    """Configuration for individual ML models"""
    name: str
    enabled: bool
    hyperparameters: Dict[str, Any]
    use_scaled_data: bool
    tune_hyperparameters: bool
    cv_folds: int

@dataclass
class FairnessConfig:
    """Configuration for fairness constraints"""
    enabled: bool
    sensitive_features: List[str]
    constraints: List[str]  # ['demographic_parity', 'equalized_odds', etc.]
    min_group_size: int
    apply_preprocessing: bool
    apply_inprocessing: bool
    apply_postprocessing: bool

@dataclass
class DataConfig:
    """Configuration for data processing"""
    data_path: str
    target_column: str
    test_size: float
    validation_size: float
    random_state: int
    balance_classes: bool
    smote_strategy: float
    handle_missing: str  # 'drop', 'median', 'mean', 'mode'
    correlation_threshold: float
    feature_selection_method: str  # 'all', 'lasso', 'rfe', 'mutual_info'

@dataclass
class FeatureEngineeringConfig:
    """Configuration for feature engineering"""
    use_dfs: bool
    dfs_max_depth: int
    create_custom_features: bool
    custom_features: List[str]
    create_interaction_terms: bool
    polynomial_degree: int
    create_categorical_buckets: bool

@dataclass
class EvaluationConfig:
    """Configuration for model evaluation"""
    metrics: List[str]  # ['accuracy', 'roc_auc', 'precision', 'recall', 'f1']
    cross_validation: bool
    cv_folds: int
    bootstrap_samples: int
    confidence_level: float
    generate_plots: bool
    plot_types: List[str]  # ['roc_curve', 'precision_recall', 'feature_importance']

@dataclass
class HRAttritionConfig:
    """Main configuration class"""
    project_name: str
    version: str
    description: str
    data: DataConfig
    feature_engineering: FeatureEngineeringConfig
    models: Dict[str, ModelConfig]
    fairness: FairnessConfig
    evaluation: EvaluationConfig
    
    def save_config(self, filepath: str):
        """Save configuration to YAML file"""
        config_dict = asdict(self)
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    @classmethod
    def load_config(cls, filepath: str):
        """Load configuration from YAML file"""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert nested dictionaries back to dataclasses
        config_dict['data'] = DataConfig(**config_dict['data'])
        config_dict['feature_engineering'] = FeatureEngineeringConfig(**config_dict['feature_engineering'])
        config_dict['fairness'] = FairnessConfig(**config_dict['fairness'])
        config_dict['evaluation'] = EvaluationConfig(**config_dict['evaluation'])
        
        # Convert model configurations
        models = {}
        for model_name, model_config in config_dict['models'].items():
            models[model_name] = ModelConfig(**model_config)
        config_dict['models'] = models
        
        return cls(**config_dict)

def create_default_config() -> HRAttritionConfig:
    """Create default configuration"""
    
    # Default model configurations
    models = {
        'Logistic Regression': ModelConfig(
            name='Logistic Regression',
            enabled=True,
            hyperparameters={
                'solver': 'liblinear',
                'penalty': 'l1',
                'C': 1.0,
                'random_state': 42,
                'max_iter': 1000
            },
            use_scaled_data=True,
            tune_hyperparameters=True,
            cv_folds=5
        ),
        
        'Random Forest': ModelConfig(
            name='Random Forest',
            enabled=True,
            hyperparameters={
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'bootstrap': True,
                'random_state': 42,
                'n_jobs': -1
            },
            use_scaled_data=False,
            tune_hyperparameters=True,
            cv_folds=5
        ),
        
        'XGBoost': ModelConfig(
            name='XGBoost',
            enabled=True,
            hyperparameters={
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 1.0,
                'colsample_bytree': 1.0,
                'random_state': 42,
                'eval_metric': 'logloss',
                'n_jobs': -1
            },
            use_scaled_data=False,
            tune_hyperparameters=True,
            cv_folds=5
        ),
        
        'LightGBM': ModelConfig(
            name='LightGBM',
            enabled=True,
            hyperparameters={
                'n_estimators': 100,
                'max_depth': -1,
                'learning_rate': 0.1,
                'subsample': 1.0,
                'colsample_bytree': 1.0,
                'random_state': 42,
                'verbose': -1,
                'n_jobs': -1
            },
            use_scaled_data=False,
            tune_hyperparameters=True,
            cv_folds=5
        ),
        
        'CatBoost': ModelConfig(
            name='CatBoost',
            enabled=True,
            hyperparameters={
                'iterations': 100,
                'depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'verbose': False,
                'thread_count': -1
            },
            use_scaled_data=False,
            tune_hyperparameters=True,
            cv_folds=5
        ),
        
        'AdaBoost': ModelConfig(
            name='AdaBoost',
            enabled=True,
            hyperparameters={
                'n_estimators': 50,
                'learning_rate': 1.0,
                'random_state': 42
            },
            use_scaled_data=False,
            tune_hyperparameters=True,
            cv_folds=5
        )
    }
    
    return HRAttritionConfig(
        project_name="HR Attrition Prediction with Bias Mitigation",
        version="1.0.0",
        description="Comprehensive HR attrition prediction with fairness constraints",
        
        data=DataConfig(
            data_path="data/HRAttrition-Revised.csv",
            target_column="Attrition",
            test_size=0.2,
            validation_size=0.2,
            random_state=42,
            balance_classes=True,
            smote_strategy=0.5,
            handle_missing="median",
            correlation_threshold=0.95,
            feature_selection_method="all"
        ),
        
        feature_engineering=FeatureEngineeringConfig(
            use_dfs=True,
            dfs_max_depth=2,
            create_custom_features=True,
            custom_features=[
                'Income_to_Age_Ratio',
                'Is_Senior',
                'ReciprocalYearsAtCompany',
                'RoleVsTenureRatio',
                'TenureBucket',
                'PromotionRecencyBucket'
            ],
            create_interaction_terms=False,
            polynomial_degree=2,
            create_categorical_buckets=True
        ),
        
        models=models,
        
        fairness=FairnessConfig(
            enabled=True,
            sensitive_features=['gender', 'religion', 'race', 'marital_status'],
            constraints=['demographic_parity'],
            min_group_size=5,
            apply_preprocessing=False,
            apply_inprocessing=True,
            apply_postprocessing=True
        ),
        
        evaluation=EvaluationConfig(
            metrics=['accuracy', 'roc_auc', 'precision', 'recall', 'f1'],
            cross_validation=True,
            cv_folds=5,
            bootstrap_samples=1000,
            confidence_level=0.95,
            generate_plots=True,
            plot_types=['roc_curve', 'precision_recall', 'feature_importance', 'confusion_matrix']
        )
    )

class HyperparameterTuner:
    """Hyperparameter tuning utilities"""
    
    @staticmethod
    def get_param_grids():
        """Get parameter grids for hyperparameter tuning"""
        return {
            'Logistic Regression': {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            
            'Random Forest': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            },
            
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            
            'LightGBM': {
                'n_estimators': [50, 100, 200],
                'max_depth': [-1, 10, 20],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            
            'CatBoost': {
                'iterations': [50, 100, 200],
                'depth': [4, 6, 8],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            
            'AdaBoost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.5, 1.0, 1.5]
            }
        }
    
    @staticmethod
    def get_reduced_param_grids():
        """Get reduced parameter grids for faster tuning"""
        return {
            'Logistic Regression': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2']
            },
            
            'Random Forest': {
                'n_estimators': [100, 200],
                'max_depth': [None, 20],
                'min_samples_split': [2, 5]
            },
            
            'XGBoost': {
                'n_estimators': [100, 200],
                'max_depth': [6, 10],
                'learning_rate': [0.1, 0.2]
            },
            
            'LightGBM': {
                'n_estimators': [100, 200],
                'max_depth': [-1, 20],
                'learning_rate': [0.1, 0.2]
            },
            
            'CatBoost': {
                'iterations': [100, 200],
                'depth': [6, 8],
                'learning_rate': [0.1, 0.2]
            },
            
            'AdaBoost': {
                'n_estimators': [50, 100],
                'learning_rate': [0.5, 1.0]
            }
        }

def save_default_config(filepath: str = "config/hr_attrition_config.yaml"):
    """Save default configuration to file"""
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    config = create_default_config()
    config.save_config(filepath)
    print(f"Default configuration saved to {filepath}")

def load_and_validate_config(filepath: str) -> HRAttritionConfig:
    """Load and validate configuration"""
    try:
        config = HRAttritionConfig.load_config(filepath)
        
        # Basic validation
        if config.data.test_size <= 0 or config.data.test_size >= 1:
            raise ValueError("test_size must be between 0 and 1")
        
        if config.fairness.enabled and not config.fairness.sensitive_features:
            raise ValueError("sensitive_features cannot be empty when fairness is enabled")
        
        if not config.evaluation.metrics:
            raise ValueError("evaluation.metrics cannot be empty")
        
        print(f"Configuration loaded successfully from {filepath}")
        return config
        
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("Using default configuration instead...")
        return create_default_config()

# Example usage
if __name__ == "__main__":
    # Create and save default configuration
    save_default_config()
    
    # Load and display configuration
    config = load_and_validate_config("config/hr_attrition_config.yaml")
    print(f"\nProject: {config.project_name}")
    print(f"Version: {config.version}")
    print(f"Models enabled: {[name for name, model in config.models.items() if model.enabled]}")
    print(f"Fairness enabled: {config.fairness.enabled}")
    print(f"Feature engineering enabled: {config.feature_engineering.use_dfs}")
