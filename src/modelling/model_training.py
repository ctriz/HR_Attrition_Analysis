# =============================================================================
# Model Training and Bias Mitigation Module
# =============================================================================

import pandas as pd
import numpy as np
import warnings
from typing import List, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.metrics import MetricFrame, demographic_parity_difference
from pathlib import Path
import os
import sys
import pickle

# Import XGBoost to compare with RandomForest
from xgboost import XGBClassifier

# Add the src directory to the system path to allow imports from other modules
script_path = Path(__file__).resolve()
src_path = script_path.parent.parent
sys.path.append(str(src_path))

# Import classes from your other modules
from data.hr_data_pipeline import HRDataProcessor
from feature.feature_engg import FeatureEngineer
# Import the new ModelOptimizer class
from optimization.model_optimization import ModelOptimizer

warnings.filterwarnings('ignore')

class ModelTrainer:
    """
    A class to train, evaluate, and mitigate bias in a predictive model for HR attrition.
    """
    
    def __init__(self, data: pd.DataFrame, features: List[str], target: str = 'Attrition', sensitive_feature: str = 'Gender'):
        """
        Initializes the ModelTrainer.
        """
        self.df = data.copy()
        self.features = features
        self.target = target
        self.sensitive_feature = sensitive_feature
        self.models: Dict[str, any] = {}
        self.evaluation_metrics: Dict[str, Dict] = {}
        self.scaler = StandardScaler()
        self.le = LabelEncoder()
        
        # Check if target column exists
        if self.target not in self.df.columns:
            raise ValueError(f"Target column '{self.target}' not found in the DataFrame.")
            
        print("ModelTrainer initialized.")

    def _prepare_data(self):
        """
        Prepares the data for training by splitting and scaling.
        """
        X = self.df[self.features]
        y = self.le.fit_transform(self.df[self.target])
        
        # Ensure sensitive feature is available and valid
        if self.sensitive_feature not in X.columns:
            print(f"Sensitive feature '{self.sensitive_feature}' not found. Using dummy for fairness analysis.")
            sensitive_features = np.zeros(X.shape[0])
        else:
            sensitive_features = X[self.sensitive_feature]
            X = X.drop(columns=[self.sensitive_feature])

        # Split data
        X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
            X, y, sensitive_features, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale numeric features
        numeric_cols = X_train.select_dtypes(include=np.number).columns
        X_train.loc[:, numeric_cols] = self.scaler.fit_transform(X_train[numeric_cols])
        X_test.loc[:, numeric_cols] = self.scaler.transform(X_test[numeric_cols])
        
        return X_train, X_test, y_train, y_test, sensitive_train, sensitive_test

    def train_and_evaluate_model(self):
        """
        Runs the full training and evaluation pipeline for multiple models.
        """
        print("\n" + "=" * 50)
        print("Starting Model Training and Evaluation Pipeline")
        print("=" * 50)
        
        X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = self._prepare_data()
        
        # A list of models to train and evaluate
        models_to_train = {
            "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42),
            "LogisticRegression": LogisticRegression(solver='liblinear', random_state=42),
            "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100, random_state=42),
            "SVC": SVC(probability=True, random_state=42)
        }
        
        for name, model in models_to_train.items():
            print("\n" + "=" * 50)
            print(f"Training standard {name}...")
            model.fit(X_train, y_train)
            self.models[name] = model
            self._evaluate(model, name, X_test, y_test, sensitive_test)
            
            # Applying bias mitigation with ExponentiatedGradient
            print("\n" + "-" * 50)
            print(f"Training debiased {name} with ExponentiatedGradient...")
            debiased_model = ExponentiatedGradient(
                estimator=model,
                constraints=DemographicParity()
            )
            debiased_model.fit(X_train, y_train, sensitive_features=sensitive_train)
            self.models[f"Debiased {name}"] = debiased_model
            self._evaluate(debiased_model, f"Debiased {name}", X_test, y_test, sensitive_test)
            print("-" * 50)

    def _evaluate(self, model, name, X_test, y_test, sensitive_test):
        """
        Evaluates a model and prints key metrics.
        """
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
        except (AttributeError, IndexError):
            y_proba = None
            roc_auc = float('nan')
            
        metrics = {
            'Accuracy': accuracy,
            'ROC AUC': roc_auc,
            'Demographic Parity Difference': demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_test)
        }
        
        print(f"Evaluating {name}...")
        print(f"Accuracy: {metrics['Accuracy']:.4f}")
        if not np.isnan(roc_auc):
            print(f"ROC AUC: {metrics['ROC AUC']:.4f}")
        else:
            print("ROC AUC cannot be computed as the model does not have a predict_proba method.")
        
        mf = MetricFrame(
            metrics={'accuracy': accuracy_score, 'ROC AUC': roc_auc_score if not np.isnan(roc_auc) else 'not_applicable'},
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=sensitive_test
        )
        print("\nFairness Metrics:")
        print(mf.by_group)
        print(f"Demographic Parity Difference: {metrics['Demographic Parity Difference']:.4f}")

        # Store the metrics for the final comparison table
        self.evaluation_metrics[name] = metrics


if __name__ == "__main__":
    try:
        # Load processed data and run feature engineering pipeline
        data_processor = HRDataProcessor()
        actual_data = data_processor.run_complete_pipeline()
        
        if actual_data is None or actual_data.empty:
            print("Failed to load data from the pipeline.")
        else:
            fe = FeatureEngineer(actual_data)
            pipeline_results = fe.run_all_steps()
            
            # Extract the processed data and the selected features
            processed_data_with_target = pipeline_results["processed_data"]
            selected_features = pipeline_results["selected_features"]
            
            print("\nFinal selected features for model training:")
            print(selected_features)
            
            # Instantiate and run the model training pipeline
            trainer = ModelTrainer(data=processed_data_with_target, features=selected_features)
            trainer.train_and_evaluate_model()

            # ---
            # Begin Hyperparameter Optimization
            # ---
            X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = trainer._prepare_data()

            print("\n" + "=" * 50)
            print("Beginning Hyperparameter Optimization for XGBoost")
            print("=" * 50)
            
            # Define parameter grid for XGBoost
            xgb_params = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2]
            }

            # Create an instance of the ModelOptimizer with the XGBoost model
            optimizer = ModelOptimizer(
                model=XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
                params=xgb_params,
                X=X_train,
                y=y_train,
                sensitive_features=sensitive_train
            )

            # Run the optimization and get the best model
            optimizer.optimize_with_grid_search()
            
            # Evaluate the best model on the test set
            optimizer.evaluate_optimized_model(X_test, y_test, sensitive_test)
            
            print("\nOptimization and final evaluation complete.")
            
            # Save the final model
            project_dir = script_path.parent.parent.parent
            api_path = project_dir / 'api'
            model_path = api_path / 'model.pkl'
            # Ensure the directory exists before saving
            api_path.mkdir(parents=True, exist_ok=True)
            try:
                with open(model_path, 'wb') as f:
                    # Save the optimized XGBoost model from the optimizer instance,
                    # and the scaler/le from the trainer instance.
                    pickle.dump((optimizer.best_model_, trainer.scaler, trainer.le), f)
                print(f"\nFinal model, scaler, and label encoder saved to {model_path}")
            except Exception as e:
                print(f"Error saving model: {e}")

    except Exception as e:
        print(f"An error occurred: {e}")
