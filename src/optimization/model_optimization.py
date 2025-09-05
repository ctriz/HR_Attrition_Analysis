# =============================================================================
# Model Optimization Module
# =============================================================================

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Any
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.metrics import demographic_parity_difference

warnings.filterwarnings('ignore')

class ModelOptimizer:
    """
    A class to optimize model hyperparameters using GridSearchCV and evaluate
    the optimized model's performance and fairness.
    """
    
    def __init__(self, model: Any, params: Dict, X: np.ndarray, y: np.ndarray, sensitive_features: pd.Series):
        """
        Initializes the ModelOptimizer.

        Parameters:
        - model (Any): The scikit-learn model object to optimize.
        - params (Dict): A dictionary of hyperparameters to search over.
        - X (np.ndarray): The feature data.
        - y (np.ndarray): The target labels.
        - sensitive_features (pd.Series): The sensitive feature values for bias mitigation.
        """
        self.model = model
        self.params = params
        self.X = X
        self.y = y
        self.sensitive_features = sensitive_features
        self.best_estimator = None
        self.best_params = None
        print("ModelOptimizer initialized.")

    def optimize_with_grid_search(self, scoring: str = 'roc_auc', cv: int = 5):
        """
        Performs a hyperparameter search using GridSearchCV.

        Parameters:
        - scoring (str): The scoring metric to use for optimization. 'roc_auc' is
                         ideal for imbalanced datasets.
        - cv (int): The number of cross-validation folds.
        
        Returns:
        - GridSearchCV object: The fitted GridSearchCV object.
        """
        print("\nStarting hyperparameter optimization with GridSearchCV...")
        
        # We use StratifiedKFold to ensure that each fold has a similar
        # proportion of the target class (Attrition), which is crucial
        # for an imbalanced dataset like this.
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.params,
            scoring=scoring,
            cv=skf,
            verbose=1,
            n_jobs=-1
        )
        
        # Fit the grid search to the data
        grid_search.fit(self.X, self.y)
        
        self.best_estimator = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        print("\nHyperparameter optimization completed.")
        print(f"Best parameters found: {self.best_params}")
        print(f"Best cross-validation {scoring} score: {grid_search.best_score_:.4f}")
        
        return grid_search

    def evaluate_optimized_model(self, X_test: np.ndarray, y_test: np.ndarray, sensitive_test: pd.Series) -> Dict:
        """
        Evaluates the performance and fairness of the optimized model.

        Parameters:
        - X_test (np.ndarray): The test feature set.
        - y_test (np.ndarray): The true test labels.
        - sensitive_test (pd.Series): The sensitive feature values for the test set.

        Returns:
        - Dict: A dictionary of evaluation metrics.
        """
        if self.best_estimator is None:
            print("No optimized model found. Please run optimize_with_grid_search() first.")
            return {}

        print("\nEvaluating the optimized model on the test set...")
        y_pred = self.best_estimator.predict(X_test)
        y_proba = self.best_estimator.predict_proba(X_test)[:, 1]

        # Calculate standard performance metrics
        accuracy = roc_auc = dp_diff = np.nan
        try:
            accuracy = roc_auc_score(y_test, y_proba)
            print(f"Optimized Model Accuracy: {accuracy:.4f}")

            # Debiasing the optimized model
            print("\nApplying bias mitigation to the optimized model...")
            debiased_model = ExponentiatedGradient(
                estimator=self.best_estimator,
                constraints=DemographicParity()
            )
            debiased_model.fit(self.X, self.y, sensitive_features=self.sensitive_features)
            debiased_pred = debiased_model.predict(X_test)
            
            dp_diff = demographic_parity_difference(y_true=y_test, y_pred=debiased_pred, sensitive_features=sensitive_test)
            
            print(f"Demographic Parity Difference (After Mitigation): {dp_diff:.4f}")

        except Exception as e:
            print(f"An error occurred during evaluation: {e}")

        return {
            'optimized_accuracy': accuracy,
            'optimized_demographic_parity_difference': dp_diff
        }
