# =============================================================================
# Feature Engineering and Bias Mitigation Module
# =============================================================================

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pathlib import Path
import featuretools as ft

# Import the HRDataProcessor from your data pipeline
# To import modules from the src directory
import os, sys
script_path = Path(__file__).resolve()
src_path = script_path.parent.parent
sys.path.append(str(src_path))
from data.hr_data_pipeline import HRDataProcessor


warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    A class to perform advanced feature engineering, bias mitigation,
    and feature selection for the HR Attrition project.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initializes the FeatureEngineer with a pre-processed DataFrame.

        Parameters:
        - data (pd.DataFrame): The pre-processed HR attrition data.
        """
        self.df = data.copy()
        self.target = 'Attrition'
        self.unwanted_cols = ['Attrition']
        self.sensitive_feature = 'Gender' # Define the sensitive feature here
        print("FeatureEngineer initialized.")
        print(f"Original DataFrame shape: {self.df.shape}")

    def simulate_demography(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Simulates additional demographic data for more robust bias analysis.
        
        Parameters:
        - df_features (pd.DataFrame): DataFrame with existing features.
        
        Returns:
        - pd.DataFrame: DataFrame with added demographic features.
        """
        print("\nSimulating demographic data...")
        # Placeholder for more complex demographic simulation
        
        return df_features

    def mitigate_bias(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Applies a pre-processing bias mitigation technique.
        
        This method is a placeholder to demonstrate where a pre-processing
        mitigation step, like reweighing or massagedata, would be applied.
        For this project, we are focusing on in-processing mitigation
        using Fairlearn's ExponentiatedGradient within the ModelTrainer.

        Parameters:
        - df_features (pd.DataFrame): DataFrame with features.
        
        Returns:
        - pd.DataFrame: The DataFrame after bias mitigation (if any).
        """
        print("\nMitigating bias with respect to 'Gender'...")
        print("Bias mitigation logic would be implemented here.")
        print("Fairlearn's ExponentiatedGradient is an excellent choice.")
        return df_features

    def generate_features_with_featuretools(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Generates new features using Featuretools.

        Parameters:
        - df_features (pd.DataFrame): The DataFrame to generate features from.
        
        Returns:
        - pd.DataFrame: A DataFrame with newly generated features.
        """
        print("\nGenerating new features with Featuretools...")
        
        # Ensure the index is a single, unique value for Featuretools
        df_features = df_features.reset_index(drop=True)
        df_features['instance_id'] = range(len(df_features))
        
        # Define entity set
        es = ft.EntitySet(id="HR_attrition")
        
        es.add_dataframe(
            dataframe_name="employees",
            dataframe=df_features,
            index="instance_id"
        )
        
        # Perform deep feature synthesis
        # ft.primitives can be added here for more feature generation
        feature_matrix, feature_defs = ft.dfs(
            entityset=es,
            target_dataframe_name="employees",
            max_depth=1, # Keep depth low to prevent explosion of features
            verbose=False
        )
        
        print(f"Featuretools generated {len(feature_matrix.columns)} features.")
        return feature_matrix

    def feature_selection_with_lasso(self, df_features: pd.DataFrame, df_target: pd.Series) -> List[str]:
        """
        Performs feature selection using Lasso regression.
        
        Parameters:
        - df_features (pd.DataFrame): The DataFrame with features.
        - df_target (pd.Series): The target variable.
        
        Returns:
        - List[str]: A list of selected feature names.
        """
        print("\nPerforming feature selection with Lasso Regression...")
        
        # Convert target variable to numerical format for Lasso
        print("Converting target variable to numerical format...")
        le = LabelEncoder()
        df_target_numeric = le.fit_transform(df_target)

        # Scale the features for Lasso
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df_features)

        # Use Lasso with cross-validation to find the best alpha
        lasso_cv = LassoCV(
            cv=5, 
            random_state=42, 
            max_iter=10000, 
            n_jobs=-1
        ).fit(scaled_features, df_target_numeric)
        
        # Get the non-zero coefficients
        selected_features_mask = lasso_cv.coef_ != 0
        selected_features = df_features.columns[selected_features_mask].tolist()
        
        print(f"Lasso selected {len(selected_features)} features.")
        return selected_features
        
    def run_all_steps(self) -> Dict:
        """
        Runs the full feature engineering pipeline.

        Returns:
        - Dict: A dictionary containing the processed data and the list of selected features.
        """
        # Separate the sensitive feature and target from the main DataFrame
        features = self.df.drop(columns=[self.target, self.sensitive_feature], errors='ignore')
        target = self.df[self.target]
        sensitive_features = self.df[self.sensitive_feature]

        # 1. Simulate additional demographic data
        features_with_demography = self.simulate_demography(features)

        # 2. One-hot encode all categorical columns except the sensitive feature
        print("\nConverting categorical features to one-hot encoded format...")
        initial_categorical_cols = features_with_demography.select_dtypes(include=['object']).columns.tolist()
        features_one_hot = pd.get_dummies(features_with_demography, columns=initial_categorical_cols, drop_first=True)
        print(f"One-hot encoding applied. New shape: {features_one_hot.shape}")
        
        # 3. Mitigate bias on the features DataFrame
        features_mitigated = self.mitigate_bias(features_one_hot)
        
        # 4. Generate new features with Featuretools.
        final_features = self.generate_features_with_featuretools(features_mitigated)
        
        # CRITICAL FIX: Ensure all columns are numeric before feature selection
        # Featuretools might generate new categorical columns.
        print("\nChecking for non-numeric columns after feature generation...")
        non_numeric_cols = final_features.select_dtypes(include='object').columns.tolist()
        if non_numeric_cols:
            print(f"One-hot encoding the following columns: {non_numeric_cols}")
            final_features = pd.get_dummies(final_features, columns=non_numeric_cols, drop_first=True)
        else:
            print("No new categorical columns found. Proceeding with feature selection.")

        # 5. Use Lasso to select features from the final combined dataset
        selected_features = self.feature_selection_with_lasso(final_features, target)
        
        print("\n" + "=" * 50)
        print("Pipeline Completed.")
        print("=" * 50)

        # Re-join the sensitive feature and target to the processed data
        processed_data_with_target = final_features[selected_features].join(target).join(sensitive_features)

        return {
            "processed_data": processed_data_with_target,
            "selected_features": selected_features,
            "sensitive_feature": self.sensitive_feature
        }
