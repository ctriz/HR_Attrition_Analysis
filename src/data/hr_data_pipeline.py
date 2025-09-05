# =============================================================================
# HR Attrition Data Pipeline - Refactored
# =============================================================================

import pandas as pd
import numpy as np
import json
import requests
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Optional: Only install if not already present

from textblob import TextBlob
from sklearn.preprocessing import LabelEncoder

# Define the file path 
# Adjust the number of `.parent` calls to match the folder structure.
script_path = Path(__file__).resolve()
project_dir= script_path.parent.parent.parent
FILE_PATH = project_dir /'data'/'raw'/ 'HRAttrition.csv'
PROCESSED_FILE_PATH= project_dir /'data'/'raw'/ 'HRAttrition_Processed.csv'
JSON_FILE_PATH = project_dir /'data'/'external'/ 'bls_unemployment_series_data.json'



class HRDataProcessor:
    def __init__(self):
        """Initialize the processor wrelative path to data files."""
        # Use the variable defined at the top of the script.
        self.data_path = FILE_PATH
        self.json_path = JSON_FILE_PATH
        self.df_main = None
        self.categorical_mappings = self._get_categorical_mappings()
        np.random.seed(42)  # For reproducibility
    
    def _get_categorical_mappings(self) -> Dict[str, Dict[int, str]]:
        """Define categorical variable mappings."""
        return {
            "Education": {1: "Below College", 2: "College", 3: "Bachelor", 4: "Master", 5: "Doctor"},
            "EnvironmentSatisfaction": {1: "Low", 2: "Medium", 3: "High", 4: "Very High"},
            "JobInvolvement": {1: "Low", 2: "Medium", 3: "High", 4: "Very High"},
            "JobLevel": {1: "Entry Level", 2: "Junior Level", 3: "Mid Level", 4: "Senior Level", 5: "Executive Level"},
            "JobSatisfaction": {1: "Low", 2: "Medium", 3: "High", 4: "Very High"},
            "RelationshipSatisfaction": {1: "Low", 2: "Medium", 3: "High", 4: "Very High"},
            "WorkLifeBalance": {1: "Bad", 2: "Good", 3: "Better", 4: "Best"},
            "PerformanceRating": {1: "Low", 2: "Good", 3: "Excellent", 4: "Outstanding"},
        }
        
    def load_ibm_data(self) -> None:
        """Loads and initializes the main IBM HR Attrition data."""
        try:
            print(f"Loading data from {self.data_path}...")
            self.df_main = pd.read_csv(self.data_path)
            print(f"Loaded {len(self.df_main)} records with {len(self.df_main.columns)} columns")
        except FileNotFoundError:
            print(f"Error: The file '{self.data_path}' was not found.")
            self.df_main = pd.DataFrame() # Create an empty DataFrame to prevent errors
        except Exception as e:
            print(f"An error occurred while loading the data: {e}")
            self.df_main = pd.DataFrame()

    def generate_synthetic_survey_data(self) -> None:
        """Generates synthetic survey data and merges it with the main DataFrame."""
        print("Generating synthetic survey data...")
        num_records = len(self.df_main)
        survey_q1 = np.random.randint(1, 6, size=num_records)
        survey_q2 = np.random.randint(1, 6, size=num_records)
        survey_q3 = np.random.randint(1, 6, size=num_records)
        survey_q4 = np.random.randint(1, 6, size=num_records)
        survey_q5 = np.random.randint(1, 6, size=num_records)
        
        survey_df = pd.DataFrame({
            'Survey_Q1': survey_q1,
            'Survey_Q2': survey_q2,
            'Survey_Q3': survey_q3,
            'Survey_Q4': survey_q4,
            'Survey_Q5': survey_q5,
        })
        self.df_main = pd.concat([self.df_main, survey_df], axis=1)

    def generate_text_survey_data(self) -> None:
        """Generates synthetic text responses and performs sentiment analysis."""
        print("Generating text survey responses with sentiment analysis...")
        text_responses = [
            "I have clear goals for my career advancement.",
            "I'm encouraged to explore new roles.",
            "Uncertain about my future path here.",
            "The company supports my growth plans.",
            "I see limited progression options.",
            "I frequently attend learning sessions.",
            "Training opportunities are limited.",
            "I balance my work and personal life well.",
            "Long hours make it hard to spend time with family.",
            "Flexible working hours help me stay motivated.",
            "I enjoy the learning culture here.",
            "My career path aligns with my interests."
        ]
        
        num_records = len(self.df_main)
        synthetic_text_responses = np.random.choice(text_responses, size=num_records)
        
        sentiments = [TextBlob(text).sentiment.polarity for text in synthetic_text_responses]
        engagement_scores = [3 if 'motivated' in text or 'well' in text or 'supports' in text else 4 if 'clear goals' in text else 3 if 'enjoy' in text else 3 for text in synthetic_text_responses]
        
        text_df = pd.DataFrame({
            'Survey_Text_Response': synthetic_text_responses,
            'Sentiment_Polarity': sentiments,
            'Engagement_Score': engagement_scores
        })
        
        self.df_main = pd.concat([self.df_main, text_df], axis=1)

    def process_unemployment_data(self) -> None:
        """Loads and merges unemployment data based on join year."""
        print("Processing unemployment data...")
        try:
            with open(self.json_path, 'r') as f:
                data = json.load(f)
            print("Data loaded successfully.")

            unemployment_rates = data['Results']['series'][0]['data']
            df_unemployment = pd.DataFrame(unemployment_rates)
            df_unemployment = df_unemployment.rename(columns={'value': 'UnemploymentRate', 'year': 'year'})
            df_unemployment['UnemploymentRate'] = df_unemployment['UnemploymentRate'].astype(float)
            df_unemployment['Trend'] = np.where(df_unemployment['UnemploymentRate'].diff() > 0, 'Rising', 'Falling')
            df_unemployment.loc[0, 'Trend'] = 'Stable'
            
            print(f"Main df shape: {self.df_main.shape}")
            print(f"Unemployment df shape: {df_unemployment.shape}")
            print(f"Unemployment df columns: {df_unemployment.columns.tolist()}")

            # Correct merge operation
            # The 'JoinYear' in the main DataFrame needs to be merged with the 'year' column in the unemployment data.
            self.df_main = pd.merge(self.df_main, df_unemployment, left_on='JoinYear', right_on='year', how='left')
            
            # Remove the now-redundant 'year' column from the merged DataFrame.
            self.df_main.drop('year', axis=1, inplace=True)

            print("Merging unemployment data...")

        except FileNotFoundError:
            print(f"Warning: The file '{self.json_path}' was not found. Skipping unemployment data processing.")
        except Exception as e:
            print(f"Error processing unemployment data: {e}. Skipping merge.")
            
    def create_join_and_leave_years(self) -> None:
        """Derives 'JoinYear' and 'LeaveYear' from existing data."""
        print("Creating join and leave years...")
        self.df_main['JoinYear'] = 2020 - self.df_main['YearsAtCompany']
        self.df_main['LeaveYear'] = 2020 # Assuming data is from 2020
    
    def apply_categorical_mappings(self) -> None:
        """Applies human-readable labels to categorical features."""
        print("Applying categorical mappings...")
        for col, mapping in self.categorical_mappings.items():
            if col in self.df_main.columns:
                self.df_main[col] = self.df_main[col].map(mapping)
                # Handle cases where mapping might produce NaNs
                # For safety, we can fill these or let them be for now.
    
    def remove_unnecessary_columns(self) -> None:
        """Removes columns that are constant or not useful for analysis."""
        cols_to_remove = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
        print(f"Removing columns: {cols_to_remove}")
        self.df_main.drop(columns=cols_to_remove, inplace=True, errors='ignore')
    
    def save_processed_data(self) -> None:
        """Saves the final processed DataFrame to a new CSV file."""
        if self.df_main is not None:
            self.df_main.to_csv(PROCESSED_FILE_PATH, index=False)
            print(f"\nProcessed data saved to {PROCESSED_FILE_PATH}")
    
    def get_data_summary(self) -> Dict:
        """Generates a summary of the processed DataFrame."""
        summary = {}
        if self.df_main is not None:
            summary['shape'] = self.df_main.shape
            summary['columns'] = self.df_main.columns.tolist()
            summary['data_types'] = self.df_main.dtypes.to_dict()
            summary['missing_values'] = self.df_main.isnull().sum().reset_index()
            summary['missing_values'].columns = ['feature', 'Total_Missing']
            summary['numeric_features'] = self.df_main.select_dtypes(include=np.number).columns.tolist()
            summary['categorical_features'] = self.df_main.select_dtypes(include='object').columns.tolist()
        return summary
        
    def run_complete_pipeline(self, include_unemployment: bool = True) -> Optional[pd.DataFrame]:
        """Runs the entire data processing pipeline."""
        try:
            print("=" * 60)
            print("Starting HR Attrition Data Processing Pipeline")
            print("=" * 60)
            
            # 1. Load data
            self.load_ibm_data()
            if self.df_main.empty:
                return None
            
            # 2. Generate and add new features
            self.generate_synthetic_survey_data()
            self.generate_text_survey_data()
            self.create_join_and_leave_years()
            
            # 3. Process and merge external data
            if include_unemployment:
                self.process_unemployment_data()
            
            # 4. Apply transformations
            self.apply_categorical_mappings()
            
            # 5. Clean and finalize
            self.remove_unnecessary_columns()

            # Final check and summary
            summary = self.get_data_summary()
            print(f"\nFinal dataset shape: {summary['shape']}")
            print(f"Categorical features: {len(summary['categorical_features'])}")
            
            # 6. Save processed data
            self.save_processed_data()
            
            print("\n" + "=" * 60)
            print("Pipeline completed successfully!")
            print("=" * 60)
            
            return self.df_main
            
        except Exception as e:
            print(f"Error in pipeline: {str(e)}")
            raise


if __name__ == "__main__":
    # The processor no longer needs to be initialized with a path.
    processor = HRDataProcessor()
    
    # Run complete pipeline
    print("Starting HR Attrition Data Processing Pipeline")

    # Run complete pipeline
    processed_df = processor.run_complete_pipeline(include_unemployment=True)
    
    #processed_df = processor.load_ibm_data()

    
    if processed_df is not None:
        # Display results
        print("\nFirst 5 rows of processed data:")
        print(processed_df.head())
        
        print(f"\nDataset info:")
        print(f"Shape: {processed_df.shape}")
        print(f"Columns: {list(processed_df.columns)}")
        """
        if not processed_df.empty:
            merged_unemployment_df = processor.process_unemployment_data(processed_df)
            print(f"Shape: {merged_unemployment_df.shape}")
            print(f"Columns: {list(merged_unemployment_df.columns)}")
        else:
         print("JSON file could not be loaded. Skipping processing...")
        """
