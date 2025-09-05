# =============================================================================
# HR Attrition Data Pipeline - EDA with some cool stuff!
# =============================================================================

import os,sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# To import modules from the src directory
script_path = Path(__file__).resolve()
src_path = script_path.parent.parent
sys.path.append(str(src_path))
from data.hr_data_pipeline import HRDataProcessor

# Define the file path 
# Adjust the number of `.parent` calls to match the folder structure.
script_path = Path(__file__).resolve()
project_dir= script_path.parent.parent.parent
FILE_PATH = project_dir /'data'/'raw'/ 'HRAttrition.csv'
PROCESSED_FILE_PATH= project_dir /'data'/'raw'/ 'HRAttrition_Processed.csv'


class AdvancedEDA:
    """
    Advanced Exploratory Data Analysis class for comprehensive dataset exploration
    """
    
    def __init__(self, data_path=None, df=None):
        """
        Initialize EDA with either a file path or DataFrame
        
        Parameters:
        data_path (str): Path to the dataset file
        df (pd.DataFrame): Pre-loaded DataFrame
        """
        if df is not None:
            self.df = df.copy()
        elif data_path is not None:
            self.df = self.load_data(data_path)
        else:
            raise ValueError("Either data_path or df must be provided")
        # Crucial check: if the DataFrame is not loaded, raise an error
        if self.df is None:
            raise ValueError("Data loading failed. Check the file path and file contents.")
            
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_cols = self.df.select_dtypes(include=['datetime64']).columns.tolist()
    
    def load_data(self, data_path):
        """Load data from various file formats"""
        try:
            if data_path.endswith('.csv'):
                return pd.read_csv(data_path)
            elif data_path.endswith('.xlsx') or data_path.endswith('.xls'):
                return pd.read_excel(data_path)
            elif data_path.endswith('.json'):
                return pd.read_json(data_path)
            elif data_path.endswith('.parquet'):
                return pd.read_parquet(data_path)
            else:
                raise ValueError("Unsupported file format")
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def dataset_overview(self):
        """Comprehensive dataset overview"""
        print("="*60)
        print("DATASET OVERVIEW")
        print("="*60)
        
        print(f"Dataset Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        print(f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print()
        
        print("Column Information:")
        print("-" * 40)
        for i, column in enumerate(self.df.columns, 1):
            dtype = self.df[column].dtype
            null_count = self.df[column].isnull().sum()
            null_pct = (null_count / len(self.df)) * 100
            unique_count = self.df[column].nunique()
            
            print(f"{i:2d}. {column:<25} | {str(dtype):<12} | "
                  f"Nulls: {null_count:>5} ({null_pct:>5.1f}%) | "
                  f"Unique: {unique_count:>5}")
        
        print()
        print(f"Numeric columns ({len(self.numeric_cols)}): {', '.join(self.numeric_cols)}")
        print(f"Categorical columns ({len(self.categorical_cols)}): {', '.join(self.categorical_cols)}")
        if self.datetime_cols:
            print(f"DateTime columns ({len(self.datetime_cols)}): {', '.join(self.datetime_cols)}")
    
    def missing_data_analysis(self):
        """Analyze missing data patterns"""
        print("\n" + "="*60)
        print("MISSING DATA ANALYSIS")
        print("="*60)
        
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing_Count': missing_data.values,
            'Missing_Percentage': missing_percent.values
        }).sort_values('Missing_Count', ascending=False)
        
        missing_df = missing_df[missing_df['Missing_Count'] > 0]
        
        if len(missing_df) > 0:
            print("Columns with missing values:")
            print(missing_df.to_string(index=False))
            
            # Visualize missing data pattern
            plt.figure(figsize=(12, 6))
            
            # Missing data heatmap
            plt.subplot(1, 2, 1)
            sns.heatmap(self.df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
            plt.title('Missing Data Pattern')
            plt.xlabel('Columns')
            
            # Missing data bar plot
            plt.subplot(1, 2, 2)
            missing_df.plot(x='Column', y='Missing_Percentage', kind='bar', ax=plt.gca())
            plt.title('Missing Data Percentage by Column')
            plt.xticks(rotation=45)
            plt.ylabel('Missing Percentage (%)')
            
            plt.tight_layout()
            plt.show()
        else:
            print("No missing values found in the dataset!")
    
    def numeric_analysis(self):
        """Analyze numeric columns"""
        if not self.numeric_cols:
            print("\nNo numeric columns found!")
            return
            
        print("\n" + "="*60)
        print("NUMERIC COLUMNS ANALYSIS")
        print("="*60)
        
        # Descriptive statistics
        print("Descriptive Statistics:")
        desc_stats = self.df[self.numeric_cols].describe()
        print(desc_stats)
        
        # Additional statistics
        print("\nAdditional Statistics:")
        additional_stats = pd.DataFrame({
            'Skewness': self.df[self.numeric_cols].skew(),
            'Kurtosis': self.df[self.numeric_cols].kurtosis(),
            'CV (%)': (self.df[self.numeric_cols].std() / self.df[self.numeric_cols].mean()) * 100
        })
        print(additional_stats)
        
        # Outlier detection using IQR
        print("\nOutlier Analysis (IQR Method):")
        outlier_summary = []
        for col in self.numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outlier_summary.append({
                'Column': col,
                'Outlier_Count': len(outliers),
                'Outlier_Percentage': (len(outliers) / len(self.df)) * 100,
                'Lower_Bound': lower_bound,
                'Upper_Bound': upper_bound
            })
        
        outlier_df = pd.DataFrame(outlier_summary)
        print(outlier_df.to_string(index=False))
    
    def univariate_analysis(self):
        """Performs and visualizes univariate analysis for all columns."""
        print("\n--- Univariate Analysis ---")
        
        # Numeric columns
        print("\nNumeric Column Summaries:")
        print(self.df[self.numeric_cols].describe().T)
        
        # Categorical columns
        print("\nCategorical Column Counts:")
        for col in self.categorical_cols:
            print(f"\nValue counts for '{col}':")
            print(self.df[col].value_counts())
            
    def bivariate_analysis(self):
        """Performs bivariate analysis and visualizations."""
        print("\n--- Bivariate Analysis ---")
        # Numeric vs Numeric
        print("\nNumeric vs Numeric Bivariate Analysis:")
        for i in range(len(self.numeric_cols)):
            for j in range(i + 1, len(self.numeric_cols)):
                col1, col2 = self.numeric_cols[i], self.numeric_cols[j]
                corr = self.df[[col1, col2]].corr().iloc[0, 1]
                print(f"Correlation between {col1} and {col2}: {corr:.2f}")

        # Categorical vs Categorical
        print("\nCategorical vs Categorical Bivariate Analysis:")
        for i in range(len(self.categorical_cols)):
            for j in range(i + 1, len(self.categorical_cols)):
                col1, col2 = self.categorical_cols[i], self.categorical_cols[j]
                
                # Gracefully handle columns with no data or insufficient unique values
                if self.df[col1].nunique() > 1 and self.df[col2].nunique() > 1:
                    try:
                        contingency_table = pd.crosstab(self.df[col1], self.df[col2])
                        # The chi-square test requires a non-empty contingency table.
                        if not contingency_table.empty:
                            chi2, p, _, _ = stats.chi2_contingency(contingency_table)
                            print(f"Chi-square test for {col1} and {col2}: chi2={chi2:.2f}, p-value={p:.3f}")
                        else:
                            print(f"Skipping chi-square test for {col1} and {col2} due to empty contingency table.")
                    except ValueError as e:
                        print(f"Skipping chi-square test for {col1} and {col2}. Error: {e}")
                else:
                    print(f"Skipping chi-square test for {col1} and {col2} due to insufficient unique values.")
    def distribution_plots(self):
        """Generates distribution plots for numeric and categorical columns."""
        print("\n--- Generating Distribution Plots ---")
        
        # Histograms for numeric columns
        self.df[self.numeric_cols].hist(figsize=(15, 10))
        plt.suptitle('Histograms of Numeric Features', y=1.02)
        plt.tight_layout()
        plt.show()

        # Count plots for categorical columns
        for col in self.categorical_cols:
            plt.figure(figsize=(10, 6))
            sns.countplot(data=self.df, x=col)
            plt.title(f'Count Plot of {col}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()



    def categorical_analysis(self):
        """Analyze categorical columns"""
        if not self.categorical_cols:
            print("\nNo categorical columns found!")
            return
            
        print("\n" + "="*60)
        print("CATEGORICAL COLUMNS ANALYSIS")
        print("="*60)
        
        for col in self.categorical_cols:
            print(f"\n{col}:")
            print("-" * (len(col) + 1))
            
            value_counts = self.df[col].value_counts()
            print(f"Unique values: {self.df[col].nunique()}")
            print(f"Most frequent: '{value_counts.index[0]}' ({value_counts.iloc[0]} times)")
            
            if len(value_counts) <= 20:
                print("\nValue counts:")
                for idx, (value, count) in enumerate(value_counts.head(10).items()):
                    percentage = (count / len(self.df)) * 100
                    print(f"  {value}: {count} ({percentage:.1f}%)")
                if len(value_counts) > 10:
                    print(f"  ... and {len(value_counts) - 10} more categories")
            else:
                print(f"Too many categories to display ({len(value_counts)} unique values)")
    
    def correlation_analysis(self):
        """Analyze correlations between numeric variables"""
        if len(self.numeric_cols) < 2:
            print("\nInsufficient numeric columns for correlation analysis!")
            return
            
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS")
        print("="*60)
        
        corr_matrix = self.df[self.numeric_cols].corr()
        
        # Find strong correlations
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:  # Strong correlation threshold
                    strong_corr.append({
                        'Variable_1': corr_matrix.columns[i],
                        'Variable_2': corr_matrix.columns[j],
                        'Correlation': corr_val
                    })
        
        if strong_corr:
            print("Strong correlations (|r| > 0.5):")
            strong_corr_df = pd.DataFrame(strong_corr).sort_values('Correlation', key=abs, ascending=False)
            print(strong_corr_df.to_string(index=False))
        else:
            print("No strong correlations found (|r| > 0.5)")
        
        # Visualize correlation matrix
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    def data_quality_report(self):
        """Generate comprehensive data quality report"""
        print("\n" + "="*60)
        print("DATA QUALITY REPORT")
        print("="*60)
        
        quality_issues = []
        
        # Check for duplicates
        duplicate_count = self.df.duplicated().sum()
        if duplicate_count > 0:
            quality_issues.append(f"Found {duplicate_count} duplicate rows")
        
        # Check for columns with single value
        single_value_cols = []
        for col in self.df.columns:
            if self.df[col].nunique() == 1:
                single_value_cols.append(col)
        
        if single_value_cols:
            quality_issues.append(f"Columns with single value: {', '.join(single_value_cols)}")
        
        # Check for high cardinality categorical columns
        high_cardinality_cols = []
        for col in self.categorical_cols:
            cardinality_ratio = self.df[col].nunique() / len(self.df)
            if cardinality_ratio > 0.9:
                high_cardinality_cols.append(f"{col} ({cardinality_ratio:.1%})")
        
        if high_cardinality_cols:
            quality_issues.append(f"High cardinality categorical columns: {', '.join(high_cardinality_cols)}")
        
        # Check for potential data type issues
        potential_numeric = []
        for col in self.categorical_cols:
            try:
                pd.to_numeric(self.df[col], errors='raise')
                potential_numeric.append(col)
            except:
                pass
        
        if potential_numeric:
            quality_issues.append(f"Categorical columns that might be numeric: {', '.join(potential_numeric)}")
        
        if quality_issues:
            print("Potential data quality issues:")
            for i, issue in enumerate(quality_issues, 1):
                print(f"{i}. {issue}")
        else:
            print("No major data quality issues detected!")
    
    def run_full_eda(self):
        """Runs the complete EDA pipeline."""
        print("="*60)
        print("Starting Comprehensive EDA Analysis")
        print("="*60)
        
        self.univariate_analysis()
        self.bivariate_analysis()
        self.distribution_plots()
        self.correlation_analysis()
        self.data_quality_report()
        
        print("\n" + "="*60)
        print("EDA ANALYSIS COMPLETE")
        print("="*60)

# Usage examples:
def main():
    """
    Example usage of the AdvancedEDA class
    """
    
    try:
        # Step 1: Run the data pipeline to generate the processed DataFrame
        print("Running data processing pipeline to generate processed data...")
        processor = HRDataProcessor()
        processed_df = processor.run_complete_pipeline()

        # Step 2: Pass the DataFrame directly to the EDA class
        print("\nStarting EDA on the processed DataFrame...")
        eda = AdvancedEDA(df=processed_df)
        eda.run_full_eda()
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()