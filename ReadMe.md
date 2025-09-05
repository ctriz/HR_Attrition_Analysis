# Technical Architecture: HR Attrition Analysis Pipeline

## Executive Summary

This project demonstrates a full-stack HR Attrition Analysis pipeline. It integrates heterogeneous data sources, applies fairness-aware machine learning, and deploys models as scalable APIs. Beyond prediction, it emphasizes **bias mitigation, automated feature engineering, and trade-off analysis** between accuracy and fairness — making it a practical case study in responsible AI systems.

## 1. Data Collection & Integration

-   **Sources**: IBM HR Attrition dataset, synthetic employee surveys (Likert & sentiment), and external labor market data (BLS unemployment).
    
-   **Integration**: Data aligned on employee identifiers and temporal keys, producing a unified dataset enriched with demographic, behavioral, and macroeconomic features.
    

## 2. Data Preprocessing

-   Schema and range validation for demographics, income, and tenure.
    
-   Standardization of categorical encodings (education, satisfaction, work-life balance).
    
-   Ensures data integrity before downstream analytics.
    

## 3. Exploratory Data Analysis

-   Correlation and multicollinearity checks for numeric features.
    
-   Categorical association analysis (chi-square, Cramér’s V).
    
-   Insights used to guide feature selection and model choice.
    

## 4. Feature Engineering

-   **Automated Features**: Generated via Featuretools (DFS, depth=1) to capture interactions and ratios.
    
-   **Demographic Simulation**: Adds synthetic attributes for bias sensitivity testing.
    
-   **Encoding**: Categorical variables one-hot encoded with scalability for new categories.
    

## 5. Bias & Fairness Analysis

-   **Protected Attribute**: Gender.
    
-   **Fairness Metrics**: Demographic parity, equalized odds, group-level accuracy/AUC.
    
-   **Mitigation**:
    
    -   Pre-processing: Data rebalancing and debiasing transforms.
        
    -   In-processing: Fairlearn’s ExponentiatedGradient with DemographicParity constraints.
        
-   Clear evaluation of fairness–accuracy trade-offs.
    

## 6. Feature Selection

-   Lasso regularization for automatic feature reduction.
    
-   Handles multicollinearity, improves interpretability, reduces overfitting risk.
    

## 7. Model Training

-   **Portfolio**: Random Forest, Logistic Regression, XGBoost, SVC.
    
-   **Cross-Validation**: Stratified k-fold (k=5) to preserve class balance.
    
-   **Dual Training**: Standard vs. fairness-constrained models for side-by-side evaluation.
    

## 8. Evaluation

-   **Performance**: Accuracy, ROC AUC, Precision/Recall.
    
-   **Fairness**: Demographic parity difference, equalized odds, subgroup analysis.
    
-   **Trade-offs**: Systematically compared to highlight where accuracy and fairness diverge.
    

## 9. Hyperparameter Optimization

-   GridSearchCV tuned for ROC AUC.
    
-   Post-optimization fairness-aware retraining ensures final models are both performant and equitable.
    

## 10. Deployment

-   **Persistence**: Model, scaler, and encoder serialized.
    
-   **API**: Flask-based REST service with JSON I/O, container-ready.
    
-   **Scalability**: Supports Docker deployment; low-latency predictions.
    

## 11. Visualization & Monitoring

-   Interactive dashboard (Plotly + Tailwind) for:
    
    -   Model performance (ROC, accuracy, F1)
        
    -   Fairness & bias metrics
        
    -   Feature importance and correlations
        
    -   Model comparison & trade-off radar charts
        

## 12. Technical Stack

-   **Data Processing**: Pandas, NumPy
    
-   **Feature Engineering**: Featuretools
    
-   **ML & Fairness**: Scikit-learn, XGBoost, Fairlearn
    
-   **Visualization**: Matplotlib, Plotly
    
-   **Deployment**: Flask, Docker
    

## 13. Pipeline Summary

```
Raw + Synthetic + Market Data → Preprocessing → Feature Engineering →
Bias-Aware ML Training → Evaluation (Performance + Fairness) →
Deployment as API → Dashboard Monitoring

```

----------

This architecture illustrates not just attrition prediction but also **responsible ML design** — integrating fairness, explainability, and deployment-readiness into a single reproducible pipeline.
