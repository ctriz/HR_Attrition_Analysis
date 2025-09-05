# =============================================================================
# HR Attrition Prediction API with Flask
# This script creates a simple API endpoint to predict employee attrition.
# =============================================================================

import os
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- 1. SET UP THE FLASK APPLICATION AND CORS ---
# CORS is needed to allow requests from your UI, which will be on a different origin.
app = Flask(__name__)
CORS(app)

# --- 2. SIMULATE MODEL TRAINING (IN A REAL SCENARIO, YOU'D LOAD A SAVED MODEL) ---
# In a production environment, you would train your model separately and save it,
# then load it here to avoid re-training on every startup.
print("Training a dummy model for demonstration...")

# Create a sample dataset similar to the one you'd get from your pipeline
data = {
    'YearsAtCompany': [5, 1, 10, 2, 7, 15, 3, 20, 1],
    'JobSatisfaction': [4, 1, 3, 2, 4, 1, 3, 4, 2],
    'MonthlyIncome': [8000, 2500, 12000, 3500, 9500, 20000, 4000, 15000, 3000],
    'Attrition': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes']
}
df = pd.DataFrame(data)

# Preprocessing: encode the target variable
le = LabelEncoder()
df['Attrition'] = le.fit_transform(df['Attrition'])

# Define features and target
features = ['YearsAtCompany', 'JobSatisfaction', 'MonthlyIncome']
target = 'Attrition'

X = df[features]
y = df[target]

# Train a simple model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

print("Dummy model trained and ready.")


# --- 3. CREATE THE PREDICTION API ENDPOINT ---
# This function will be called whenever a POST request is made to '/predict'.
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request body
        request_data = request.get_json(force=True)

        # Check for required input parameters
        required_params = ['YearsAtCompany', 'JobSatisfaction', 'MonthlyIncome']
        for param in required_params:
            if param not in request_data:
                return jsonify({"error": f"Missing parameter: {param}"}), 400

        # Convert the incoming JSON data to a pandas DataFrame, matching the
        # format of the data the model was trained on.
        input_df = pd.DataFrame([request_data], columns=required_params)

        # Make a prediction and get the probability
        prediction = model.predict(input_df)[0]
        # The probability of the 'Yes' class (class 1)
        probability = model.predict_proba(input_df)[0][1] 

        # Map the numeric prediction back to 'Yes' or 'No'
        attrition_result = le.inverse_transform([prediction])[0]
        
        # Return the prediction and probability as a JSON object
        return jsonify({
            "prediction": attrition_result,
            "probability": round(probability, 4)
        })

    except Exception as e:
        # Return an error message if something goes wrong
        return jsonify({"error": str(e)}), 500


# --- 4. RUN THE FLASK DEVELOPMENT SERVER ---
if __name__ == '__main__':
    # The 'host=0.0.0.0' allows the server to be accessible from outside the container/localhost
    # This is important for public demo services.
    app.run(host='0.0.0.0', port=5000, debug=True)
