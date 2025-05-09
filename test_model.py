import joblib
import pandas as pd

# Load model and feature columns
model = joblib.load('model.pkl')
model_columns = joblib.load('model_columns.pkl')

# Risky customer (likely to churn)
sample = {
    'gender': 'Male',
    'SeniorCitizen': 1,
    'Partner': 'No',
    'Dependents': 'No',
    'tenure': 1,
    'MonthlyCharges': 120,
    'TotalCharges': 100,
    'InternetService': 'Fiber optic',
    'Contract': 'Month-to-month'
}

# Convert to DataFrame
df = pd.DataFrame([sample])
df = pd.get_dummies(df)

# Add missing columns
for col in model_columns:
    if col not in df.columns:
        df[col] = 0
df = df[model_columns]

# Predict
result = model.predict(df)[0]
print("Prediction:", result)  # 1 = churn, 0 = no churn
