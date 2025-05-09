from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load your model (replace 'model.pkl' with your model file path)
model = joblib.load('model.pkl')  # Ensure 'model.pkl' is in the correct location
model_columns = joblib.load('model_columns.pkl')  # Load model columns used during training

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data sent by the frontend
        data = request.get_json()
        
        # Extract features from the 'input' key (the frontend should send input data in this structure)
        user_input = data.get('input')
        
        if user_input is None:
            return jsonify({'error': 'No input data found'}), 400
        
        # Ensure that the input data has all the columns the model expects
        input_df = pd.DataFrame([user_input])

        # One-hot encode the input (to match the model's expected format)
        input_df = pd.get_dummies(input_df)

        # Add missing columns (if any) and reorder to match the training data's columns
        for col in model_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[model_columns]

        # Make the prediction
        prediction = model.predict(input_df)[0]  # Get the prediction from the model
        
        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
