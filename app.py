from flask import Flask, request, render_template, jsonify
import numpy as np
import joblib

# Load the pre-trained Logistic Regression model (use your trained model)
model = joblib.load('rf_model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get JSON data from frontend

    # Extract features from the form
    features = np.array([[
        data['age'], data['sex'], data['cp'], data['trestbps'],
        data['chol'], data['fbs'], data['restecg'], data['thalach'],
        data['exang'], data['oldpeak'], data['slope'], data['ca'], data['thal']
    ]])

    # Make prediction using the model
    prediction = model.predict(features)

    # Return the result as JSON
    return jsonify({
        'prediction': int(prediction[0]),
        'message': 'Heart disease detected in You' if prediction[0] == 1 else 'No heart disease'
    })

if __name__ == '__main__':
    app.run(debug=True)