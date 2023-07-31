from flask import Flask, request, jsonify
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

# Load the models
model_diabetes = tf.keras.models.load_model('model_diabetes')
model_hypertension = tf.keras.models.load_model('model_hypertension')
model_heart_disease = tf.keras.models.load_model('model_heart_disease')

# Manually recreate the encoders and scaler
data = pd.read_csv('diabetes_prediction_dataset_binary.csv')  # Load the same data used during training
smoking_encoder = LabelEncoder()
data['smoking_history'] = smoking_encoder.fit_transform(data['smoking_history'])  # Encode the smoking_history column
scaler = StandardScaler()
scaler.fit(data[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'smoking_history']])


@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Get the data from the request
        age = request.args.get('age')
        bmi = request.args.get('bmi')
        HbA1c_level = request.args.get('HbA1c_level')
        blood_glucose_level = request.args.get('blood_glucose_level')
        smoking_history = request.args.get('smoking_history')

        # Preprocess the data
        new_data = pd.DataFrame({
            'age': [age],
            'bmi': [bmi],
            'HbA1c_level': [HbA1c_level],
            'blood_glucose_level': [blood_glucose_level],
            'smoking_history': [smoking_history]
        })
        new_data['smoking_history'] = smoking_encoder.transform(new_data['smoking_history'])
        new_data = scaler.transform(new_data)

        # Make predictions
        diabetes_prediction = model_diabetes.predict(new_data)
        hypertension_prediction = model_hypertension.predict(new_data)
        heart_disease_prediction = model_heart_disease.predict(new_data)

        # Return the predictions as JSON
        return jsonify({
            'diabetes': diabetes_prediction.tolist(),
            'hypertension': hypertension_prediction.tolist(),
            'heart_disease': heart_disease_prediction.tolist()
        })
    except Exception:
        # You can return a custom error message or status code if desired
        return jsonify({'error': 'An error occurred during prediction'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
