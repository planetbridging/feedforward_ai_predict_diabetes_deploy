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

@app.route('/predicthtml', methods=['GET'])
def predicthtml():
    try:
        # Check if the form has been submitted
        if request.args:
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

            # Convert predictions to percentages
            diabetes_percentage = diabetes_prediction[0][0] * 100
            hypertension_percentage = hypertension_prediction[0][0] * 100
            heart_disease_percentage = heart_disease_prediction[0][0] * 100

            # Create the results table with borders
            results_table = f"""
            <table class="table" style="border: 1px solid black;">
                <thead>
                    <tr>
                        <th>Condition</th>
                        <th>Prediction</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Diabetes</td>
                        <td>{diabetes_percentage:.2f}% chance</td>
                    </tr>
                    <tr>
                        <td>Hypertension</td>
                        <td>{hypertension_percentage:.2f}% chance</td>
                    </tr>
                    <tr>
                        <td>Heart Disease</td>
                        <td>{heart_disease_percentage:.2f}% chance</td>
                    </tr>
                </tbody>
            </table>
            """
        else:
            results_table = ""

        # Create the form
        form_html = f"""
        <form action="/predicthtml" method="get">
            <div class="form-group">
    <label for="age">Age:</label>
    <input type="number" class="form-control" name="age" min="0" max="150" required>
</div>

            <div class="form-group">
    <label for="bmi">BMI:</label>
    <input type="number" step="0.1" class="form-control" name="bmi" min="0" max="200" required>
</div>

           <div class="form-group">
    <label for="HbA1c_level">HbA1c Level:</label>
    <input type="number" step="0.1" class="form-control" name="HbA1c_level" min="0" max="20" required>
</div>

            <div class="form-group">
                <label for="blood_glucose_level">Blood Glucose Level:</label>
                <input type="text" class="form-control" name="blood_glucose_level" required>
            </div>
           <div class="form-group">
    <label for="smoking_history">Smoking History:</label>
    <select class="form-control" name="smoking_history" required>
        <option value="never">never</option>
        <option value="No Info">No Info</option>
        <option value="current">current</option>
        <option value="former">former</option>
        <option value="ever">ever</option>
        <option value="not current">not current</option>
    </select>
</div>

            <button type="submit" class="btn btn-primary">Predict</button>
        </form>
        {results_table}
        """

        return form_html
    except Exception:
        # You can return a custom error message or status code if desired
        return jsonify({'error': 'An error occurred during prediction'}), 500

# Existing /predict route remains unchanged

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
