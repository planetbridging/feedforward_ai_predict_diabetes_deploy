const tf = require("@tensorflow/tfjs");
const path = require("path");
const fs = require("fs");

// Function to encode smoking history
function encodeSmokingHistory(smokingHistory) {
  const smokingEncoder = {
    never: [1, 0, 0, 0, 0],
    current: [0, 1, 0, 0, 0],
    "no info": [0, 0, 1, 0, 0],
    former: [0, 0, 0, 1, 0],
    "not current": [0, 0, 0, 0, 1],
  };

  return smokingEncoder[smokingHistory.toLowerCase()] || [0, 0, 0, 0, 0]; // Default to [0, 0, 0, 0, 0] if smoking history is not recognized
}

// Function to preprocess user input
function preprocessInput(userInput) {
  const user_data = { ...userInput }; // Assuming userInput is an object containing age, bmi, HbA1c_level, blood_glucose_level, and smoking_history
  user_data.smoking_history = encodeSmokingHistory(user_data.smoking_history);
  return user_data;
}

// Function to load the models
async function loadModels() {
  const model_diabetes_path = path.resolve(
    __dirname,
    "model_diabetes_js/model.json"
  );
  const model_hypertension_path = path.resolve(
    __dirname,
    "model_hypertension_js/model.json"
  );
  const model_heart_disease_path = path.resolve(
    __dirname,
    "model_heart_disease_js/model.json"
  );

  const model_diabetes = await tf.loadGraphModel(model_diabetes_path);
  const model_hypertension = await tf.loadGraphModel(model_hypertension_path);
  const model_heart_disease = await tf.loadGraphModel(model_heart_disease_path);

  return { model_diabetes, model_hypertension, model_heart_disease };
}

// Function to make predictions
async function predict(userInput) {
  const { model_diabetes, model_hypertension, model_heart_disease } =
    await loadModels();

  const user_data = preprocessInput(userInput);
  const inputTensor = tf.tensor2d(
    [Object.values(user_data)],
    [1, Object.values(user_data).length]
  );

  // Make predictions
  const diabetes_prediction = model_diabetes.predict(inputTensor);
  const hypertension_prediction = model_hypertension.predict(inputTensor);
  const heart_disease_prediction = model_heart_disease.predict(inputTensor);

  // Get the probabilities or binary predictions based on the threshold, if needed
  // Example: diabetes_prediction = diabetes_prediction.arraySync()[0][0] > 0.5 ? 1 : 0;

  return {
    diabetes: diabetes_prediction.arraySync()[0][0],
    hypertension: hypertension_prediction.arraySync()[0][0],
    heart_disease: heart_disease_prediction.arraySync()[0][0],
  };
}

module.exports = { predict };
