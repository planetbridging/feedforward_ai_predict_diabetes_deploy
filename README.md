# Health Prediction API

This repository contains a Flask application that serves a machine learning model to predict the likelihood of diabetes, hypertension, and heart disease based on various health parameters.

## Models

The application uses three pre-trained TensorFlow models for predicting:
- Diabetes
- Hypertension
- Heart Disease

## Features

The models take the following features as input:
- Age
- BMI (Body Mass Index)
- HbA1c Level
- Blood Glucose Level
- Smoking History (encoded as 'never', 'former', etc.)

## Endpoint

The application exposes a single endpoint `/predict` that accepts GET requests with the above features as query parameters.
