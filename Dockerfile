FROM tensorflow/tensorflow:latest

# Install Flask, pandas, and scikit-learn
RUN pip install Flask pandas scikit-learn

# Copy the app, models, and CSV file
COPY app.py /app/
COPY model_diabetes /app/model_diabetes
COPY model_hypertension /app/model_hypertension
COPY model_heart_disease /app/model_heart_disease
COPY diabetes_prediction_dataset_binary.csv /app/

# Set the working directory
WORKDIR /app

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]
