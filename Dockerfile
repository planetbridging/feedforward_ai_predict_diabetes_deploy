FROM tensorflow/tensorflow:latest

# Install Flask
RUN pip install Flask pandas scikit-learn

# Copy the app and models
COPY app.py /app/
COPY model_diabetes /app/model_diabetes
COPY model_hypertension /app/model_hypertension
COPY model_heart_disease /app/model_heart_disease

# Set the working directory
WORKDIR /app

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]
