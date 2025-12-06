import json
import joblib
import numpy as np
import os


def init():
    """
    Called once when the service is started. Loads the trained model into memory.
    Azure ML will call this function to initialise the endpoint.
    """
    global model
    # Determine which model file to load, defaulting to the random forest
    model_file = os.getenv('MODEL_FILE', 'random_forest.joblib')
    model_dir = os.getenv('AZUREML_MODEL_DIR', 'models')
    model_path = os.path.join(model_dir, model_file)
    model = joblib.load(model_path)


def run(raw_data):
    """
    Makes a prediction using the trained model. Expects `raw_data` to be a JSON
    string representing a dictionary of feature values. Returns the prediction as
    a JSON-serialisable dict.
    """
    try:
        # Parse the input JSON string
        data = json.loads(raw_data)
        # Build feature array. Ensure keys match the feature columns used in training.
        features = np.array([[
            data['repo_duration_scaled'],
            data['commits'],
            data['branches'],
            data['contributors'],
            data['stars'],
            data['issues'],
            data['pullrequests'],
            data['size'],
            data['language_encoded']
        ]])
        # Generate the prediction
        prediction = model.predict(features)[0]
        return {'prediction': int(prediction)}
    except Exception as e:
        # Return the error message to facilitate debugging
        return {'error': str(e)}