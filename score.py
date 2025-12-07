import os
import json
import joblib
import numpy as np

model = None

def init():
    """Load the model from the Azure ML model directory."""
    global model
    # Azure sets AZUREML_MODEL_DIR to something like:
    # /var/azureml-app/azureml-models/<model_name>/<version>
    model_dir = os.getenv("AZUREML_MODEL_DIR", ".")
    model_path = os.path.join(model_dir, "random_forest.joblib")
    model = joblib.load(model_path)


def run(request):
    """
    Azure ML may pass either:
      - a JSON string, or
      - a Python dict.

    We handle both, and we accept either:
      {"data": [ {feature dict} ]}  or  {feature dict}
    """
    try:
        # If request is a JSON string, parse it
        if isinstance(request, str):
            payload = json.loads(request)
        else:
            payload = request

        # Support both {"data":[{...}]} and just {...}
        if "data" in payload:
            row = payload["data"][0]
        else:
            row = payload

        # Extract features in the SAME order used during training
        features = np.array([[
            row["repo_duration_scaled"],
            row["commits"],
            row["branches"],
            row["contributors"],
            row["stars"],
            row["issues"],
            row["pullrequests"],
            row["size"],
            row["language_encoded"],
        ]])

        pred = int(model.predict(features)[0])
        return {"prediction": pred}

    except Exception as e:
        # Return error string so it appears in logs / test UI
        return {"error": str(e)}
