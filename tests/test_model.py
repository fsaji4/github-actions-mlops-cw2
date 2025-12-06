import pytest
import joblib
import os
from train_model import train_and_select_model


def test_train_model(tmp_path):
    """
    Ensure that training returns a valid model file and that the saved
    model can be loaded. If the cleaned dataset is missing, the test is
    skipped gracefully.
    """
    try:
        # Use a temporary directory for model output
        models_dir = tmp_path / 'models'
        models_dir.mkdir(exist_ok=True)
        model_name = train_and_select_model(
            data_path='data/cleaned_github_actions.csv',
            models_dir=str(models_dir)
        )
        model_path = models_dir / model_name
        model = joblib.load(model_path)
        assert model is not None, 'Loaded model should not be None'
    except FileNotFoundError:
        pytest.skip('Cleaned dataset not found; skipping train_model test')