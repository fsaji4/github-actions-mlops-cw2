import pytest
from prepare_data import prepare_data


def test_prepare_data(tmp_path):
    """
    Verify that prepare_data() returns a non-empty DataFrame and includes
    the expected encoded outcome column. The test uses a temporary output
    path to avoid overwriting real data during the test.
    If the raw data files are missing, the test is skipped gracefully.
    """
    try:
        # Use temporary directory for output to avoid collisions
        output_path = tmp_path / 'temp_clean.csv'
        df = prepare_data(
            repositories_path='data/repositories.csv',
            workflows_path='data/workflows.csv',
            auxiliaries_path='data/workflows_auxiliaries.csv',
            output_path=str(output_path)
        )
        assert not df.empty, "Cleaned dataframe should not be empty"
        assert 'conclusion_encoded' in df.columns
    except FileNotFoundError:
        pytest.skip('Raw data files not found; skipping prepare_data test')