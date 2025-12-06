import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import os


def prepare_data(
    repositories_path='data/repositories.csv',
    workflows_path='data/workflows.csv',
    auxiliaries_path='data/workflows_auxiliaries.csv',
    output_path='data/cleaned_github_actions.csv'
):
    """
    Load raw GitHub Actions data from CSV files, perform basic cleaning and feature
    engineering, and save the processed data to a new CSV file.

    Parameters
    ----------
    repositories_path : str
        Path to the repositories CSV file.
    workflows_path : str
        Path to the workflows CSV file.
    auxiliaries_path : str
        Path to the auxiliaries CSV file (unused but kept for completeness).
    output_path : str
        Path where the cleaned dataset will be saved.

    Returns
    -------
    pd.DataFrame
        The cleaned and processed DataFrame.
    """
    # Read the raw CSV files. Users should ensure these exist before running.
    repos = pd.read_csv(repositories_path)
    workflows = pd.read_csv(workflows_path)

    # Standardise column names by stripping whitespace
    repos.columns = repos.columns.str.strip()
    workflows.columns = workflows.columns.str.strip()

    # Convert created/updated columns to datetime; coerce errors to NaT
    repos['created'] = pd.to_datetime(repos['created'], errors='coerce')
    repos['updated'] = pd.to_datetime(repos['updated'], errors='coerce')

    # Calculate repository duration in seconds and fill missing values with the median
    repos['repo_duration'] = (repos['updated'] - repos['created']).dt.total_seconds()
    repos['repo_duration'] = repos['repo_duration'].fillna(repos['repo_duration'].median())

    # Merge workflow metadata with repository-level features. Only include relevant columns.
    df = workflows.merge(
        repos[['name', 'repo_duration', 'commits', 'branches', 'contributors',
               'stars', 'issues', 'pullrequests', 'size', 'language']],
        left_on='repository',
        right_on='name',
        how='left'
    )

    # Create a categorical outcome column based on valid_workflow flag
    if 'valid_workflow' in df.columns:
        df['conclusion'] = df['valid_workflow'].map({True: 'success', False: 'failure'})
    else:
        # Fallback if the column doesn't exist
        df['conclusion'] = 'unknown'

    # For numeric columns, replace missing values with the median of each column
    numeric_cols = ['repo_duration', 'commits', 'branches', 'contributors',
                    'stars', 'issues', 'pullrequests', 'size']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Encode the categorical columns using a single LabelEncoder instance
    le = LabelEncoder()
    df['conclusion_encoded'] = le.fit_transform(df['conclusion'])

    # Map language column to numeric codes, filling missing values first
    df['language'] = df['language'].fillna('unknown')
    df['language_encoded'] = le.fit_transform(df['language'])

    # Scale the repository duration to [0, 1] range
    scaler = MinMaxScaler()
    df['repo_duration_scaled'] = scaler.fit_transform(df[['repo_duration']])

    # Remove columns that are no longer needed
    drop_cols = ['name', 'repository', 'valid_workflow']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Save the cleaned DataFrame
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == '__main__':
    cleaned = prepare_data()
    print('Cleaned dataset created with shape:', cleaned.shape)