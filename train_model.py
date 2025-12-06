import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


def train_and_select_model(data_path='data/cleaned_github_actions.csv', models_dir='models'):
    """
    Train logistic regression and random forest classifiers on the cleaned dataset,
    compare their performance, and save the best-performing model to disk.

    Parameters
    ----------
    data_path : str
        Path to the preprocessed CSV file produced by prepare_data().
    models_dir : str
        Directory where trained models will be saved.

    Returns
    -------
    str
        The filename of the best model saved in `models_dir`.
    """
    df = pd.read_csv(data_path)

    # ✅ Use a random sample to speed up training (good enough for coursework)
    if len(df) > 100000:
        df = df.sample(n=100000, random_state=42)


    # Define features and target
    feature_cols = [
        'repo_duration_scaled', 'commits', 'branches', 'contributors',
        'stars', 'issues', 'pullrequests', 'size', 'language_encoded'
    ]
    X = df[feature_cols]
    y = df['conclusion_encoded']

    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model 1: Logistic Regression
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    y_pred_lr = log_reg.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)

    # Model 2: Random Forest
    rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    n_jobs=-1,
    random_state=42
    )

    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)

    # Choose the best model based on accuracy
    os.makedirs(models_dir, exist_ok=True)
    if acc_rf >= acc_lr:
        best_model = rf
        model_name = 'random_forest.joblib'
        best_accuracy = acc_rf
    else:
        best_model = log_reg
        model_name = 'logistic_regression.joblib'
        best_accuracy = acc_lr

    model_path = os.path.join(models_dir, model_name)
    joblib.dump(best_model, model_path)

    print(f'Saved best model as {model_name} with accuracy {best_accuracy:.4f}')
    # Print full classification report for transparency
    print(classification_report(y_test, best_model.predict(X_test)))

    return model_name


if __name__ == '__main__':
    train_and_select_model()