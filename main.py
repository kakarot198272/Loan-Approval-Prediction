"""
Loan Approval Prediction
Entry point for training and evaluating models.
"""

import os
import pandas as pd

from src.data import (
    load_train_test,
    split_xy,
    make_train_valid_split
)
from src.preprocessing import build_preprocessor
from src.models import get_models
from src.evaluation import evaluate_models


# ------------------
# Configuration
# ------------------
DATA_DIR = "data"
OUTPUT_DIR = "outputs"

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

TARGET_COL = "loan_status"
TEST_SIZE = 0.2
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("Loading data...")
    train_df, _ = load_train_test(TRAIN_PATH, TEST_PATH)

    print("Splitting features and target...")
    X, y = split_xy(train_df, TARGET_COL)

    print("Creating train-validation split...")
    X_train, X_valid, y_train, y_valid = make_train_valid_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # Identify feature types
    numeric_features = X_train.select_dtypes(include="number").columns.tolist()
    categorical_features = X_train.select_dtypes(exclude="number").columns.tolist()

    print("Building preprocessing pipeline...")
    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features
    )

    # Compute scale_pos_weight for XGBoost (if used)
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

    print("Loading models...")
    models = get_models(
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE
    )

    print("Evaluating models...")
    results_df = evaluate_models(
        models,
        preprocessor,
        X_train,
        X_valid,
        y_train,
        y_valid
    )

    print("\nModel comparison (sorted by ROC-AUC):")
    print(results_df.to_string(index=False))

    # Save results
    output_path = os.path.join(OUTPUT_DIR, "baseline_model_results.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

