import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def evaluate_models(
    models: dict,
    preprocessor,
    X_train,
    X_valid,
    y_train,
    y_valid
):
    """
    Trains each model in a pipeline and evaluates on validation data.
    Returns a DataFrame with metrics.
    """

    results = []

    for name, model in models.items():
        pipe = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)

        y_prob = pipe.predict_proba(X_valid)[:, 1]
        y_pred = pipe.predict(X_valid)

        metrics = {
            "model": name,
            "roc_auc": roc_auc_score(y_valid, y_prob),
            "accuracy": accuracy_score(y_valid, y_pred),
            "precision": precision_score(y_valid, y_pred),
            "recall": recall_score(y_valid, y_pred),
            "f1": f1_score(y_valid, y_pred)
        }

        results.append(metrics)

    return pd.DataFrame(results).sort_values(
        by="roc_auc", ascending=False
    )

