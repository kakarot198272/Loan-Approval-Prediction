import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)
import os


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
    Returns:
        - results_df: DataFrame with evaluation metrics
        - model_probs: dict {model_name: predicted probabilities}
    """

    results = []
    model_probs = {}

    for name, model in models.items():
        pipe = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)

        y_prob = pipe.predict_proba(X_valid)[:, 1]
        y_pred = pipe.predict(X_valid)

        results.append({
            "model": name,
            "roc_auc": roc_auc_score(y_valid, y_prob),
            "accuracy": accuracy_score(y_valid, y_pred),
            "precision": precision_score(y_valid, y_pred),
            "recall": recall_score(y_valid, y_pred),
            "f1": f1_score(y_valid, y_pred)
        })

        model_probs[name] = y_prob

    results_df = pd.DataFrame(results).sort_values(
        by="roc_auc", ascending=False
    )

    return results_df, model_probs


def plot_roc_curves(model_probs, y_true, save_path):
    """
    Plots ROC curves for all models and saves the figure.
    """

    plt.figure(figsize=(8, 6))

    for model_name, probs in model_probs.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
