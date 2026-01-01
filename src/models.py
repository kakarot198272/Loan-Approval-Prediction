from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

# Optional dependency
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


def get_models(scale_pos_weight=None, random_state=42):
    """
    Returns a dictionary of models to evaluate.
    """

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state
        ),
        "kNN": KNeighborsClassifier(),
        "SVM": SVC(
            probability=True,
            class_weight="balanced",
            random_state=random_state
        ),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state
        ),
        "AdaBoost": AdaBoostClassifier(
            random_state=random_state
        ),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            early_stopping=True,
            random_state=random_state
        )
    }

    if XGBClassifier is not None and scale_pos_weight is not None:
        models["XGBoost"] = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=random_state
        )

    return models

