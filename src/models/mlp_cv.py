from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

from src.pipelines.model_pipeline import build_model_pipeline


def evaluate_mlp_cv(X, y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        alpha=0.01,
        early_stopping=True,
        max_iter=300,
        random_state=42
    )

    pipeline = build_model_pipeline(model)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    roc_auc_scores = cross_val_score(
        pipeline,
        X,
        y_encoded,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )

    return {
        "roc_auc_mean": np.mean(roc_auc_scores),
        "roc_auc_std": np.std(roc_auc_scores)
    }


