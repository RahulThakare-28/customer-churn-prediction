import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier

from src.pipelines.model_pipeline import build_model_pipeline
from src.evaluation.metrics import classification_metrics
from src.evaluation.metrics_logger import save_metrics


def train_mlp(X, y):
    # ---- Encode target ----
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    joblib.dump(le, "artifacts/selected_models/label_encoder.joblib")

    # ---- Train-test split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        stratify=y_encoded,
        random_state=42
    )

    # ---- MLP Classifier (with basic fine-tuning) ----
    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        alpha=0.01,  # L2 regularization (IMPORTANT)
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        random_state=42
    )

    # ---- Pipeline ----
    # Important: scale numeric features before MLP
    pipeline = build_model_pipeline(model)
    pipeline.fit(X_train, y_train)

    # ---- Predictions ----
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # ---- Metrics ----
    metrics = classification_metrics(y_test, y_pred, y_prob)

    # ---- Save metrics ----
    save_metrics(
        {
            "accuracy": metrics["accuracy"],
            "recall": metrics["recall"],
            "roc_auc": metrics["roc_auc"],
            "tn": metrics["confusion_matrix"][0][0],
            "fp": metrics["confusion_matrix"][0][1],
            "fn": metrics["confusion_matrix"][1][0],
            "tp": metrics["confusion_matrix"][1][1],
        },
        model_name="MLPClassifier"
    )

    # ---- Save model ----
    joblib.dump(
        pipeline,
        "artifacts/selected_models/mlp_classifier_model.joblib"
    )

    return metrics
