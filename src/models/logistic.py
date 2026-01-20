from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.pipelines.model_pipeline import build_model_pipeline
from src.evaluation.metrics import classification_metrics
from src.utils.artifact_utils import ensure_artifact_dir
import joblib


def train_logistic(X, y):

    # ---- Encode target ----
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)   # No=0, Yes=1

    # ---- Train-test split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        stratify=y_encoded,
        random_state=42
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs"
    )

    # ---- CORRECT pipeline (loads preprocessing) ----
    pipeline = build_model_pipeline(model)

    # ---- Fit ----
    pipeline.fit(X_train, y_train)

    # ---- Evaluate ----
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = classification_metrics(y_test, y_pred, y_prob)

    # ---- Save model ----
    artifact_dir = ensure_artifact_dir()
    joblib.dump(pipeline, artifact_dir / "logistic_model.joblib")

    return metrics
