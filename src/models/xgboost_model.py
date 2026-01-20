import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.pipelines.model_pipeline import build_model_pipeline
from src.evaluation.metrics import classification_metrics
from src.evaluation.metrics_logger import save_metrics
from src.utils.artifact_utils import ensure_artifact_dir

def train_xgboost(X, y):
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

    # ---- XGBoost Model ----
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1


    )

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
        model_name="XGBoost"
    )

    # ---- Save model ----
    artifact_dir = ensure_artifact_dir()

    joblib.dump(
        pipeline,
        artifact_dir / "xgboost_model.joblib"
    )
    return metrics
