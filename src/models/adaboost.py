import joblib
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sklearn

from src.pipelines.model_pipeline import build_model_pipeline
from src.evaluation.metrics import classification_metrics
from src.evaluation.metrics_logger import save_metrics
from src.utils.artifact_utils import ensure_artifact_dir


def train_adaboost(X, y):

    # ---- Encode target ----
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # ---- Train-test split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        stratify=y_encoded,
        random_state=42
    )

    # ---- Base Estimator ----
    base_estimator = DecisionTreeClassifier(
        max_depth=2,
        min_samples_leaf=10,
        random_state=42
    )

    # ---- AdaBoost (version-safe) ----
    if sklearn.__version__ >= "1.2":
        model = AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=300,
            learning_rate=0.05,
            random_state=42
        )
    else:
        model = AdaBoostClassifier(
            base_estimator=base_estimator,
            n_estimators=300,
            learning_rate=0.05,
            random_state=42
        )

    # ---- Pipeline ----
    pipeline = build_model_pipeline(model)
    pipeline.fit(X_train, y_train)

    # ---- Predictions ----
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # ---- Metrics ----
    metrics = classification_metrics(y_test, y_pred, y_prob)

    tn, fp, fn, tp = metrics["confusion_matrix"].ravel()

    # ---- Save metrics ----
    save_metrics(
        {
            "accuracy": metrics["accuracy"],
            "recall": metrics["recall"],
            "roc_auc": metrics["roc_auc"],
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        model_name="AdaBoost"
    )

    # ---- Save model (pipeline includes preprocessing) ----
    artifact_dir = ensure_artifact_dir()
    joblib.dump(pipeline, artifact_dir / "adaboost_model.joblib")

    return metrics
