from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.pipelines.full_preprocessing_pipeline import build_full_pipeline
from src.evaluation.metrics import classification_metrics


def train_logistic(X, y):

    # ---- Encode target (correct) ----
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)   # No=0, Yes=1

    # ---- Train-test split FIRST ----
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        stratify=y_encoded,
        random_state=42
    )

    # ---- Fresh preprocessing (NOT pre-fitted) ----
    preprocessing = build_full_pipeline()

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs"
    )

    # ---- SAFE pipeline ----
    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessing),
        ("model", model)
    ])

    # ---- Fit ONLY on train ----
    pipeline.fit(X_train, y_train)

    # ---- Evaluate on test ----
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = classification_metrics(y_test, y_pred, y_prob)

    return pipeline, metrics
