from pathlib import Path

ARTIFACT_DIR = Path("artifacts/selected_models")

def ensure_artifact_dir():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACT_DIR
