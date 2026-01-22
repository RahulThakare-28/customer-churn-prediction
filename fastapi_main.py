from fastapi import FastAPI

from src.api.routes import router
from src.inference.predictor import ChurnPredictor
app = FastAPI(title="Customer Churn API")

#app.include_router(router)

predictor = ChurnPredictor()

@app.post("/predict")
def predict(payload: dict):
    return predictor.predict(payload)

@app.get("/")
def health():
    return {"status": "ok"}
# def root():
#     return {"status": "API is running"}
