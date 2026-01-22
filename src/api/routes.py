from fastapi import APIRouter
from pydantic import BaseModel
from src.inference.predictor import predict

router = APIRouter()


class CustomerInput(BaseModel):
    Gender: str
    Age: int
    Married: str
    Dependents: str
    Tenure_in_Months: int
    Contract: str
    Monthly_Charges: float
    Total_Charges: float
    Internet_Service: str
    Payment_Method: str


@router.post("/predict")
def predict_churn(data: CustomerInput):
    return predict(data.dict())
