from pydantic import BaseModel


class CustomerInput(BaseModel):
    Gender: str
    Age: int
    Married: str
    Dependents: str
    Tenure_in_Months: int
    Contract: str
    Monthly_Charge: float
    Total_Revenue: float
    Internet_Service: str
    Payment_Method: str


class PredictionResponse(BaseModel):
    churn_prediction: str
    churn_probability: float
