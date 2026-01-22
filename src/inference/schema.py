# Single source of truth

MODEL_FEATURES = [
    "Gender",
    "Age",
    "Married",
    "Dependents",
    "Tenure in Months",
    "Contract",
    "Monthly Charge",
    "Total Charges",
    "Internet Service",
    "Payment Method",
]

# API → Training schema
API_TO_MODEL_COLS = {
    "Gender": "Gender",
    "Age": "Age",
    "Married": "Married",
    "Dependents": "Dependents",
    "Tenure_in_Months": "Tenure in Months",
    "Contract": "Contract",
    "Monthly_Charges": "Monthly Charge",
    "Total_Charges": "Total Charges",
    "Internet_Service": "Internet Service",
    "Payment_Method": "Payment Method",
}
