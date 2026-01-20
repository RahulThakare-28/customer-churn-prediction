

MODEL_FEATURES = [
    "Gender", "Age", "Married", "Dependents", "Tenure in Months",
    "Contract", "Monthly Charge", "Total Charges", "Internet Service",
    "Payment Method"
]

# Mapping for API → Training
API_TO_MODEL_COLS = {
    "Age": "Age",
    "Tenure_in_Months": "Tenure in Months",
    "Monthly_Charges": "Monthly Charge",
    "Total_Charges": "Total Charges",
    "Internet_Service": "Internet Service",
    "Payment_Method": "Payment Method",
    "Gender": "Gender",
    "Married": "Married",
    "Dependents": "Dependents",
    "Contract": "Contract"
}
