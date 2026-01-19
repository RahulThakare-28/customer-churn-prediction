from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel




def importance_selector():
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    return SelectFromModel(model, threshold='median')