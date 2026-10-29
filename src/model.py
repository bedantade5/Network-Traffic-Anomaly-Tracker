"""Consistent categorical encoding and Isolation Forest scoring."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_model(numeric: list[str], categorical: list[str], contamination: float = 0.01):
    if not 0 < contamination <= 0.5:
        raise ValueError("contamination must be in (0, 0.5]")
    encoder = ColumnTransformer([
        ("numeric", "passthrough", numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return Pipeline([
        ("preprocess", encoder),
        ("forest", IsolationForest(contamination=contamination, random_state=42,
                                    n_estimators=100, n_jobs=-1)),
    ])


def score(model, features):
    # Higher anomaly_score means more unusual. decision_function < 0 is anomalous.
    prediction = model.predict(features)
    anomaly_score = -model.decision_function(features)
    return np.asarray(prediction), np.asarray(anomaly_score)
