import pytest
from fastapi.testclient import TestClient
from api.main import app
import joblib
import pandas as pd
from src.features.preprocess import build_preprocessor

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_integration_pipeline_match(client):
    # Load artifact manually
    model = joblib.load("models/early_warning_model.pkl")
    
    # Input
    payload = {
        "age": 16, "sex": "M", "address": "R", "family_size": "LE3", "parents_cohabitation": "T",
        "mother_education": 2, "father_education": 2, "mother_job": "other", "father_job": "other",
        "study_time": 2, "failures": 1, "school_support": "no", "family_support": "no",
        "paid_classes": "no", "extra_activities": "yes", "internet_access": "yes",
        "romantic_relationship": "yes", "free_time": 4, "going_out": 4, "weekday_alcohol": 2,
        "weekend_alcohol": 3, "health_status": 3, "absences": 4
    }
    
    # API prediction
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    api_proba = response.json()["risk_probability"]
    
    # Manual prediction
    df = pd.DataFrame([payload])
    # Reorder according to included features in metadata
    features = [
        "age", "sex", "address", "family_size", "parents_cohabitation", 
        "mother_education", "father_education", "mother_job", "father_job", 
        "study_time", "failures", "school_support", "family_support", 
        "paid_classes", "extra_activities", "internet_access", "romantic_relationship", 
        "free_time", "going_out", "weekday_alcohol", "weekend_alcohol", 
        "health_status", "absences"
    ]
    df = df[features]
    manual_proba = model.predict_proba(df)[0][1]
    
    # Assert they match perfectly
    assert abs(api_proba - manual_proba) < 1e-6
