import pytest
from fastapi.testclient import TestClient
from api.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] == True

def test_model_info(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["experiment_track"] == "Early Warning"
    assert "grade_period1" in data["excluded_prior_grade_features"]

def test_predict_valid(client):
    payload = {
        "age": 15,
        "sex": "F",
        "address": "U",
        "family_size": "GT3",
        "parents_cohabitation": "T",
        "mother_education": 4,
        "father_education": 4,
        "mother_job": "teacher",
        "father_job": "teacher",
        "study_time": 4,
        "failures": 0,
        "school_support": "yes",
        "family_support": "yes",
        "paid_classes": "yes",
        "extra_activities": "yes",
        "internet_access": "yes",
        "romantic_relationship": "no",
        "free_time": 3,
        "going_out": 2,
        "weekday_alcohol": 1,
        "weekend_alcohol": 1,
        "health_status": 5,
        "absences": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert "risk_tier" in data
    assert "decision_threshold" in data

def test_predict_invalid_missing(client):
    payload = {"age": 15}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # Validation Error

def test_predict_invalid_range(client):
    payload = {
        "age": 5, # Invalid age
        "sex": "F",
        "address": "U",
        "family_size": "GT3",
        "parents_cohabitation": "T",
        "mother_education": 4,
        "father_education": 4,
        "mother_job": "teacher",
        "father_job": "teacher",
        "study_time": 4,
        "failures": 0,
        "school_support": "yes",
        "family_support": "yes",
        "paid_classes": "yes",
        "extra_activities": "yes",
        "internet_access": "yes",
        "romantic_relationship": "no",
        "free_time": 3,
        "going_out": 2,
        "weekday_alcohol": 1,
        "weekend_alcohol": 1,
        "health_status": 5,
        "absences": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_extra_fields_ignored(client):
    payload = {
        "age": 15,
        "sex": "F",
        "address": "U",
        "family_size": "GT3",
        "parents_cohabitation": "T",
        "mother_education": 4,
        "father_education": 4,
        "mother_job": "teacher",
        "father_job": "teacher",
        "study_time": 4,
        "failures": 0,
        "school_support": "yes",
        "family_support": "yes",
        "paid_classes": "yes",
        "extra_activities": "yes",
        "internet_access": "yes",
        "romantic_relationship": "no",
        "free_time": 3,
        "going_out": 2,
        "weekday_alcohol": 1,
        "weekend_alcohol": 1,
        "health_status": 5,
        "absences": 0,
        "final_grade": 20 # Extra field
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # We forbid extra fields
