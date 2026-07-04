import pytest
import pandas as pd
import joblib
from src.data.ingest import load_data
from src.data.validate import create_target
from src.features.preprocess import build_preprocessor, prepare_data

def test_early_warning_feature_exclusion():
    df = load_data("data/raw/student-mat.csv")
    df = create_target(df, threshold=10)
    X, y = prepare_data(df, include_prior_grades=False)
    assert "grade_period1" not in X.columns
    assert "grade_period2" not in X.columns
    assert "final_grade" not in X.columns

def test_performance_model_feature_inclusion():
    df = load_data("data/raw/student-mat.csv")
    df = create_target(df, threshold=10)
    X, y = prepare_data(df, include_prior_grades=True)
    assert "grade_period1" in X.columns
    assert "grade_period2" in X.columns
    assert "final_grade" not in X.columns

def test_artifact_loading_and_inference():
    model = joblib.load("models/early_warning_model.pkl")
    df = load_data("data/raw/student-mat.csv")
    df = create_target(df, threshold=10)
    X, y = prepare_data(df, include_prior_grades=False)
    
    # Infer on first 5 samples
    y_pred = model.predict(X.head())
    y_proba = model.predict_proba(X.head())
    assert len(y_pred) == 5
    assert y_proba.shape == (5, 2)
    assert (y_proba >= 0).all() and (y_proba <= 1).all()

def test_preprocessing_consistency():
    prep1 = build_preprocessor(include_prior_grades=False)
    prep2 = build_preprocessor(include_prior_grades=True)
    assert prep1 != prep2
