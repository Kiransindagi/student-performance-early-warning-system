import pytest
import pandas as pd
from src.data.ingest import load_data
from src.data.validate import validate_data, create_target
from src.features.preprocess import build_preprocessor, prepare_data

def test_load_data():
    df = load_data("data/raw/student-mat.csv")
    assert not df.empty
    # check columns mapping
    expected_cols = ['age', 'sex', 'address', 'family_size', 'parents_cohabitation', 
                     'mother_education', 'father_education', 'mother_job', 'father_job', 
                     'study_time', 'failures', 'school_support', 'family_support', 
                     'paid_classes', 'extra_activities', 'internet_access', 'romantic_relationship', 
                     'free_time', 'going_out', 'weekday_alcohol', 'weekend_alcohol', 
                     'health_status', 'absences', 'grade_period1', 'grade_period2', 'final_grade']
    for col in expected_cols:
        assert col in df.columns

def test_validate_data():
    df = load_data("data/raw/student-mat.csv")
    assert validate_data(df) == True
    
    # test failure case
    df_bad = df.copy()
    df_bad.loc[0, 'age'] = 150 # impossible range
    assert validate_data(df_bad) == False

def test_create_target():
    df = pd.DataFrame({'final_grade': [5, 10, 15]})
    df_out = create_target(df, threshold=10)
    assert 'at_risk' in df_out.columns
    assert df_out['at_risk'].tolist() == [1, 0, 0]

def test_preprocessing():
    df = load_data("data/raw/student-mat.csv")
    df = create_target(df, threshold=10)
    
    X, y = prepare_data(df, include_prior_grades=False)
    
    preprocessor = build_preprocessor(include_prior_grades=False)
    X_trans = preprocessor.fit_transform(X)
    
    assert X_trans.shape[0] == X.shape[0]
    # Check that grade_period1 and 2 are dropped
    assert 'grade_period1' not in X.columns
    assert 'grade_period2' not in X.columns
