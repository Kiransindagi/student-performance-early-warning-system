import pandas as pd
from pathlib import Path

def load_data(data_path: str = "data/raw/student-mat.csv") -> pd.DataFrame:
    """
    Load raw student dataset and map original columns to readable project names.
    We default to student-mat.csv.
    """
    df = pd.read_csv(data_path, sep=';')
    
    # Mapping original columns to project feature names
    column_mapping = {
        'age': 'age',
        'sex': 'sex',
        'address': 'address',
        'famsize': 'family_size',
        'Pstatus': 'parents_cohabitation',
        'Medu': 'mother_education',
        'Fedu': 'father_education',
        'Mjob': 'mother_job',
        'Fjob': 'father_job',
        'studytime': 'study_time',
        'failures': 'failures',
        'schoolsup': 'school_support',
        'famsup': 'family_support',
        'paid': 'paid_classes',
        'activities': 'extra_activities',
        'internet': 'internet_access',
        'romantic': 'romantic_relationship',
        'freetime': 'free_time',
        'goout': 'going_out',
        'Dalc': 'weekday_alcohol',
        'Walc': 'weekend_alcohol',
        'health': 'health_status',
        'absences': 'absences',
        'G1': 'grade_period1',
        'G2': 'grade_period2',
        'G3': 'final_grade'
    }
    
    # We only keep the columns we mapped
    columns_to_keep = list(column_mapping.keys())
    df = df[[col for col in columns_to_keep if col in df.columns]]
    df = df.rename(columns=column_mapping)
    
    return df
