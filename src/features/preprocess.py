from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

def build_preprocessor(include_prior_grades: bool = False):
    """
    Build reusable scikit-learn preprocessing pipeline.
    """
    numerical_features = [
        'age', 'mother_education', 'father_education', 'study_time', 
        'failures', 'free_time', 'going_out', 
        'weekday_alcohol', 'weekend_alcohol', 'health_status', 'absences'
    ]
    
    categorical_features = [
        'sex', 'address', 'family_size', 'parents_cohabitation', 'mother_job', 
        'father_job', 'school_support', 'family_support', 
        'paid_classes', 'extra_activities', 'internet_access', 'romantic_relationship'
    ]
    
    # We might need to map some ordinal/binary manual if we want, but OHE is fine for categorical
    # Some features like family_size (GT3/LE3) could be just label encoded or OHE.
    
    if include_prior_grades:
        numerical_features.extend(['grade_period1', 'grade_period2'])
        
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='if_binary'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        
    return preprocessor

def prepare_data(df: pd.DataFrame, include_prior_grades: bool = False):
    """
    Separate features and target
    """
    X = df.drop(columns=['final_grade', 'at_risk'])
    if not include_prior_grades:
        X = X.drop(columns=['grade_period1', 'grade_period2'])
    y = df['at_risk']
    return X, y
