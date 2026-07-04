import pandas as pd

def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate the loaded data to ensure it meets requirements.
    Checks:
    - missing values
    - duplicate rows
    - impossible numerical ranges
    """
    
    # 1. Missing values
    if df.isnull().any().any():
        print("Validation Failed: Missing values found.")
        return False
        
    # 2. Duplicate rows
    if df.duplicated().any():
        print(f"Validation Warning: {df.duplicated().sum()} duplicate rows found. We will keep them for now, but log a warning.")
        
    # 3. Impossible numerical ranges
    # grades are between 0 and 20
    grade_cols = ['grade_period1', 'grade_period2', 'final_grade']
    for col in grade_cols:
        if col in df.columns:
            if not df[col].between(0, 20).all():
                print(f"Validation Failed: {col} has values outside 0-20 range.")
                return False
                
    if not df['age'].between(10, 30).all(): # realistic range for students in this dataset
        print("Validation Failed: 'age' has unrealistic values.")
        return False
        
    if not df['absences'].between(0, 93).all(): # max absences in this dataset is typically 93, but let's just say >= 0
        if (df['absences'] < 0).any():
            print("Validation Failed: 'absences' cannot be negative.")
            return False

    return True

def create_target(df: pd.DataFrame, threshold: int = 10) -> pd.DataFrame:
    """
    Create configurable binary target 'at_risk'
    At Risk (1): final grade < threshold
    Not At Risk (0): final grade >= threshold
    """
    df_out = df.copy()
    df_out['at_risk'] = (df_out['final_grade'] < threshold).astype(int)
    return df_out
