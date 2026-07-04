import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import pandas as pd
import numpy as np
from src.data.ingest import load_data
from src.data.validate import validate_data, create_target

df = load_data("data/raw/student-mat.csv")
print("Data Loaded. Shape:", df.shape)

is_valid = validate_data(df)
print("Is Data Valid?", is_valid)

df = create_target(df, threshold=10)

print("\n--- Missing Values ---")
print(df.isnull().sum().sum())

print("\n--- Duplicate Rows ---")
print(df.duplicated().sum())

print("\n--- Target Class Balance ---")
print(df['at_risk'].value_counts(normalize=True))

print("\n--- Possible Leakage Risks ---")
print("grade_period1 and grade_period2 are highly correlated with final_grade and thus at_risk. Including them in an early warning system before grades are available is a leak of future information.")

print("\n--- Correlations with Target (Numerical) ---")
num_cols = df.select_dtypes(include=[np.number]).columns
corr = df[num_cols].corr()['at_risk'].sort_values()
print(corr)

with open("reports/eda_report.md", "w") as f:
    f.write("# EDA Report\n\n")
    f.write(f"Shape: {df.shape}\n\n")
    f.write("## Target Class Balance\n")
    f.write(df['at_risk'].value_counts(normalize=True).to_string())
    f.write("\n\n## Correlations\n")
    f.write(corr.to_string())
    
