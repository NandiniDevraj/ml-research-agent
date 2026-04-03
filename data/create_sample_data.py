# data/create_sample_data.py
# Generates a realistic fake hospital readmission dataset for testing

import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

df = pd.DataFrame({
    "age":               np.random.randint(18, 90, n),
    "num_medications":   np.random.randint(1, 20, n),
    "num_procedures":    np.random.randint(0, 10, n),
    "num_diagnoses":     np.random.randint(1, 9, n),
    "time_in_hospital":  np.random.randint(1, 14, n),
    "num_lab_tests":     np.random.randint(1, 50, n),
    "gender":            np.random.choice(["Male", "Female"], n),
    "admission_type":    np.random.choice(["Emergency", "Elective", "Urgent"], n),
    "discharge_to":      np.random.choice(["Home", "SNF", "Rehab", "Other"], n),
    "primary_diagnosis": np.random.choice(["Diabetes", "Heart Failure", "COPD", "Pneumonia"], n),
    "A1C_result":        np.random.choice(["Normal", "Abnormal", "Not Tested"], n),
    "readmitted":        np.random.choice([0, 1], n, p=[0.78, 0.22])  # 22% readmission rate
})

# Introduce some missing values (realistic)
df.loc[np.random.choice(n, 50, replace=False), "A1C_result"] = np.nan
df.loc[np.random.choice(n, 30, replace=False), "num_lab_tests"] = np.nan

# Introduce some outliers
df.loc[np.random.choice(n, 10, replace=False), "num_medications"] = 99

df.to_csv("data/hospital_readmission.csv", index=False)
print(f"Dataset created: {len(df)} rows, {len(df.columns)} columns")
print(df.head())