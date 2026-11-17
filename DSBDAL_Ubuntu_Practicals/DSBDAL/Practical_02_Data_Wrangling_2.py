# Practical 2: Data Wrangling II
# Missing values, inconsistencies, outlier handling and data transformation.

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df = iris.frame.copy()
df.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in df.columns]

# Add a deliberate missing value and an extreme value for demonstration.
df.loc[0, "sepal_length"] = np.nan
df.loc[1, "sepal_length"] = 100.0

print("Missing values before cleaning:\n", df.isnull().sum())
print("\nNumeric summary before cleaning:\n", df.describe())

# Fill missing numeric values with their column median.
df["sepal_length"] = df["sepal_length"].fillna(df["sepal_length"].median())

# Detect and cap outliers using the IQR rule.
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df[col] = df[col].clip(lower, upper)

# Log transformation reduces right-skew where values are non-negative.
measurement = "sepal_length"
df["log_" + measurement] = np.log1p(df[measurement])

print("\nMissing values after cleaning:\n", df.isnull().sum())
print("\nCleaned and transformed data:\n", df.head())
print("\nPractical 2 completed.")
