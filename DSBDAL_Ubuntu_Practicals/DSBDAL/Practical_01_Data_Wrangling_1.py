# Practical 1: Data Wrangling I
# Load data, inspect missing values/statistics/types, encode categories, normalize values.

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

iris = load_iris(as_frame=True)
df = iris.frame.copy()
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
df.drop(columns=["target"], inplace=True)

print("First five rows:\n", df.head())
print("\nMissing values:\n", df.isnull().sum())
print("\nSummary statistics:\n", df.describe())
print("\nDimensions (rows, columns):", df.shape)
print("\nData types:\n", df.dtypes)
print("\nVariable descriptions:\n", df.columns.tolist())

# Convert categorical species into numeric codes.
df["species_code"] = df["species"].astype("category").cat.codes
print("\nCategorical variable converted:\n", df[["species", "species_code"]].head())

# Normalize the four numeric measurement columns to the range 0 to 1.
numeric_cols = iris.feature_names
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("\nNormalized data:\n", df.head())
print("\nPractical 1 completed.")
