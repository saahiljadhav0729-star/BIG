# Practical 3: Basic Statistics - Central Tendency and Variance

import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df = iris.frame.copy()
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))

# Group numeric measurements by the categorical species variable.
print("Statistics grouped by species:")
print(df.groupby("species", observed=False)["sepal length (cm)"].agg(
    ["mean", "median", "min", "max", "std", "var"]
))

# Numeric code for each categorical response.
species_codes = df["species"].astype("category").cat.codes.tolist()
print("\nNumeric species codes (first 15):", species_codes[:15])

# Basic statistical details for each Iris species.
for species in iris.target_names:
    subset = df[df["species"] == species]
    print("\nSpecies:", species)
    print(subset[iris.feature_names].describe(percentiles=[0.25, 0.50, 0.75]))

print("\nPractical 3 completed.")
