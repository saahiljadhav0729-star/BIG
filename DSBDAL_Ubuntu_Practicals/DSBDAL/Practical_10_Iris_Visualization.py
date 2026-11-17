# Practical 10: Iris dataset feature analysis, histograms and boxplots

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df = iris.data.copy()
df["species"] = iris.target.map(dict(enumerate(iris.target_names)))

print("First five rows:\n", df.head())
print("\nNumber of features:", len(iris.feature_names))
print("Feature names:", iris.feature_names)
print("\nFeature types:\n", df[iris.feature_names].dtypes)
print("\nSummary:\n", df.describe())

# Histogram for every numeric feature.
df[iris.feature_names].hist(figsize=(10, 8), bins=15, edgecolor="black")
plt.suptitle("Histograms of Iris Features")
plt.tight_layout()
plt.show()

# Boxplot for every numeric feature.
plt.figure(figsize=(10, 6))
df[iris.feature_names].boxplot()
plt.title("Boxplots of Iris Features")
plt.ylabel("Measurement (cm)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

print("\nCompare medians, spread and points beyond whiskers to identify possible outliers.")
print("\nPractical 10 completed.")
