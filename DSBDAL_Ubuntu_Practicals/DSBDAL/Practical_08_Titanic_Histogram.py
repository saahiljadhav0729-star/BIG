# Practical 8: Histogram of Titanic passenger fares using Seaborn

import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("titanic")
print("First five rows:\n", df.head())
print("\nDataset information:")
df.info()
print("\nMissing values:\n", df.isnull().sum())

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="fare", bins=30, kde=True)
plt.title("Distribution of Titanic Passenger Fares")
plt.xlabel("Ticket Fare")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.show()
