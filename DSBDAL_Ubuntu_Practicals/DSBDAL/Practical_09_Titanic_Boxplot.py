# Practical 9: Box plot of passenger age by gender and survival

import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("titanic")
print("First five rows:\n", df.head())

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="sex", y="age", hue="survived")
plt.title("Age Distribution by Gender and Survival")
plt.xlabel("Gender")
plt.ylabel("Age")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.tight_layout()
plt.show()

print("\nObservation:")
print("The box plot compares median age, spread and possible outliers for each gender,")
print("with separate groups for passengers who did and did not survive.")
