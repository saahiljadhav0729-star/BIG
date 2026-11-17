# Practical 4: Linear Regression for Boston Housing price prediction
# Preferred: put the college-provided BostonHousing.csv in this folder.
# Expected target column: MEDV (or medv / target / price).

import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

csv_file = sys.argv[1] if len(sys.argv) > 1 else "BostonHousing.csv"

try:
    df = pd.read_csv(csv_file)
    print("Loaded local dataset:", csv_file)
except FileNotFoundError:
    print("Local CSV not found. Trying OpenML (internet required)...")
    from sklearn.datasets import fetch_openml
    data = fetch_openml(name="boston", version=1, as_frame=True)
    df = data.frame

df.columns = [str(c).strip() for c in df.columns]
target_candidates = ["MEDV", "medv", "target", "price"]
target_col = next((c for c in target_candidates if c in df.columns), None)

if target_col is None:
    raise ValueError("Target column not found. Check your CSV; expected MEDV or price.")

df = df.dropna()
X = df.drop(columns=[target_col])
X = pd.get_dummies(X, drop_first=True)
y = pd.to_numeric(df[target_col])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Actual values (first 5):", list(y_test.head()))
print("Predicted values (first 5):", predictions[:5])
print("Mean Squared Error:", mean_squared_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))
print("\nPractical 4 completed.")
