# Practical 5: Logistic Regression on Social_Network_Ads.csv
# Run: python3 Practical_05_Logistic_Regression.py Social_Network_Ads.csv

import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

csv_file = sys.argv[1] if len(sys.argv) > 1 else "Social_Network_Ads.csv"
df = pd.read_csv(csv_file)
df.columns = [c.strip() for c in df.columns]

target_col = "Purchased"
if target_col not in df.columns:
    raise ValueError("CSV must contain a 'Purchased' target column.")

# Remove identifier columns; encode optional categorical columns.
drop_cols = [c for c in ["User ID", target_col] if c in df.columns]
X = df.drop(columns=drop_cols)
y = df[target_col]
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
pred = model.predict(X_test)

labels = sorted(pd.Series(y_test).unique())
cm = confusion_matrix(y_test, pred, labels=labels)
print("Confusion matrix:\n", cm)

if len(labels) == 2:
    tn, fp, fn, tp = cm.ravel()
    print("TN:", tn, "FP:", fp, "FN:", fn, "TP:", tp)

print("Accuracy:", accuracy_score(y_test, pred))
print("Error rate:", 1 - accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred, pos_label=labels[-1], zero_division=0))
print("Recall:", recall_score(y_test, pred, pos_label=labels[-1], zero_division=0))
print("\nPractical 5 completed.")
