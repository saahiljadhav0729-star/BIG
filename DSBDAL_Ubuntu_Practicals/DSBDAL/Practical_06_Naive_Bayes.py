# Practical 6: Gaussian Naive Bayes classification on Iris dataset

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = GaussianNB()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Confusion matrix:\n", confusion_matrix(y_test, pred))
print("Accuracy:", accuracy_score(y_test, pred))
print("Error rate:", 1 - accuracy_score(y_test, pred))
print("Precision (macro):", precision_score(y_test, pred, average="macro", zero_division=0))
print("Recall (macro):", recall_score(y_test, pred, average="macro", zero_division=0))

print("\nClass names:", list(iris.target_names))
print("\nPractical 6 completed.")
