import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

train = pd.read_csv(r"C:\neurogen\output\train_400.csv")
test  = pd.read_csv(r"C:\neurogen\output\test_100.csv")

X_train = train.drop("Label", axis=1)
y_train = train["Label"]

X_test = test.drop("Label", axis=1)
y_test = test["Label"]


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Get probability for genuine class
probs = knn.predict_proba(X_test)[:, 1]


THRESHOLD = 0.75

y_pred = (probs >= THRESHOLD).astype(int)


cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()


accuracy = (TP + TN) / (TP + TN + FP + FN)
FAR = FP / (FP + TN) if (FP + TN) > 0 else 0
FRR = FN / (FN + TP) if (FN + TP) > 0 else 0

print("Threshold:", THRESHOLD)
print("Confusion Matrix:\n", cm)
print(f"Accuracy: {accuracy:.2f}")
print(f"FAR: {FAR:.2f}")
print(f"FRR: {FRR:.2f}")
