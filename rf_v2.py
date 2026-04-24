import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

train = pd.read_csv(r"C:\neurogen\output\train_v2.csv")
test  = pd.read_csv(r"C:\neurogen\output\test_v2.csv")

X_train = train.drop("Label", axis=1)
y_train = train["Label"]

X_test = test.drop("Label", axis=1)
y_test = test["Label"]

rf = RandomForestClassifier(n_estimators=150, random_state=42)
rf.fit(X_train, y_train)

pred = rf.predict(X_test)
print("RF v2 Accuracy:", accuracy_score(y_test, pred))
