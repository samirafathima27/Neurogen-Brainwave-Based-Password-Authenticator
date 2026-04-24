import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

train = pd.read_csv(r"C:\neurogen\output\train_v2.csv")
test  = pd.read_csv(r"C:\neurogen\output\test_v2.csv")

X_train = train.drop("Label", axis=1)
y_train = train["Label"]

X_test = test.drop("Label", axis=1)
y_test = test["Label"]

svm = SVC(kernel="rbf")
svm.fit(X_train, y_train)

pred = svm.predict(X_test)
print("SVM v2 Accuracy:", accuracy_score(y_test, pred))
