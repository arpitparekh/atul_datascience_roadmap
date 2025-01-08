path = "/home/arpit-parekh/Downloads/bascom_projects/atul_datascience_roadmap/classification/diabetes_dataset.csv"

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression    # Logistic regression
from sklearn.metrics import accuracy_score,confusion_matrix, roc_curve, auc
from matplotlib import pyplot as plt
import seaborn as sns


df = pd.read_csv(path)
print(df.head())
print(df.info())
df["Pregnancies"] = df["Pregnancies"].astype(float)
df["Glucose"] = df["Glucose"].astype(float)
df["BloodPressure"] = df["BloodPressure"].astype(float)
df["SkinThickness"] = df["SkinThickness"].astype(float)
df["Insulin"] = df["Insulin"].astype(float)
df["Age"] = df["Age"].astype(float)
df["Outcome"] = df["Outcome"].astype(float)
print(df.info())

"""
 #   Column                    Non-Null Count  Dtype
---  ------                    --------------  -----
 0   Pregnancies               768 non-null    float64
 1   Glucose                   768 non-null    float64
 2   BloodPressure             768 non-null    float64
 3   SkinThickness             768 non-null    float64
 4   Insulin                   768 non-null    float64
 5   BMI                       768 non-null    float64
 6   DiabetesPedigreeFunction  768 non-null    float64
 7   Age                       768 non-null    float64
 8   Outcome                   768 non-null    float64
"""

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred =  model.predict(X_test)
print(y_pred)
print("Accurecy Score is : ", accuracy_score(y_test, y_pred)*100,"%")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Diabetes", "Diabetes"], yticklabels=["No Diabetes", "Diabetes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
