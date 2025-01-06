path = "/home/arpit-parekh/Downloads/archive(21)/Toyota_Data.csv"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv(path)
print(df.head())
print(df.info())

"""
Data columns (total 7 columns):
#   Column     Non-Null Count  Dtype
---  ------     --------------  -----
0   Date       11291 non-null  object   // ignore
1   Adj Close  11291 non-null  float64
2   Close      11291 non-null  float64
3   High       11291 non-null  float64
4   Low        11291 non-null  float64
5   Open       11291 non-null  float64
6   Volume     11291 non-null  int64
"""

df['Volume'] = df['Volume'].astype(float)

X = df.drop(["Date","High"], axis=1)
y = df["High"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# print(accuracy_score(y_test, y_pred))

plt.scatter(X["Low"],y,c="red")
# plt.plot(X_test["Low"],y_pred,c="blue")
plt.show()
