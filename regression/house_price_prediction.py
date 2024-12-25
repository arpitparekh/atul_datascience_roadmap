import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

path = "/home/arpit-parekh/Downloads/bascom_projects/atul_datascience_roadmap/datasets/Real estate.csv"

df = pd.read_csv(path)
print(df.head())
print(df.info())

X = df[['X2 house age','X3 distance to the nearest MRT station','X4 number of convenience stores']]
y = df['Y house price of unit area']

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model= LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

error = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", error)

plt.scatter(X_test['X3 distance to the nearest MRT station'], y_test, color='red')
plt.scatter(X_test['X3 distance to the nearest MRT station'], y_pred, color='blue')

plt.xlabel('X3 distance to the nearest MRT station')
plt.ylabel('Y house price of unit area')
plt.title('Linear Regression')
plt.legend(['Actual', 'Predicted'])
plt.show()
