import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


path = "/home/arpit-parekh/Downloads/archive(13)/Advertising.csv"
df = pd.read_csv(path)
print(df.head())
print(df.info())

X = df[['TV','Radio','Newspaper']]
y = df["Sales"]

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model= LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

error = mean_squared_error(y_test,y_pred)      # [actual(observed)-predicted]square/sum of numbers = mean square error
print("Mean Squared Error:",error)

plt.scatter(X_test['Radio'],y_test,color='red')
plt.scatter(X_test['Radio'],y_pred,color='blue')

plt.xlabel('Radio')
plt.ylabel('Sales')
plt.title('Advertisement Prediction')
plt.legend(['Actual','Predicted'])
plt.show()


# take value from user input
print("Enter Tv sales value")
tv_sales = float(input())
print("Enter Radio sales value: ")
radio_sales = float(input())  
print("Enter Newspaper sales value: ")
newspaper_sales = float(input())

predict = model.predict([[tv_sales,radio_sales,newspaper_sales]])
print("Predicted Sales: ",predict)
