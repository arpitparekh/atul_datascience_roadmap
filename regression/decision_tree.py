"""
Decision Tree

Decision trees are a non-parametric supervised learning method used for classification and regression.
The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.
Common Application Areas:

Medical diagnosis (patient symptoms → conditions)
Risk assessment (credit scoring, insurance)
Customer segmentation
Fraud detection
Environmental studies
Quality control in manufacturing

"""
import numpy as np
import pandas as pd
# decision tree regression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

path = "/home/arpit-parekh/Downloads/archive(20)/seattle-weather.csv"
df = pd.read_csv(path)
print(df.head())
print(df.info())

"""
Data columns (total 6 columns):
#   Column         Non-Null Count  Dtype
---  ------         --------------  -----
0   date           1461 non-null   object
1   precipitation  1461 non-null   float64
2   temp_max       1461 non-null   float64
3   temp_min       1461 non-null   float64
4   wind           1461 non-null   float64
5   weather        1461 non-null   object
"""

df['date'] = pd.to_datetime(df['date'])
df['weather'] = LabelEncoder().fit_transform(df['weather']).astype(float)

print(df.head())
print(df.info())

X = df.drop(['weather','date'], axis=1)
y = df['weather']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# print("Accuracy:", accuracy_score(y_test, y_pred))

plot_tree(model)

# plot the data in tree like structure
plt.figure(figsize=(20,10))
# plt.show()
plt.scatter(X_test['precipitation'], y_test, color='red')
# plt.plot(X_test['precipitation'], y_pred, color='blue')
plt.title('Decision Tree Regression')
plt.xlabel('temp_max')
plt.ylabel('Weather')
plt.show()
plt.show()
