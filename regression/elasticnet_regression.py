# regression
# elastic net
# is combination of ridge and lasso regression

import kagglehub

# Download latest version
path = kagglehub.dataset_download("stealthtechnologies/predict-student-performance-dataset")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv("/home/arpit-parekh/Downloads/archive(18)/data.csv")
print(df.head())
print(df.info())
"""

Data columns (total 5 columns):
 #   Column               Non-Null Count  Dtype
---  ------               --------------  -----
 0   Socioeconomic Score  1388 non-null   float64
 1   Study Hours          1388 non-null   float64
 2   Sleep Hours          1388 non-null   float64
 3   Attendance (%)       1388 non-null   float64
 4   Grades               1388 non-null   float64

"""

X = df.drop("Grades", axis=1)
y = df["Grades"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of ElasticNet
model = ElasticNet(l1_ratio=0.5, alpha=0.1)
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)



"""
  Socioeconomic Score  1388 non-null   float64
 1   Study Hours          1388 non-null   float64
 2   Sleep Hours          1388 non-null   float64
 3   Attendance (%)       1388 non-null   float64

"""

# take value from user input
socio_score = float(input("Enter Socioeconomic Score: "))
study_hours = float(input("Enter Study Hours: "))
sleep_hours = float(input("Enter Sleep Hours: "))
attendance = float(input("Enter Attendance (%): "))

# create a new dataframe with user input
df = pd.DataFrame({"Socioeconomic Score": [socio_score], "Study Hours": [study_hours], "Sleep Hours": [sleep_hours], "Attendance (%)": [attendance]})

prediction =  model.predict(df)
print("Predicted Grades:", prediction[0])

plt.scatter(X_test["Socioeconomic Score"], y_test, color="black")
plt.scatter(X_test["Socioeconomic Score"], y_pred, color="red")
plt.scatter(df["Socioeconomic Score"], prediction, color="yellow")
plt.xlabel("Socioeconomic Score")
plt.ylabel("Grades")
plt.show()
