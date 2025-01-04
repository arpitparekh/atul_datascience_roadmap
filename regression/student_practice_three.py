import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from  sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# label encoding
from sklearn.preprocessing import LabelEncoder

path = "/home/arpit-parekh/Downloads/archive(19)/loan_approval_dataset.csv"
df = pd.read_csv(path)
print(df.head())
print(df.info())
"""
#   Column                    Non-Null Count  Dtype
---  ------                    --------------  -----
 0   loan_id                   4269 non-null   int64  ignore
 1   no_of_dependents          4269 non-null   int64  ignore
 2   education                 4269 non-null   int64
 3   self_employed             4269 non-null   int64
 4   income_annum              4269 non-null   int64  ignore
 5   loan_amount               4269 non-null   int64
 6   loan_term                 4269 non-null   int64  ignore
 7   cibil_score               4269 non-null   int64  ignore
 8   residential_assets_value  4269 non-null   int64  ignore
 9   commercial_assets_value   4269 non-null   int64  ignore
 10  luxury_assets_value       4269 non-null   int64  ignore
 11  bank_asset_value          4269 non-null   int64  ignore
 12  loan_status               4269 non-null   int64
dtypes: int64(13)
"""
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

encoder = LabelEncoder()
df['self_employed'] = encoder.fit_transform(df['self_employed'])
df['education'] = encoder.fit_transform(df['education'])
df['loan_status'] = encoder.fit_transform(df['loan_status'])

print(df.head())
print(df.info())

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

encoder = LabelEncoder()
df['self_employed'] = encoder.fit_transform(df['self_employed'])
df['education'] = encoder.fit_transform(df['education'])
df['loan_status'] = encoder.fit_transform(df['loan_status'])

# create a new dataframe with user input
df = pd.DataFrame({
    "no_of_dependents": ['no_of_dependents'],
    "education": ['education'],
    "self_employed": ['self_employed'],
    "income_annum": ['income_annum'],
    "loan_amount": ['loan_amount'],
    "loan_term": ['loan_term'],
    "cibil_score": ['cibil_score'],
    "residential_assets_value": ['residential_assets_value'],
    "commercial_assets_value": ['commercial_assets_value'],
    "luxury_assets_value": ['luxury_assets_value'],
    "bank_asset_value": ['bank_asset_value']
})

prediction = model.predict(df)
print("Predicted loan_status:", prediction[0])

plt.scatter(X_test["loan_amount"], y_test, color="black")
plt.scatter(X_test["loan_amount"], y_pred, color="red")
plt.scatter(df["loan_amount"], prediction, color="yellow")
plt.xlabel("loan_amount")
plt.ylabel("loan_status")
plt.show()
