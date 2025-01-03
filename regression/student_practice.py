# Ridge regression practice

path = "/home/arpit-parekh/Downloads/archive(19)/loan_approval_dataset.csv"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# label encoding
from sklearn.preprocessing import LabelEncoder

"""
#   Column                    Non-Null Count  Dtype
---  ------                    --------------  -----
0   loan_id                   4269 non-null   int64   ignore
1   no_of_dependents          4269 non-null   int64
2   education                 4269 non-null   int64
3   self_employed             4269 non-null   int64
4   income_annum              4269 non-null   int64
5   loan_amount               4269 non-null   int64  result
6   loan_term                 4269 non-null   int64
7   cibil_score               4269 non-null   int64
8   residential_assets_value  4269 non-null   int64
9   commercial_assets_value   4269 non-null   int64
10  luxury_assets_value       4269 non-null   int64
11  bank_asset_value          4269 non-null   int64
12  loan_status               4269 non-null   int64

"""

# Reading the data
df = pd.read_csv(path)
print(df.head())
print(df.info())

encoder = LabelEncoder()
df['self_employed'] = encoder.fit_transform(df['self_employed'])
df['education'] = encoder.fit_transform(df['education'])
df['loan_status'] = encoder.fit_transform(df['loan_status'])

print(df.head())
print(df.info())

X = df.drop(['loan_id','loan_amount'], axis=1)
y = df['loan_amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ridge = Ridge(alpha=0.5)
ridge.fit(X_train, y_train)

y_pred = ridge.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)

# plot the data

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Loan Approval')
plt.tight_layout()
plt.show()



"""

# Add these plotting code after the existing code

# 1. Actual vs Predicted Values Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.tight_layout()
plt.show()

# 2. Feature Importance Plot
feature_importance = pd.DataFrame({'Feature': X.columns, 'Coefficient': ridge.coef_})
feature_importance = feature_importance.sort_values('Coefficient', key=abs, ascending=False)

plt.figure(figsize=(12, 6))
plt.bar(range(len(feature_importance)), abs(feature_importance['Coefficient']))
plt.xticks(range(len(feature_importance)), feature_importance['Feature'], rotation=90)
plt.xlabel('Features')
plt.ylabel('Absolute Coefficient Value')
plt.title('Feature Importance in Ridge Regression')
plt.tight_layout()
plt.show()

# 3. Residuals Plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs Predicted Price')
plt.tight_layout()
plt.show()


"""
