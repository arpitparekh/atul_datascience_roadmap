path = "/home/arpit-parekh/Downloads/archive(17)/Mumbai.csv"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# label encoding
from sklearn.preprocessing import LabelEncoder


# Reading the data
df = pd.read_csv(path)
print(df.head())

"""
 #   Column               Non-Null Count  Dtype
---  ------               --------------  -----
 0   Price                7719 non-null   int64
 1   Area                 7719 non-null   int64
 2   Location             7719 non-null   object
 3   No. of Bedrooms      7719 non-null   int64
 4   Resale               7719 non-null   int64
 5   MaintenanceStaff     7719 non-null   int64
 6   Gymnasium            7719 non-null   int64
 7   SwimmingPool         7719 non-null   int64
 8   LandscapedGardens    7719 non-null   int64
 9   JoggingTrack         7719 non-null   int64
 10  RainWaterHarvesting  7719 non-null   int64
 11  IndoorGames          7719 non-null   int64
 12  ShoppingMall         7719 non-null   int64
 13  Intercom             7719 non-null   int64
 14  SportsFacility       7719 non-null   int64
 15  ATM                  7719 non-null   int64
 16  ClubHouse            7719 non-null   int64
 17  School               7719 non-null   int64
 18  24X7Security         7719 non-null   int64
 19  PowerBackup          7719 non-null   int64
 20  CarParking           7719 non-null   int64
 21  StaffQuarter         7719 non-null   int64
 22  Cafeteria            7719 non-null   int64
 23  MultipurposeRoom     7719 non-null   int64
 24  Hospital             7719 non-null   int64
 25  WashingMachine       7719 non-null   int64
 26  Gasconnection        7719 non-null   int64
 27  AC                   7719 non-null   int64
 28  Wifi                 7719 non-null   int64
 29  Children'splayarea   7719 non-null   int64
 30  LiftAvailable        7719 non-null   int64
 31  BED                  7719 non-null   int64
 32  VaastuCompliant      7719 non-null   int64
 33  Microwave            7719 non-null   int64
 34  GolfCourse           7719 non-null   int64
 35  TV                   7719 non-null   int64
 36  DiningTable          7719 non-null   int64
 37  Sofa                 7719 non-null   int64
 38  Wardrobe             7719 non-null   int64
 39  Stadium              7719 non-null   int64
"""

df['Location'] = LabelEncoder().fit_transform(df['Location'])

print(df.info())

X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ridge = Ridge()
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
plt.title('Actual vs Predicted House Prices')
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
