# random forest regression

path = "/home/arpit-parekh/Downloads/archive(22)/Life Expectancy Data.csv"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# load the data
df = pd.read_csv(path)

df.dropna(inplace=True)
print(df.head())
print(df.info())

"""
Data columns (total 22 columns):
#   Column                           Non-Null Count  Dtype
---  ------                           --------------  -----
0   Country                          1649 non-null   object  // drop
1   Year                             1649 non-null   int64   // drop
2   Status                           1649 non-null   object   // drop
3   Life expectancy                  1649 non-null   float64  // predict
4   Adult Mortality                  1649 non-null   float64
5   infant deaths                    1649 non-null   int64
6   Alcohol                          1649 non-null   float64
7   percentage expenditure           1649 non-null   float64
8   Hepatitis B                      1649 non-null   float64
9   Measles                          1649 non-null   int64
10   BMI                             1649 non-null   float64
11  under-five deaths                1649 non-null   int64
12  Polio                            1649 non-null   float64
13  Total expenditure                1649 non-null   float64
14  Diphtheria                       1649 non-null   float64
15   HIV/AIDS                        1649 non-null   float64
16  GDP                              1649 non-null   float64
17  Population                       1649 non-null   float64
18   thinness  1-19 years            1649 non-null   float64
19   thinness 5-9 years              1649 non-null   float64
20  Income composition of resources  1649 non-null   float64
21  Schooling                        1649 non-null   float64
"""

df["infant deaths"] = df["infant deaths"].astype(float)
df["Measles"] = df["Measles"].astype(float)
df["under-five deaths"] = df["under-five deaths"].astype(float)

print(df.info())


X = df.drop(['Year', 'Status', 'Life expectancy','Country'], axis=1)
y = df['Life expectancy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


plt.scatter(X_test["BMI"], y_test, color='blue')
plt.scatter(X_test["BMI"], y_pred, color='red')

plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

# take large data for prediction
adult_mortality = float(input("Enter Adult Mortality: "))
infant_deaths = float(input("Enter infant deaths: "))
alcohol = float(input("Enter Alcohol: "))
percentage_expenditure = float(input("Enter percentage expenditure: "))
hepatitis_b = float(input("Enter Hepatitis B: "))
measles = float(input("Enter Measles: "))
bmi = float(input("Enter BMI: "))
under_five_deaths = float(input("Enter under-five deaths: "))
polio = float(input("Enter Polio: "))
total_expenditure = float(input("Enter Total expenditure: "))
diphtheria = float(input("Enter Diphtheria: "))
hiv_aids = float(input("Enter HIV/AIDS: "))
gdp = float(input("Enter GDP: "))
population = float(input("Enter Population: "))
thinness_1_19_years = float(input("Enter thinness 1-19 years: "))
thinness_5_9_years = float(input("Enter thinness 5-9 years: "))
income_composition_of_resources = float(input("Enter income composition of resources: "))
schooling = float(input("Enter Schooling: "))

userInput = pd.DataFrame({
    'Adult Mortality': [adult_mortality],
    'infant deaths': [infant_deaths],
    'Alcohol': [alcohol],
    'percentage expenditure': [percentage_expenditure],
    'Hepatitis B': [hepatitis_b],
    'Measles': [measles],
    'BMI': [bmi],
    'under-five deaths': [under_five_deaths],
    'Polio': [polio],
    'Total expenditure': [total_expenditure],
    'Diphtheria': [diphtheria],
    'HIV/AIDS': [hiv_aids],
    'GDP': [gdp],
    'Population': [population],
    'thinness  1-19 years': [thinness_1_19_years],
    'thinness 5-9 years': [thinness_5_9_years],
    'Income composition of resources': [income_composition_of_resources],
    'Schooling': [schooling]
})

userPrediction = model.predict(userInput)
print("Predicted Life Expectancy: ", userPrediction[0])



