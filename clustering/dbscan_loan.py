import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

path = "/home/arpit-parekh/Downloads/bascom_projects/atul_datascience_roadmap/clustering/loan_data.csv"

df = pd.read_csv(path)
print(df.head())
print(df.info())

"""
 #   Column                            Non-Null Count  Dtype
---  ------                            --------------  -----
 0   CUST_ID                           8950 non-null   object
 1   BALANCE                           8950 non-null   float64
 2   BALANCE_FREQUENCY                 8950 non-null   float64
 3   PURCHASES                         8950 non-null   float64
 4   ONEOFF_PURCHASES                  8950 non-null   float64
 5   INSTALLMENTS_PURCHASES            8950 non-null   float64
 6   CASH_ADVANCE                      8950 non-null   float64
 7   PURCHASES_FREQUENCY               8950 non-null   float64
 8   ONEOFF_PURCHASES_FREQUENCY        8950 non-null   float64
 9   PURCHASES_INSTALLMENTS_FREQUENCY  8950 non-null   float64
 10  CASH_ADVANCE_FREQUENCY            8950 non-null   float64
 11  CASH_ADVANCE_TRX                  8950 non-null   int64
 12  PURCHASES_TRX                     8950 non-null   int64
 13  CREDIT_LIMIT                      8949 non-null   float64
 14  PAYMENTS                          8950 non-null   float64
 15  MINIMUM_PAYMENTS                  8637 non-null   float64
 16  PRC_FULL_PAYMENT                  8950 non-null   float64
 17  TENURE                            8950 non-null   int64
"""

df.fillna(0,inplace=True)

X = df.drop(['CUST_ID'],axis=1)

# scale the data
scaler = StandardScaler()
X = scaler.fit_transform(X)
# create a DBSCAN model

dbscan = DBSCAN(eps=2.0, min_samples=5)
dbscan.fit(X)
# get the cluster labels
labels = dbscan.labels_
print(labels)
# plot the clusters
plt.scatter(X[:, 0], X[:, 13], c=labels, cmap='viridis')
plt.xlabel('BALANCE')
plt.ylabel('TENURE')
plt.title('DBSCAN Clustering')
plt.colorbar()
plt.show()

# create virtual histogram
# plt.hist(labels, bins=10)
# plt.xlabel('Cluster Labels')
# plt.ylabel('Frequency')
# plt.title('DBSCAN Clustering')
# plt.show()
