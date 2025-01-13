# different classification in python
# Random Forest Classification
path = "/home/arpit-parekh/Downloads/archive(27)/Iris.csv"
from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import seaborn as sns

# load dataset
df = read_csv(path)


df['Species'] = LabelEncoder().fit_transform(df['Species'])
print(df.head())
print(df.info())

X = df.drop(['Species','Id'], axis=1)
y = df['Species']

# split dataset
#into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# make predictions on test set
y_pred = model.predict(X_test)
# evaluate accuracy

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# evaluate classification report
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix as a bar plot
labels = ["Setosa", "Versicolor", "Virginica"]
categories = ["Predicted Setosa", "Predicted Versicolor", "Predicted Virginica"]
plt.figure(figsize=(8, 6))
cm_df = pd.DataFrame(cm, index=labels, columns=categories)
sns.heatmap(cm_df, annot=True, cmap="YlGnBu", fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
