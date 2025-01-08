import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns

# Load the dataset
path = "/home/arpit-parekh/Downloads/bascom_projects/atul_datascience_roadmap/classification/diabetes_dataset.csv"
df = pd.read_csv(path)

# Print the first few rows and data types
print(df.head())
print(df.info())

# Convert relevant columns to float
df["Pregnancies"] = df["Pregnancies"].astype(float)
df["Glucose"] = df["Glucose"].astype(float)
df["BloodPressure"] = df["BloodPressure"].astype(float)
df["SkinThickness"] = df["SkinThickness"].astype(float)
df["Insulin"] = df["Insulin"].astype(float)
df["Age"] = df["Age"].astype(float)
df["Outcome"] = df["Outcome"].astype(float)

# Print the updated data types
print(df.info())

# Separate features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
print(y_pred)
print("Accuracy Score is : ", accuracy_score(y_test, y_pred) * 100, "%")

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix as a bar plot
labels = ["No Diabetes", "Diabetes"]
categories = ["Predicted No Diabetes", "Predicted Diabetes"]

# Create a DataFrame for the confusion matrix
cm_df = pd.DataFrame(cm, index=labels, columns=categories)

# Plot the bar plot
plt.figure(figsize=(10, 6))
cm_df.plot(kind="bar", stacked=True, color=["blue", "orange"], alpha=0.7)
plt.xlabel("Actual")
plt.ylabel("Count")
plt.title("Confusion Matrix")
plt.xticks(rotation=0)
plt.legend(title="Predicted", loc="upper right")
plt.show()
