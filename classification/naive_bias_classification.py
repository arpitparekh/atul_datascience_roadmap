# /home/arpit-parekh/Downloads/archive(29)/spam_dataset.csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from matplotlib.pylab  import plt

df = pd.read_csv('/home/arpit-parekh/Downloads/archive(29)/spam_dataset.csv')
print(df.head())
print(df.info())

"""
 #   Column    Non-Null Count  Dtype
---  ------    --------------  -----
 0   Category  5572 non-null   object
 1   Message   5572 non-null   object
"""
df['Category'] = LabelEncoder().fit_transform(df['Category'])
vector = CountVectorizer()

# paragraph to array
"""
[[0 0 1 0 0 0 0 1 1 0 1 2]
 [0 0 0 0 1 0 1 0 1 0 1 2]
 [1 1 0 1 0 1 0 0 0 1 0 0]]
"""
X = vector.fit_transform(df['Message'])

X = X.toarray()

y = df['Category']

model = MultinomialNB()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

plt.bar(df['Category'].unique(), df['Category'].value_counts())
plt.xlabel('Category')
plt.ylabel('Count')
plt.title('Spam Dataset')
plt.show()

# testing by user data
message = f"20% descount on your purchase"
message = vector.transform([message])
print(model.predict(message))
