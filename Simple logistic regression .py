import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    'price': [30000, 45000, 50000, 70000, 28000, 26000, 60000, 52000, 31000, 100000],
    'age': [1, 3, 2, 4, 1, 5, 2, 3, 4, 6],
    'mileage': [10000, 30000, 20000, 40000, 15000, 50000, 10000, 30000, 25000, 60000],
    'purchased': [1, 1, 0, 0, 1, 0, 1, 0, 0, 0] 
}

df = pd.DataFrame(data)
X = df[['price', 'age', 'mileage']]
y = df['purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')
