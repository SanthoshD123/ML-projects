import pandas as pd

df = pd.read_csv('C:\\Users\\User\\PycharmProjects\\breast cancer prediction\\data.csv')  # Replace with your actual file path

X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

X = X.fillna(X.median())
y = y.map({'M': 1, 'B': 0})
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42) #used random forest classifier for better prediction
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
