from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

# KNN using KD-Tree
knn_kdtree = KNeighborsClassifier(n_neighbors=3, algorithm='kd_tree')
start = time.time()
knn_kdtree.fit(X_train, y_train)
preds = knn_kdtree.predict(X_test)
end = time.time()

# Results
print(f"Accuracy: {accuracy_score(y_test, preds)}")
print(f"Inference time: {end - start:.6f} sec")
