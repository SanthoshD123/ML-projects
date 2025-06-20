from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

data = load_iris()
X = MinMaxScaler().fit_transform(data.data)

# Initialize SOM
som = MiniSom(x=7, y=7, input_len=4, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(X, 100)

# Plot
plt.bone()
for i, x in enumerate(X):
    w = som.winner(x)
    plt.text(w[0], w[1], str(data.target[i]), ha='center', va='center',
             bbox=dict(facecolor='white', alpha=0.5, lw=0))
plt.title("Self-Organizing Map - Iris Dataset")
plt.show()
