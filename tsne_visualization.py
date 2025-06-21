from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
X_embedded = tsne.fit_transform(X)

sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=y, palette="deep")
plt.title("t-SNE Visualization of Iris Dataset")
plt.show()
