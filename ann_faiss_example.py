import faiss
import numpy as np

# Create dummy data: 10,000 vectors, each of dimension 128
d = 128
nb = 10000
np.random.seed(123)
xb = np.random.random((nb, d)).astype('float32')

# Build index
index = faiss.IndexFlatL2(d)  # L2 = Euclidean distance
index.add(xb)

# Query with 5 random vectors
xq = np.random.random((5, d)).astype('float32')
D, I = index.search(xq, k=3)  # Search top 3 nearest neighbors

print("Nearest neighbor indices:\n", I)
print("Distances:\n", D)
