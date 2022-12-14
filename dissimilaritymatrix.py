from sklearn.manifold import MDS
import numpy as np

x = np.array([[0,0,0],[0,0,1],[1,1,1],[0,1,0],[0,1,1]])

mds = MDS(random_state = 0)

xtransform = mds.fit_transform(x)

print(xtransform)
