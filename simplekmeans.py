import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data = pd.read_csv('Mall_Customers.csv')

x = data.iloc[:, [3,4]].values

li = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, random_state=42)
    kmeans.fit(x)
    li.append(kmeans.inertia_)


plt.plot(range(1,11),li)
plt.title('The elbow method graph')
plt.xlabel('number of clusters')
plt.ylabel('list')
plt.show()


kmeans = KMeans(n_clusters = 5, random_state = 42)
ypred = kmeans.fit_predict(x)

plt.scatter(x[ypred == 0, 0], x[ypred == 0, 1], s = 100, c = 'blue', label = 'c1')
plt.scatter(x[ypred == 1, 0], x[ypred == 1, 1], s = 100, c = 'green', label = 'c2')
plt.scatter(x[ypred == 2, 0], x[ypred == 2, 1], s = 100, c = 'red', label = 'c3')
plt.scatter(x[ypred == 3, 0], x[ypred == 3, 1], s = 100, c = 'cyan', label = 'c4')
plt.scatter(x[ypred == 4, 0], x[ypred == 4, 1], s = 100, c = 'magenta', label = 'c5')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = "yellow", label = 'Centroid')

plt.title('cluster of customers')
plt.xlabel('annual incomes')
plt.ylabel('spending score(1 - 100)')
plt.legend()
plt.show()
